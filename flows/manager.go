package flows

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/utils/events"
	"github.com/google/uuid"
)

// EventNodeEntered fires each time the flow enters a node, including the first
// node and a node it re-enters, with a NodeEntered. Handlers run in order
// before the node's LLM response starts, so they must be fast; start a
// goroutine for slow work.
//
//	events.On(fm.Events(), flows.EventNodeEntered,
//	    func(ctx context.Context, entered flows.NodeEntered) { … })
const EventNodeEntered = "on_node_entered"

// placeholder is a {{ key }} or {{ key.sub.key }} placeholder in a node's
// prompt text. Only identifiers and dots are accepted, so prose braces are left
// alone. A leading backslash escapes it: \{{ key }} renders as {{ key }}.
var placeholder = regexp.MustCompile(
	`(\\?)\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\}\}`)

// acknowledged is the placeholder tool result used for a function that
// transitioned without a result of its own. The model called it and is owed an
// answer either way.
const acknowledged = `{"status": "acknowledged"}`

// summaryTimeout bounds how long a transition waits for the conversation
// summary the reset-with-summary strategy needs. A transition that waits longer
// than this falls back to appending, which keeps the conversation whole.
const summaryTimeout = 5 * time.Second

// Enqueuer injects frames into the running pipeline; *pipeline.Worker satisfies
// it. Everything a flow does to the conversation it does by queueing a frame, so
// the change is ordered against whatever else is in flight.
type Enqueuer interface {
	QueueFrame(f frames.Frame, dir ...processor.Direction)
	QueueFrames(fs []frames.Frame, dir ...processor.Direction)
}

// Config configures a FlowManager. The references are the same instances wired
// into the pipeline.
type Config struct {
	// Enqueuer queues the frames a flow produces; usually the *pipeline.Worker.
	Enqueuer Enqueuer
	// Watcher reports frames reaching the end of the pipeline, which is how an
	// action learns it has finished; usually the same *pipeline.Worker. Leaving it
	// nil gives up the actions that wait on the pipeline.
	Watcher Watcher
	// Aggregators are the conversation aggregators the pipeline shares. The flow
	// reads the conversation through them, and asks the assistant side whether a
	// turn's tool calls have all reported before it transitions.
	Aggregators *aggregators.Pair
	// LLM answers the one-shot inference the reset-with-summary strategy needs. It
	// is only required by that strategy; the concrete LLM services satisfy it.
	LLM Inferencer
	// ContextStrategy is what happens to the conversation when a node is entered.
	// Nil appends, keeping everything said so far.
	ContextStrategy *ContextStrategyConfig
	// GlobalFunctions are offered at every node, ahead of the node's own.
	GlobalFunctions []NodeFunction
}

// FlowManager drives a conversation through a graph of nodes.
//
// It is not a pipeline processor. It holds the conversation and the task and
// steers the conversation by queueing frames as nodes are entered: the persona,
// the node's objective and its toolset. The LLM service's tool loop then carries
// out each transition. Build one with New and enter the graph with Initialize.
//
// A node's prompts may refer to the flow's state with {{ key }} placeholders,
// in RoleMessage, the text of TaskMessages, and "tts_say" text. They are
// filled in each time the node is entered, so a value a handler stored earlier
// in the conversation can appear in a later prompt, including after a context
// reset. {{ order.size }} walks into a stored map, a key that is missing fails
// the entry with ErrFlow, and \{{ key }} is left as the literal {{ key }}.
//
// An RTVI client is told about each node too, so it can follow the
// conversation through the flow. On entering a node the manager queues an
// rtvi.FlowNodeEnteredFrame, which reaches the client as a flow-node-entered
// message carrying the fields of NodeEntered; a field with no value is left
// out. The message is sent to RTVI clients of protocol 2.4.0 or later, and to
// any client before it sends client-ready, so keep anything the end user
// should not see out of node names, function names and metadata. Metadata that
// cannot be written as a JSON object is left out of the message, with a
// warning; the handlers of EventNodeEntered still receive it.
type FlowManager struct {
	events events.Registry

	enq      Enqueuer
	aggs     *aggregators.Pair
	llm      Inferencer
	strategy ContextStrategyConfig
	globals  []NodeFunction
	actions  *actionManager
	adapter  summaryAdapter

	mu          sync.Mutex
	initialized bool
	current     string
	currentFns  map[string]bool
	// currentFnMetadata is the metadata of the current node's functions that
	// have any, by function name.
	currentFnMetadata map[string]map[string]any
	// pending holds a transition waiting for the turn's tool calls to finish. It
	// is set by an edge function and taken by the context-updated callback.
	pending *pendingTransition

	stateMu sync.Mutex
	state   map[string]any
}

// pendingTransition is a transition an edge function asked for, held until the
// turn's calls have all reported.
type pendingTransition struct {
	next     *NodeConfig
	function string
}

// New builds a FlowManager from cfg.
func New(cfg Config) (*FlowManager, error) {
	if cfg.Enqueuer == nil {
		return nil, errf(ErrFlow, "an Enqueuer is required")
	}
	if cfg.ContextStrategy != nil {
		if err := cfg.ContextStrategy.Validate(); err != nil {
			return nil, err
		}
	}
	for _, fn := range cfg.GlobalFunctions {
		if err := validateFunction(fn); err != nil {
			return nil, err
		}
	}

	strategy := ContextStrategyConfig{Strategy: ContextStrategyAppend}
	if cfg.ContextStrategy != nil {
		strategy = *cfg.ContextStrategy
	}

	fm := &FlowManager{
		enq:        cfg.Enqueuer,
		aggs:       cfg.Aggregators,
		llm:        cfg.LLM,
		strategy:   strategy,
		globals:    cfg.GlobalFunctions,
		currentFns: make(map[string]bool),
		state:      make(map[string]any),
	}
	fm.actions = newActionManager(cfg.Enqueuer, cfg.Watcher, fm)
	fm.events.Register(EventNodeEntered, true)
	return fm, nil
}

// Events is the registry the manager raises its events on: see
// EventNodeEntered.
func (fm *FlowManager) Events() *events.Registry { return &fm.events }

// State is the data a flow keeps across nodes: what the assistant has gathered
// so far, and anything else the handlers need to share. It is safe for
// concurrent use.
func (fm *FlowManager) State() *State { return &State{fm: fm} }

// State reads and writes a flow's shared data.
type State struct{ fm *FlowManager }

// Get returns the value stored under key, and whether there was one.
func (s *State) Get(key string) (any, bool) {
	s.fm.stateMu.Lock()
	defer s.fm.stateMu.Unlock()
	v, ok := s.fm.state[key]
	return v, ok
}

// Set stores value under key.
func (s *State) Set(key string, value any) {
	s.fm.stateMu.Lock()
	s.fm.state[key] = value
	s.fm.stateMu.Unlock()
}

// Delete removes key.
func (s *State) Delete(key string) {
	s.fm.stateMu.Lock()
	delete(s.fm.state, key)
	s.fm.stateMu.Unlock()
}

// CurrentNode returns the name of the node the flow is on, or "" before
// Initialize.
func (fm *FlowManager) CurrentNode() string {
	fm.mu.Lock()
	defer fm.mu.Unlock()
	return fm.current
}

// CurrentContext returns the conversation as it stands.
func (fm *FlowManager) CurrentContext() ([]frames.Message, error) {
	if fm.aggs == nil {
		return nil, errf(ErrFlow, "no aggregators available")
	}
	return fm.aggs.Context().Messages(), nil
}

// RegisterAction records a handler for an action type, so a node may use it in
// its pre- or post-actions. The built-in types ("tts_say", "end_conversation"
// and "function") are registered already.
func (fm *FlowManager) RegisterAction(actionType string, h ActionHandler) error {
	return fm.actions.register(actionType, h)
}

// Initialize enters the flow.
//
// Passing a node starts the conversation there. Passing nil initializes the
// manager without entering a node, for a flow whose opening node is decided
// later and set with SetNode. Initializing twice is reported and otherwise
// ignored.
func (fm *FlowManager) Initialize(ctx context.Context, node *NodeConfig) error {
	fm.mu.Lock()
	if fm.initialized {
		fm.mu.Unlock()
		slog.WarnContext(ctx, "flows: manager is already initialized")
		return nil
	}
	fm.initialized = true
	fm.mu.Unlock()

	slog.DebugContext(ctx, "flows: initialized")

	if node == nil {
		return nil
	}
	if err := fm.setNode(ctx, nodeName(node), node, ""); err != nil {
		fm.mu.Lock()
		fm.initialized = false
		fm.mu.Unlock()
		return wrapErr(ErrFlowInitialization, err)
	}
	return nil
}

// SetNode transitions the flow to node. Enter the flow with Initialize first.
//
// It is the manual transition, for a caller that is not part of the graph: a
// processor moving the conversation on something the model was never asked
// about. A node function transitions by returning the next node instead, which
// keeps the move on the tool loop that made it.
func (fm *FlowManager) SetNode(ctx context.Context, node *NodeConfig) error {
	if node == nil {
		return errf(ErrFlowTransition, "node is nil")
	}
	return fm.setNode(ctx, nodeName(node), node, "")
}

// setNode enters a node, reporting any failure against the node that failed.
// via names the function whose call caused the move, if any.
func (fm *FlowManager) setNode(ctx context.Context, id string, node *NodeConfig, via string) error {
	if err := fm.enterNode(ctx, id, node, via); err != nil {
		return &flowError{kind: ErrFlow, msg: fmt.Sprintf("failed to set node %q", id), cause: err}
	}
	return nil
}

// enterNode does the work of entering a node: it runs the pre-actions, applies
// the persona, the objective and the toolset, calls the EventNodeEntered
// handlers and tells an RTVI client, asks for a response where the node speaks
// on entry, and runs the post-actions.
func (fm *FlowManager) enterNode(ctx context.Context, id string, node *NodeConfig, via string) error {
	fm.mu.Lock()
	initialized := fm.initialized
	// Whatever transition was pending belongs to the node being left. Clearing it
	// here covers every way of arriving: the tool loop, which has already taken
	// it, and a manual transition, which never had one.
	fm.pending = nil
	fm.mu.Unlock()

	if !initialized {
		return errf(ErrFlowTransition, "manager must be initialized first")
	}

	if err := validateNode(id, node); err != nil {
		return err
	}
	node, err := fm.renderNode(id, node)
	if err != nil {
		return err
	}
	slog.DebugContext(ctx, "flows: setting node", "node", id)

	// The post-actions still waiting belong to the node being left.
	fm.actions.clearDeferredPostActions()

	if err = fm.registerActions(node); err != nil {
		return err
	}

	if err = fm.actions.executeActions(ctx, node.PreActions); err != nil {
		return err
	}

	tools, names, metadata, err := fm.nodeTools(node)
	if err != nil {
		return err
	}

	if err := fm.updateContext(ctx, node, tools); err != nil {
		return err
	}

	// via is a function of the node being left, so its metadata is read before
	// that node's functions are replaced.
	fm.mu.Lock()
	entered := NodeEntered{
		Node:         id,
		PreviousNode: fm.current,
		Function:     via,
		Metadata:     node.Metadata,
		Parent:       node.Parent,
	}
	if via != "" {
		entered.FunctionMetadata = fm.currentFnMetadata[via]
	}
	fm.current = id
	fm.currentFns = names
	fm.currentFnMetadata = metadata
	fm.mu.Unlock()

	fm.events.Call(ctx, EventNodeEntered, fm, entered)
	fm.enq.QueueFrame(rtvi.NewFlowNodeEnteredFrame(nodeEnteredData(ctx, entered)))

	respond := node.respondsImmediately()
	if respond {
		fm.enq.QueueFrame(frames.NewLLMRunFrame())
	}

	if len(node.PostActions) > 0 {
		if respond {
			if err := fm.actions.executeActions(ctx, node.PostActions); err != nil {
				return err
			}
		} else {
			// The node waits for the user, so its post-actions wait for the turn that
			// eventually happens rather than running into the silence.
			fm.actions.scheduleDeferredPostActions(node.PostActions)
		}
	}

	slog.DebugContext(ctx, "flows: set node", "node", id)
	return nil
}

// registerActions registers the handlers a node's actions bring with them.
func (fm *FlowManager) registerActions(node *NodeConfig) error {
	for _, actions := range [][]ActionConfig{node.PreActions, node.PostActions} {
		for _, a := range actions {
			if err := fm.actions.registerFromConfig(a); err != nil {
				return err
			}
		}
	}
	return nil
}

// nodeEnteredData is the flow-node-entered message data that announces a node.
// Metadata that cannot be written as a JSON object is left out.
func nodeEnteredData(ctx context.Context, entered NodeEntered) rtvi.FlowNodeEnteredData {
	return rtvi.FlowNodeEnteredData{
		Node:             entered.Node,
		PreviousNode:     entered.PreviousNode,
		Function:         entered.Function,
		Metadata:         asJSONObject(ctx, entered.Metadata, "metadata", entered.Node),
		FunctionMetadata: asJSONObject(ctx, entered.FunctionMetadata, "function_metadata", entered.Node),
		Parent:           entered.Parent,
	}
}

// asJSONObject is value as a client receives it, or nil when it cannot be
// written as a JSON object.
func asJSONObject(ctx context.Context, value map[string]any, field, node string) map[string]any {
	if value == nil {
		return nil
	}
	var sent map[string]any
	raw, err := json.Marshal(value)
	if err == nil {
		err = json.Unmarshal(raw, &sent)
	}
	if err != nil || sent == nil {
		slog.WarnContext(ctx, "flows: not a JSON object, so it is left out of the RTVI message",
			"node", node, "field", field)
		return nil
	}
	return sent
}

// renderNode fills the {{ key }} placeholders in the node's prompts from the
// flow's state: in RoleMessage, the text of each of TaskMessages, and "tts_say"
// action text. {{ order.size }} walks into a stored map, and \{{ key }} is left
// as the literal {{ key }}. Rendering happens on every entry and on a copy, so a
// prompt sees whatever handlers have stored by then and the node given is left
// as it was. A placeholder naming a key that is not in state is an ErrFlow.
func (fm *FlowManager) renderNode(id string, node *NodeConfig) (*NodeConfig, error) {
	var failure error
	render := func(text string) string {
		return placeholder.ReplaceAllStringFunc(text, func(match string) string {
			groups := placeholder.FindStringSubmatch(match)
			if groups[1] != "" {
				return match[1:]
			}
			value, ok := fm.stateValue(groups[2])
			if !ok {
				if failure == nil {
					failure = errf(ErrFlow, fmt.Sprintf(
						"node '%s' uses '{{ %s }}', which is not in state", id, groups[2]))
				}
				return match
			}
			return value
		})
	}

	rendered := *node
	if node.RoleMessage != "" {
		rendered.RoleMessage = render(node.RoleMessage)
	}
	rendered.TaskMessages = make([]frames.Message, len(node.TaskMessages))
	for i, message := range node.TaskMessages {
		if message.Content == nil {
			message.Text = render(message.Text)
		}
		rendered.TaskMessages[i] = message
	}
	renderActions := func(actions []ActionConfig) []ActionConfig {
		if len(actions) == 0 {
			return actions
		}
		out := make([]ActionConfig, len(actions))
		for i, action := range actions {
			if action.Type == actionTTSSay {
				action.Text = render(action.Text)
			}
			out[i] = action
		}
		return out
	}
	rendered.PreActions = renderActions(node.PreActions)
	rendered.PostActions = renderActions(node.PostActions)
	if failure != nil {
		return nil, failure
	}
	return &rendered, nil
}

// stateValue reads the value at a dotted path in the flow's state, written out
// as text. Each part but the last must name a map with string keys.
func (fm *FlowManager) stateValue(path string) (string, bool) {
	fm.stateMu.Lock()
	defer fm.stateMu.Unlock()
	var current any = fm.state
	for part := range strings.SplitSeq(path, ".") {
		v := reflect.ValueOf(current)
		if v.Kind() != reflect.Map || v.Type().Key().Kind() != reflect.String {
			return "", false
		}
		item := v.MapIndex(reflect.ValueOf(part).Convert(v.Type().Key()))
		if !item.IsValid() {
			return "", false
		}
		current = item.Interface()
	}
	return fmt.Sprint(current), true
}

// nodeTools builds the toolset a node advertises: the global functions followed
// by the node's own, each carrying the handler that runs it. It also returns
// their names, and the metadata of those that have any.
func (fm *FlowManager) nodeTools(
	node *NodeConfig,
) ([]frames.Tool, map[string]bool, map[string]map[string]any, error) {
	all := make([]NodeFunction, 0, len(fm.globals)+len(node.Functions))
	all = append(all, fm.globals...)
	all = append(all, node.Functions...)

	tools := make([]frames.Tool, 0, len(all))
	names := make(map[string]bool, len(all))
	metadata := make(map[string]map[string]any)
	for _, fn := range all {
		if err := validateFunction(fn); err != nil {
			return nil, nil, nil, err
		}
		params, err := fn.parameters()
		if err != nil {
			return nil, nil, nil, errf(ErrInvalidFunction,
				fmt.Sprintf("function %q has parameters that cannot be described: %v", fn.Name, err))
		}
		tools = append(tools, frames.Tool{
			Name:                 fn.Name,
			Description:          fn.Description,
			Parameters:           params,
			Handler:              fm.transitionFunc(fn),
			CancelOnInterruption: fn.cancelOnInterruption(),
			TimeoutSecs:          fn.TimeoutSecs,
		})
		names[fn.Name] = true
		if fn.Metadata != nil {
			metadata[fn.Name] = fn.Metadata
		}
	}
	return tools, names, metadata, nil
}

// updateContext queues the frames that make the node current: the persona, the
// node's objective and its toolset, in that order.
func (fm *FlowManager) updateContext(
	ctx context.Context, node *NodeConfig, tools []frames.Tool,
) error {
	out := make([]frames.Frame, 0, 3)

	if node.RoleMessage != "" {
		// The persona is the LLM service's system instruction, so it stands until
		// another node replaces it rather than being one more thing said.
		out = append(out, frames.NewLLMUpdateSettingsFrame(&settings.LLM{
			SystemInstruction: settings.Set(node.RoleMessage),
		}))
	}

	strategy := fm.strategy
	if node.ContextStrategy != nil {
		strategy = *node.ContextStrategy
	}

	messages := make([]frames.Message, 0, len(node.TaskMessages)+1)

	if strategy.Strategy == ContextStrategyResetWithSummary {
		summary := fm.summarize(ctx, strategy.SummaryPrompt)
		if summary != "" {
			messages = append(messages, fm.adapter.formatSummaryMessage(summary))
		} else {
			// Without a summary a reset would throw the conversation away, so the
			// node appends instead and nothing is lost.
			slog.WarnContext(ctx, "flows: no summary produced, appending instead")
			strategy.Strategy = ContextStrategyAppend
		}
	}

	messages = append(messages, node.TaskMessages...)

	var contextFrame frames.Frame
	switch strategy.Strategy {
	case ContextStrategyReset, ContextStrategyResetWithSummary:
		contextFrame = frames.NewLLMMessagesUpdateFrame(messages)
	default:
		contextFrame = frames.NewLLMMessagesAppendFrame(messages)
	}

	// A node's context and tools must land even if the user interrupts while
	// they are queued, or the LLM runs with the previous node's.
	contextFrame.Base().SetInterruptible(false)
	toolsFrame := frames.NewLLMSetToolsFrame(tools)
	toolsFrame.SetInterruptible(false)
	out = append(out, contextFrame, toolsFrame)

	fm.enq.QueueFrames(out)
	slog.DebugContext(ctx, "flows: updated context", "strategy", strategy.Strategy)
	return nil
}

// summarize produces the conversation summary the reset-with-summary strategy
// puts at the head of the new context, or "" when it cannot.
func (fm *FlowManager) summarize(ctx context.Context, prompt string) string {
	if fm.llm == nil || fm.aggs == nil {
		slog.WarnContext(ctx, "flows: reset-with-summary needs an LLM and aggregators")
		return ""
	}
	ctx, cancel := context.WithTimeout(ctx, summaryTimeout)
	defer cancel()
	return fm.adapter.generateSummary(ctx, fm.llm, prompt, fm.aggs.Context())
}

// transitionFunc adapts a node function into the handler the LLM service runs.
//
// It runs the function, reports what it produced, and says what should happen
// next: answer from the result, stay silent, or transition once the turn's calls
// have all reported.
func (fm *FlowManager) transitionFunc(fn NodeFunction) llm.FunctionCallHandler {
	return func(ctx context.Context, params llm.FunctionCallParams) error {
		slog.DebugContext(ctx, "flows: function called", "function", fn.Name)

		result, next, err := fn.Handler(ctx, params.Arguments, fm)
		if err != nil {
			// The model called the function and is owed an answer. Reporting the
			// failure as the result lets it say something about it, where reporting
			// nothing would leave the call showing as in progress.
			slog.ErrorContext(ctx, "flows: function failed", "function", fn.Name, "err", err)
			return params.Result(ctx, fmt.Sprintf(`{"status": "error", "error": %q}`, err.Error()), nil)
		}

		switch next {
		case TransitionInYAML:
			err := errf(ErrInvalidFunction, fmt.Sprintf(
				"function %s returned TransitionInYAML, but this node was not built from a flow config, "+
					"so nothing can decide the transition; return a NodeConfig or nil", fn.Name))
			slog.ErrorContext(ctx, "flows: function failed", "function", fn.Name, "err", err)
			return params.Result(ctx, fmt.Sprintf(`{"status": "error", "error": %q}`, err.Error()), nil)

		case NoResponse:
			// The function has already said its piece. The result still reaches the
			// conversation, so the model knows what happened when the user speaks
			// next, but nothing is generated from it.
			stay := false
			return params.Result(ctx, result, &frames.FunctionCallResultProperties{RunLLM: &stay})

		case nil:
			// A node function: stay here and answer from the result.
			run := true
			return params.Result(ctx, result, &frames.FunctionCallResultProperties{RunLLM: &run})

		default:
			// An edge function. The transition waits until the result is in the
			// conversation and the turn's other calls have reported, so the node is
			// entered against a finished turn rather than a half-finished one.
			if result == "" {
				result = acknowledged
			}
			fm.mu.Lock()
			fm.pending = &pendingTransition{next: next, function: fn.Name}
			fm.mu.Unlock()

			run := false
			return params.Result(ctx, result, &frames.FunctionCallResultProperties{
				RunLLM:           &run,
				OnContextUpdated: fm.checkAndExecuteTransition,
			})
		}
	}
}

// checkAndExecuteTransition makes the pending transition once the turn's tool
// calls have all reported. It runs after a result has been written to the
// conversation, which is where a turn with several calls finally settles.
func (fm *FlowManager) checkAndExecuteTransition(ctx context.Context) error {
	fm.mu.Lock()
	pending := fm.pending
	fm.mu.Unlock()
	if pending == nil {
		return nil
	}

	if fm.aggs != nil && fm.aggs.Assistant().HasFunctionCallsInProgress() {
		// Another call of this turn is still running. Whichever reports last comes
		// back through here and makes the move.
		return nil
	}

	fm.mu.Lock()
	pending = fm.pending
	fm.pending = nil
	fm.mu.Unlock()
	if pending == nil {
		return nil
	}

	slog.DebugContext(ctx, "flows: transitioning",
		"function", pending.function, "node", nodeName(pending.next))
	return fm.setNode(ctx, nodeName(pending.next), pending.next, pending.function)
}

// validateNode checks that a node can be entered.
func validateNode(id string, node *NodeConfig) error {
	if node == nil {
		return errf(ErrFlow, "node is nil")
	}
	if node.TaskMessages == nil {
		return errf(ErrFlow, fmt.Sprintf("node %q is missing its TaskMessages", id))
	}
	for _, fn := range node.Functions {
		if err := validateFunction(fn); err != nil {
			return err
		}
	}
	return nil
}

// validateFunction checks that a function can be offered.
func validateFunction(fn NodeFunction) error {
	switch {
	case fn.Name == "":
		return errf(ErrInvalidFunction, "function has no name")
	case fn.Handler == nil:
		return errf(ErrInvalidFunction, fmt.Sprintf("function %q has no handler", fn.Name))
	}
	return nil
}

// nodeName returns the node's name, generating one when it has none so two
// unnamed nodes are told apart in the logs.
func nodeName(node *NodeConfig) string {
	if node.Name != "" {
		return node.Name
	}
	return uuid.NewString()
}
