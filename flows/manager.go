package flows

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
	"github.com/google/uuid"
)

// acknowledged is the placeholder tool result used for a function that
// transitioned without a result of its own. The model called it and is owed an
// answer either way.
const acknowledged = `{"status": "acknowledged"}`

// summaryTimeout bounds how long a transition waits for the conversation
// summary the reset-with-summary strategy needs. A transition that waits longer
// than this falls back to appending, which keeps the conversation whole.
const summaryTimeout = 5 * time.Second

// Enqueuer injects frames into the running pipeline; *pipeline.Task satisfies
// it. Everything a flow does to the conversation it does by queueing a frame, so
// the change is ordered against whatever else is in flight.
type Enqueuer interface {
	QueueFrame(f frames.Frame)
	QueueFrames(fs []frames.Frame)
}

// Config configures a FlowManager. The references are the same instances wired
// into the pipeline.
type Config struct {
	// Enqueuer queues the frames a flow produces; usually the *pipeline.Task.
	Enqueuer Enqueuer
	// Watcher reports frames reaching the end of the pipeline, which is how an
	// action learns it has finished; usually the same *pipeline.Task. Leaving it
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
type FlowManager struct {
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
	return fm, nil
}

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
	if err := fm.setNode(ctx, nodeName(node), node); err != nil {
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
	return fm.setNode(ctx, nodeName(node), node)
}

// setNode enters a node, reporting any failure against the node that failed.
func (fm *FlowManager) setNode(ctx context.Context, id string, node *NodeConfig) error {
	if err := fm.enterNode(ctx, id, node); err != nil {
		return &flowError{kind: ErrFlow, msg: fmt.Sprintf("failed to set node %q", id), cause: err}
	}
	return nil
}

// enterNode does the work of entering a node: it runs the pre-actions, applies
// the persona, the objective and the toolset, asks for a response where the node
// speaks on entry, and runs the post-actions.
func (fm *FlowManager) enterNode(ctx context.Context, id string, node *NodeConfig) error {
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
	slog.DebugContext(ctx, "flows: setting node", "node", id)

	// The post-actions still waiting belong to the node being left.
	fm.actions.clearDeferredPostActions()

	for _, a := range node.PreActions {
		if err := fm.actions.registerFromConfig(a); err != nil {
			return err
		}
	}
	for _, a := range node.PostActions {
		if err := fm.actions.registerFromConfig(a); err != nil {
			return err
		}
	}

	if err := fm.actions.executeActions(ctx, node.PreActions); err != nil {
		return err
	}

	tools, names, err := fm.nodeTools(node)
	if err != nil {
		return err
	}

	if err := fm.updateContext(ctx, node, tools); err != nil {
		return err
	}

	fm.mu.Lock()
	fm.current = id
	fm.currentFns = names
	fm.mu.Unlock()

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

// nodeTools builds the toolset a node advertises: the global functions followed
// by the node's own, each carrying the handler that runs it.
func (fm *FlowManager) nodeTools(node *NodeConfig) ([]frames.Tool, map[string]bool, error) {
	all := make([]NodeFunction, 0, len(fm.globals)+len(node.Functions))
	all = append(all, fm.globals...)
	all = append(all, node.Functions...)

	tools := make([]frames.Tool, 0, len(all))
	names := make(map[string]bool, len(all))
	for _, fn := range all {
		if err := validateFunction(fn); err != nil {
			return nil, nil, err
		}
		params, err := fn.parameters()
		if err != nil {
			return nil, nil, errf(ErrInvalidFunction,
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
	}
	return tools, names, nil
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

	switch strategy.Strategy {
	case ContextStrategyReset, ContextStrategyResetWithSummary:
		out = append(out, frames.NewLLMMessagesUpdateFrame(messages))
	default:
		out = append(out, frames.NewLLMMessagesAppendFrame(messages))
	}

	out = append(out, frames.NewLLMSetToolsFrame(tools))

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
	return fm.setNode(ctx, nodeName(pending.next), pending.next)
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
