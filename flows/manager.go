package flows

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/llm"
)

// acknowledged is the placeholder tool result used for a transition whose
// handler returns no result of its own.
const acknowledged = `{"status": "acknowledged"}`

//nolint:gochecknoglobals // sentinel error
var errNilNode = errors.New("flows: initial node is nil")

//nolint:gochecknoglobals // sentinel error
var errFuncNoName = errors.New("flows: function has an empty name")

//nolint:gochecknoglobals // sentinel error
var errFuncNoHandler = errors.New("flows: function has no handler")

//nolint:gochecknoglobals // sentinel error
var errFuncDuplicate = errors.New("flows: duplicate function name")

// LLM is the part of the LLM service the manager drives: it registers a handler
// for each node function so the service dispatches matching calls to it. The
// concrete services satisfy it through the embedded llm.Base.
type LLM interface {
	RegisterFunction(name string, h llm.FunctionCallHandler, opts ...llm.RegisterOption)
}

// Enqueuer injects a frame into the running pipeline; *pipeline.Task satisfies
// it. The manager uses it only to trigger the assistant's first response.
type Enqueuer interface {
	QueueFrame(f frames.Frame)
}

// Config configures a FlowManager. The three references are the same instances
// wired into the pipeline: the LLM service, the conversation context shared by
// the aggregators, and the task that runs the pipeline.
type Config struct {
	// LLM is the tool-capable LLM service in the pipeline.
	LLM LLM `validate:"required"`
	// Context is the conversation the pipeline's aggregators share.
	Context *frames.LLMContext `validate:"required"`
	// Enqueuer triggers the first response; usually the *pipeline.Task.
	Enqueuer Enqueuer `validate:"required"`
	// GlobalFunctions are offered at every node in addition to the node's own. A
	// node function of the same name takes precedence.
	GlobalFunctions []NodeFunction
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// FlowManager drives a conversation through a graph of nodes. It is not a
// pipeline processor: it holds the LLM service and the shared context and steers
// the conversation by swapping the system prompt, task messages and toolset as
// nodes are entered, leaving the LLM service's tool loop to carry out each
// transition. Build one with New and enter the graph with Initialize.
type FlowManager struct {
	llm     LLM
	convo   *frames.LLMContext
	enq     Enqueuer
	globals []NodeFunction

	mu      sync.Mutex
	current *NodeConfig
}

// New builds a FlowManager from cfg.
func New(cfg Config) (*FlowManager, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("flows: invalid config: %w", err)
	}
	if err := validateFunctions(cfg.GlobalFunctions); err != nil {
		return nil, fmt.Errorf("flows: global functions: %w", err)
	}
	return &FlowManager{
		llm:     cfg.LLM,
		convo:   cfg.Context,
		enq:     cfg.Enqueuer,
		globals: cfg.GlobalFunctions,
	}, nil
}

// Initialize enters the flow at node and, unless the node waits for the user,
// triggers the assistant's first response. Call it once per session, for example
// when the transport connects.
func (fm *FlowManager) Initialize(ctx context.Context, node *NodeConfig) error {
	if node == nil {
		return errNilNode
	}
	if err := fm.setNode(ctx, node); err != nil {
		return err
	}
	if respondsImmediately(node) {
		fm.enq.QueueFrame(frames.NewLLMRunFrame())
	}
	return nil
}

// CurrentNode returns the name of the node the flow is on, or "" before
// Initialize. A node with no name reports a placeholder.
func (fm *FlowManager) CurrentNode() string {
	fm.mu.Lock()
	defer fm.mu.Unlock()
	if fm.current == nil {
		return ""
	}
	return nodeName(fm.current)
}

// setNode makes node current: it applies the persona, appends the task messages,
// swaps the toolset and registers the node's handlers. It does not trigger a
// generation itself — Initialize does that for the opening node, and for a
// transition the LLM service's tool loop regenerates once the handler returns.
func (fm *FlowManager) setNode(ctx context.Context, node *NodeConfig) error {
	if err := validateFunctions(node.Functions); err != nil {
		return fmt.Errorf("flows: node %q: %w", nodeName(node), err)
	}
	slog.DebugContext(ctx, "flows: entering node", "node", nodeName(node))

	// The persona is sticky: replace the system prompt only when the node sets
	// one of its own.
	if node.RoleMessage != "" {
		fm.convo.SetSystem(node.RoleMessage)
	}

	// Append the node's objective to the running history so the model pursues it
	// while keeping the prior conversation.
	for _, m := range node.TaskMessages {
		fm.appendMessage(m)
	}

	// Offer the node's functions plus the globals, and register a handler for
	// each so the service can dispatch the model's calls.
	funcs := fm.nodeFunctions(node)
	tools := make([]frames.Tool, 0, len(funcs))
	for _, fn := range funcs {
		tools = append(tools, frames.Tool{
			Name:        fn.Name,
			Description: fn.Description,
			Parameters:  fn.Parameters,
		})
		fm.llm.RegisterFunction(fn.Name, fm.wrap(fn))
	}
	fm.convo.SetTools(tools)

	fm.mu.Lock()
	fm.current = node
	fm.mu.Unlock()
	return nil
}

// wrap adapts a node function's Handler into the llm.FunctionCallHandler the
// service calls. When the handler returns a next node, wrap performs the
// transition in place: it swaps the node, then either lets the tool loop
// regenerate (the new node responds on entry) or ends the turn so the assistant
// waits for the user.
func (fm *FlowManager) wrap(fn NodeFunction) llm.FunctionCallHandler {
	return func(ctx context.Context, params llm.FunctionCallParams) error {
		result, next, err := fn.Handler(ctx, params.Arguments, fm)
		if err != nil {
			return err
		}
		if next == nil {
			return params.Result(ctx, result, nil)
		}
		if result == "" {
			result = acknowledged
		}
		if terr := fm.setNode(ctx, next); terr != nil {
			// The result still goes back: the model called the function and is
			// owed an answer whether or not the transition took.
			_ = params.Result(ctx, result, nil)
			return terr
		}
		if !respondsImmediately(next) {
			// The node entered waits for the user, so the result is recorded
			// without the assistant being asked to say anything about it.
			stay := false
			return params.Result(ctx, result, &frames.FunctionCallResultProperties{RunLLM: &stay})
		}
		return params.Result(ctx, result, nil)
	}
}

// appendMessage adds a task message to the conversation. Objectives phrased as
// user or system content alike enter as a user turn, because the providers carry
// the system prompt separately from the message history.
func (fm *FlowManager) appendMessage(m frames.Message) {
	if m.Role == frames.RoleAssistant {
		fm.convo.AddAssistantMessage(m.Text)
		return
	}
	fm.convo.AddUserMessage(m.Text)
}

// nodeFunctions returns the node's functions followed by any global function the
// node does not already override by name.
func (fm *FlowManager) nodeFunctions(node *NodeConfig) []NodeFunction {
	funcs := make([]NodeFunction, 0, len(node.Functions)+len(fm.globals))
	funcs = append(funcs, node.Functions...)
	seen := make(map[string]bool, len(node.Functions))
	for _, fn := range node.Functions {
		seen[fn.Name] = true
	}
	for _, fn := range fm.globals {
		if !seen[fn.Name] {
			funcs = append(funcs, fn)
		}
	}
	return funcs
}

// validateFunctions checks that every function has a name and handler and that
// no two share a name.
func validateFunctions(fns []NodeFunction) error {
	seen := make(map[string]bool, len(fns))
	for _, fn := range fns {
		switch {
		case fn.Name == "":
			return errFuncNoName
		case fn.Handler == nil:
			return fmt.Errorf("%w: %q", errFuncNoHandler, fn.Name)
		case seen[fn.Name]:
			return fmt.Errorf("%w: %q", errFuncDuplicate, fn.Name)
		}
		seen[fn.Name] = true
	}
	return nil
}

// respondsImmediately reports whether the node should generate a response on
// entry. The zero value (nil) means it should.
func respondsImmediately(node *NodeConfig) bool {
	return node.RespondImmediately == nil || *node.RespondImmediately
}

// nodeName returns the node's name, or a placeholder when it has none.
func nodeName(node *NodeConfig) string {
	if node.Name != "" {
		return node.Name
	}
	return "(unnamed)"
}
