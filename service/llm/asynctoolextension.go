package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
)

// AsyncToolExtension teaches the model about its asynchronous tools, and
// gives it a cancel tool for each it may stop.
//
// An asynchronous tool (registered with WithCancelOnInterruption(false)) keeps
// running after the model has replied, and its result arrives later. While any
// is registered, the extension adds the guidance on such tools to the system
// instruction. A tool registered with WithCancellableByLLM(true) gets a
// built-in cancel tool named for it (cancel_<name>), with the guidance on
// canceling, so the model can drop work whose result is no longer wanted.
//
// Every LLM service has one, from construction; it contributes nothing until a
// tool is registered that calls for it, and follows the registered tools as
// they come and go.
type AsyncToolExtension struct {
	BaseExtension

	mu            sync.Mutex
	hasAsyncTools bool
	// cancelToolNames are the cancel tools advertised, by name.
	cancelToolNames map[string]bool
	// warnedCollisions are the cancel-tool names that collide with a tool of the
	// application's own, warned about once.
	warnedCollisions map[string]bool
}

// NewAsyncToolExtension builds the async-tool extension. A service attaches
// its own at construction, so an application never needs to.
func NewAsyncToolExtension() *AsyncToolExtension {
	return &AsyncToolExtension{
		cancelToolNames:  make(map[string]bool),
		warnedCollisions: make(map[string]bool),
	}
}

// Order implements Extension. It sits after the extensions an application
// attaches, whose instructions come first, as the tool guidance closes the
// system instruction.
func (e *AsyncToolExtension) Order() int { return 1 }

// SystemInstruction implements Extension: the cancellation guidance while a
// cancel tool is advertised, and the async-tool guidance while an asynchronous
// tool is registered.
//
// The async-tool guidance is the standing policy on results that arrive after
// the reply. The message carrying each result says the same thing, but it
// lands in a conversation whose most recent turn is the user asking for
// something else, and a model weighing the two follows the nearer request.
// Stating it up front puts it in force before any result exists.
func (e *AsyncToolExtension) SystemInstruction() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	var parts []string
	if len(e.cancelToolNames) > 0 {
		parts = append(parts, asyncToolCancellationInstructions)
	}
	if e.hasAsyncTools {
		parts = append(parts, frames.AsyncToolInstructions)
	}
	return strings.Join(parts, "\n\n")
}

// BuiltinTools implements Extension: one cancel tool per cancellable tool,
// named for it.
//
// It is reconciled from a single reading of the registry, so the advertised set
// can never drift from what is actually cancellable. A tool that never opted in
// has no cancel tool at all, so there is nothing for the model to call against
// it.
func (e *AsyncToolExtension) BuiltinTools(registered map[string]FunctionCallRegistryItem) []ExtensionTool {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.hasAsyncTools = false
	for name, item := range registered {
		if !e.cancelToolNames[name] && !item.CancelOnInterruption {
			e.hasAsyncTools = true
			break
		}
	}
	wanted := make(map[string]string)
	for name, item := range registered {
		if name != catchAllFunction && item.CancellableByLLM {
			wanted[CancelToolName(name)] = name
		}
	}
	var tools []ExtensionTool
	for _, name := range slices.Sorted(maps.Keys(wanted)) {
		target := wanted[name]
		if _, claimed := registered[name]; claimed && !e.cancelToolNames[name] {
			// Once per name: this runs before every inference, and the collision
			// persists for as long as both tools are registered.
			if !e.warnedCollisions[name] {
				e.warnedCollisions[name] = true
				slog.Warn("a cancellable tool's cancel tool cannot be advertised: that name is "+
					"already a tool of its own, so rename one of them or the model has no way to stop it",
					"function", target, "cancel_tool", name)
			}
			continue
		}
		tools = append(tools, ExtensionTool{Schema: buildCancelToolSchema(target, e.cancel)})
	}
	e.cancelToolNames = make(map[string]bool, len(tools))
	for _, tool := range tools {
		e.cancelToolNames[tool.Schema.Name] = true
	}
	return tools
}

// cancel answers a cancel_<tool> call from the model.
//
// Which work to stop comes from the tool the model called, so the common case
// takes no arguments at all. A tool_call_id only has to be given to choose
// between several running calls of the same tool.
func (e *AsyncToolExtension) cancel(ctx context.Context, params FunctionCallParams) error {
	target := strings.TrimPrefix(params.FunctionName, CancelToolPrefix)

	var args struct {
		ToolCallID string `json:"tool_call_id"`
	}
	if len(params.Arguments) > 0 {
		_ = json.Unmarshal(params.Arguments, &args)
	}

	refuse := func(reason string, extra map[string]any) error {
		slog.DebugContext(ctx, "declining to cancel", "function", target, "reason", reason)
		payload := map[string]any{cancelledKey: nil, "reason": reason}
		maps.Copy(payload, extra)
		return reportCancelResult(ctx, params, payload)
	}

	var candidates []FunctionCallRunnerItem
	for _, call := range params.LLM.RunningFunctionCalls() {
		if call.FunctionName == target && call.RegistryItem.CancellableByLLM {
			candidates = append(candidates, call)
		}
	}
	if len(candidates) == 0 {
		return refuse(fmt.Sprintf("no %s call is running", target), nil)
	}

	var match FunctionCallRunnerItem
	switch {
	case args.ToolCallID != "":
		i := slices.IndexFunc(candidates, func(c FunctionCallRunnerItem) bool {
			return c.ToolCallID == args.ToolCallID
		})
		if i < 0 {
			return refuse(fmt.Sprintf("no running %s call has that tool_call_id", target), nil)
		}
		match = candidates[i]
	case len(candidates) > 1:
		// Several calls of the one tool: the model has to say which, so the
		// refusal carries the ids to say it with. The choices are spelled out in
		// the reason itself, since a model reading a refusal acts on what it says
		// far more readily than on a field beside it.
		choices := make([]string, 0, len(candidates))
		running := make([]map[string]any, 0, len(candidates))
		for _, c := range candidates {
			choices = append(choices, fmt.Sprintf("%s (called with %s)", c.ToolCallID, c.Arguments))
			running = append(running, map[string]any{
				"tool_call_id": c.ToolCallID,
				"arguments":    c.Arguments,
			})
		}
		return refuse(
			fmt.Sprintf("%d %s calls are running: %s. Call %s again with tool_call_id set to the one to stop.",
				len(candidates), target, strings.Join(choices, "; "), params.FunctionName),
			map[string]any{"running": running})
	default:
		match = candidates[0]
	}

	slog.DebugContext(ctx, "canceling function call at the model's request",
		"function", match.FunctionName, "tool_call_id", match.ToolCallID)
	params.LLM.CancelFunctionCall(ctx, match.ToolCallID, "user request")
	return reportCancelResult(ctx, params, map[string]any{
		cancelledKey: match.ToolCallID, "function_name": match.FunctionName,
	})
}

// reportCancelResult reports what a cancel tool did, asking for inference so the
// model can carry on having been told the outcome.
func reportCancelResult(ctx context.Context, params FunctionCallParams, payload map[string]any) error {
	result, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	run := true
	return params.Result(ctx, string(result), &frames.FunctionCallResultProperties{RunLLM: &run})
}
