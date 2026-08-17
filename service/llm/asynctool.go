package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// errCallNotRunning reports a cancellation aimed at a call that has already
// finished, was already canceled, or never existed.
//
//nolint:gochecknoglobals // sentinel error
var errCallNotRunning = errors.New("no such function call is running")

// CancelToolPrefix begins the name of every built-in cancel tool.
const CancelToolPrefix = "cancel_"

// cancelledKey names the field a cancel tool reports its outcome under: the id
// of the call it stopped, or null when it stopped none. The spelling is the
// protocol's, not prose.
//
//nolint:misspell // the key the model is told to expect
const cancelledKey = "cancelled"

// CancelToolName is the name of the tool that cancels calls of functionName: a
// running write_report call is stopped by cancel_write_report.
func CancelToolName(functionName string) string { return CancelToolPrefix + functionName }

// asyncToolCancellationInstructions tell the model which of its tools can be
// stopped early and how to stop one. They are composed into the system
// instruction only while a cancellable tool is registered, so a session with
// none never carries them.
//
//nolint:misspell // prompt text, sent to the model exactly as written
const asyncToolCancellationInstructions = `ASYNC TOOL CANCELLATION:
Some of your tools keep running in the background after you have replied, and some of ` +
	`those can be stopped early.

Work that can be stopped early has its own cancel tool, named for it: a running ` +
	`write_report call is stopped by cancel_write_report. Work with no such tool cannot be ` +
	`stopped and will finish on its own.

When the user no longer wants a result you are still waiting on, call the corresponding ` +
	`cancel tool. Only the call stops the work: saying you canceled it, or that you will skip ` +
	`it, leaves it running and its result will still arrive and contradict you. So when the same ` +
	`turn also asks for something else, make the call and answer, rather than just answering.

Call the cancel tool with no arguments and it stops the one call that is running. If ` +
	`several calls of that same tool are running, it needs a tool_call_id to say which. Each ` +
	`call's id is already in the conversation: find the tool message that reported it running, ` +
	`which carries "status": "running", its "tool_call_id", and the arguments that call was ` +
	`given. Copy that id exactly as written; never invent or guess one.`

// buildCancelToolSchema declares the tool that cancels calls of functionName.
//
// Which work to stop is carried by the tool the model picks, so a single running
// call is stopped without arguments. A tool_call_id only has to be given to
// choose between several calls of the same tool, and the handler refuses with
// the ids when it is needed and missing.
func buildCancelToolSchema(functionName string) frames.Tool {
	return frames.Tool{
		Name: CancelToolName(functionName),
		Description: fmt.Sprintf(
			"Stop a running %[1]s call whose result is no longer needed: the user says to "+
				"drop it, asks for something that replaces it, or says something that makes "+
				"the pending result stale. Call it with no arguments to stop the one running "+
				"call; only when several %[1]s calls are running does it need a tool_call_id "+
				"to say which.", functionName),
		Parameters: json.RawMessage(fmt.Sprintf(`{
			"type": "object",
			"properties": {
				"tool_call_id": {
					"type": "string",
					"description": %s
				}
			},
			"required": []
		}`, cancelToolIDDescription(functionName))),
	}
}

// cancelToolIDDescription describes the tool_call_id a cancel tool takes, as a
// JSON string ready to sit in the schema.
func cancelToolIDDescription(functionName string) string {
	described, err := json.Marshal(fmt.Sprintf(
		"Which %s call to stop. Needed only when several are running, and carried "+
			"by the message in the conversation that reported each one running.", functionName))
	if err != nil {
		return `""`
	}
	return string(described)
}

// WithCancellableByLLM says whether the model may cancel this tool's calls while
// they run. A tool that opts in is advertised alongside a cancel tool named for
// it, and a tool that does not has no cancel tool at all, so there is nothing
// for the model to call against it.
//
// It is only meaningful with WithCancelOnInterruption(false): a synchronous call
// holds the model until it returns, so there is no moment at which it could ask
// for the call to stop. Pairing the two is reported and leaves the tool alone.
func WithCancellableByLLM(on bool) RegisterOption {
	return func(i *registryItem) { i.cancellableByLLM = on }
}

// WithAsyncToolCancellation lets the model cancel every asynchronous call, not
// only the tools that opted in.
//
// Deprecated: register the tools the model may cancel with
// WithCancellableByLLM(true) instead. This widens cancellation to every tool
// registered with WithCancelOnInterruption(false), which lets a model that
// wrongly decides a pending result is unwanted destroy work nobody asked it to.
func WithAsyncToolCancellation() Option {
	return func(b *Base) { b.asyncToolCancellation = true }
}

// resolveCancellableByLLM settles whether the model may cancel a tool's calls,
// refusing the pairing that cannot work.
func (b *Base) resolveCancellableByLLM(item *registryItem) {
	if !item.cancellableByLLM || !item.cancelOnInterruption {
		return
	}
	// A synchronous call holds the model until it returns, so it is never
	// running at a moment the model could ask to cancel it.
	slog.Warn("a synchronous tool cannot be canceled by the model; pair it with WithCancelOnInterruption(false)",
		"service", b.Name(), "function", item.name)
	item.cancellableByLLM = false
}

// isCancellable reports whether the model may cancel calls of a registered tool.
// It is the one place that decides it, so what is advertised and what a cancel
// tool will actually stop cannot drift apart.
func (b *Base) isCancellable(item registryItem) bool {
	return item.cancellableByLLM || (b.asyncToolCancellation && !item.cancelOnInterruption)
}

// syncCancelTools matches the advertised cancel tools to the tools currently
// registered: one cancel tool per cancellable tool, named for it, reconciled
// from a single reading of the registry so what is advertised can never drift
// from what is actually cancellable. It runs wherever the registry changes.
func (b *Base) syncCancelTools() {
	b.handlersMu.Lock()
	wanted := b.wantedCancelToolsLocked()
	withdrawn := b.withdrawCancelToolsLocked(wanted)
	added := b.addCancelToolsLocked(wanted)
	active := slices.Sorted(maps.Keys(b.cancelTools))
	b.handlersMu.Unlock()

	if len(withdrawn) == 0 && len(added) == 0 {
		return
	}
	slog.Debug("cancel tools reconciled", "service", b.Name(), "added", added, "withdrawn", withdrawn)
	b.applyCancelTools(active, withdrawn)
	// The guidance comes and goes with the tools it describes.
	b.composeSystemInstruction()
}

// cancelToolNames are the cancel tools currently registered, in order.
func (b *Base) cancelToolNames() []string {
	b.handlersMu.RLock()
	defer b.handlersMu.RUnlock()
	return slices.Sorted(maps.Keys(b.cancelTools))
}

// wantedCancelToolsLocked maps each cancel tool that should be offered to the
// tool it cancels. Call it with handlersMu held.
func (b *Base) wantedCancelToolsLocked() map[string]string {
	wanted := make(map[string]string)
	for name, item := range b.handlers {
		if name == catchAllFunction || b.cancelTools[name] {
			continue
		}
		if b.isCancellable(item) {
			wanted[CancelToolName(name)] = name
		}
	}
	return wanted
}

// withdrawCancelToolsLocked drops the cancel tools no longer wanted, returning
// their names. Call it with handlersMu held.
func (b *Base) withdrawCancelToolsLocked(wanted map[string]string) []string {
	var withdrawn []string
	for stale := range b.cancelTools {
		if _, keep := wanted[stale]; keep {
			continue
		}
		delete(b.handlers, stale)
		delete(b.cancelTools, stale)
		withdrawn = append(withdrawn, stale)
	}
	return withdrawn
}

// addCancelToolsLocked registers the cancel tools not yet offered, returning
// their names. Call it with handlersMu held.
func (b *Base) addCancelToolsLocked(wanted map[string]string) []string {
	var added []string
	for name, target := range wanted {
		if b.cancelTools[name] {
			continue
		}
		if _, claimed := b.handlers[name]; claimed {
			b.warnCancelCollisionLocked(name, target)
			continue
		}
		if b.handlers == nil {
			b.handlers = make(map[string]registryItem)
		}
		b.handlers[name] = registryItem{
			name:                 name,
			handler:              b.cancelToolHandler,
			cancelOnInterruption: true,
		}
		if b.cancelTools == nil {
			b.cancelTools = make(map[string]bool)
		}
		b.cancelTools[name] = true
		added = append(added, name)
	}
	return added
}

// warnCancelCollisionLocked reports, once per name, that a cancellable tool's
// cancel tool cannot be advertised because the application registered a tool of
// its own under that name. It is reported once because this runs on every
// inference and the collision stands until one of them is renamed. Call it with
// handlersMu held.
func (b *Base) warnCancelCollisionLocked(name, target string) {
	if b.warnedCancelCollisions[name] {
		return
	}
	if b.warnedCancelCollisions == nil {
		b.warnedCancelCollisions = make(map[string]bool)
	}
	b.warnedCancelCollisions[name] = true
	slog.Warn("a cancellable tool's cancel tool cannot be advertised: that name is "+
		"already a tool of its own, so rename one of them or the model has no way to stop it",
		"service", b.Name(), "function", target, "cancel_tool", name)
}

// BuiltinToolHolder is implemented by an adapter the service can add the tools
// it implements itself to. adapter.Base satisfies it, so every jargo adapter
// does.
type BuiltinToolHolder interface {
	// SetBuiltin adds a tool sent on every request from now on.
	SetBuiltin(adapter.Builtin)
	// RemoveBuiltin withdraws the tool registered under name.
	RemoveBuiltin(name string) bool
}

// AdapterHolder is implemented by a service that converts a conversation
// through an adapter, which is where a tool the service implements itself
// belongs: the conversation is shared, so writing the tool into it would offer
// it to every other service reading that conversation.
type AdapterHolder interface {
	// LLMAdapter returns the adapter this service converts through.
	LLMAdapter() BuiltinToolHolder
}

// applyCancelTools brings the tools this service sends into line with the cancel
// tools registered. They belong to the service rather than to the conversation,
// so they come and go with what is registered instead of being written into a
// context the application owns.
func (b *Base) applyCancelTools(active, withdrawn []string) {
	holder, ok := b.gen.(AdapterHolder)
	if !ok {
		// A service that converts without an adapter has nowhere to put the tool.
		// It cannot offer cancellation, and saying so once is better than the
		// model being told to call a tool it is never sent.
		b.warnNoAdapter.Do(func() {
			slog.Warn("canceling a tool call needs a service that converts through an adapter",
				"service", b.Name())
		})
		return
	}
	adapt := holder.LLMAdapter()
	for _, name := range withdrawn {
		adapt.RemoveBuiltin(name)
	}
	for _, name := range active {
		adapt.SetBuiltin(adapter.Builtin{
			Tool: buildCancelToolSchema(strings.TrimPrefix(name, CancelToolPrefix)),
		})
	}
}

// cancelToolHandler answers a cancel_<tool> call from the model.
//
// Which work to stop comes from the tool the model called, so the common case
// takes no arguments at all. A tool_call_id only has to be given to choose
// between several running calls of the same tool.
func (b *Base) cancelToolHandler(ctx context.Context, params FunctionCallParams) error {
	target := strings.TrimPrefix(params.FunctionName, CancelToolPrefix)

	var args struct {
		ToolCallID string `json:"tool_call_id"`
	}
	if len(params.Arguments) > 0 {
		_ = json.Unmarshal(params.Arguments, &args)
	}

	refuse := func(reason string, extra map[string]any) error {
		slog.DebugContext(ctx, "declining to cancel", "service", b.Name(),
			"function", target, "reason", reason)
		payload := map[string]any{cancelledKey: nil, "reason": reason}
		maps.Copy(payload, extra)
		return b.reportCancelResult(ctx, params, payload)
	}

	candidates := b.cancellableCallsOf(target)
	if len(candidates) == 0 {
		return refuse(fmt.Sprintf("no %s call is running", target), nil)
	}

	var match *functionCall
	switch {
	case args.ToolCallID != "":
		for _, c := range candidates {
			if c.toolCallID == args.ToolCallID {
				match = c
				break
			}
		}
		if match == nil {
			return refuse(fmt.Sprintf("no running %s call has that tool_call_id", target), nil)
		}
	case len(candidates) > 1:
		// Several calls of the one tool: the model has to say which, so the
		// refusal carries the ids to say it with. The choices are spelled out in
		// the reason itself, since a model reading a refusal acts on what it says
		// far more readily than on a field beside it.
		choices := make([]string, 0, len(candidates))
		running := make([]map[string]any, 0, len(candidates))
		for _, c := range candidates {
			choices = append(choices, fmt.Sprintf("%s (called with %s)", c.toolCallID, c.args))
			running = append(running, map[string]any{
				"tool_call_id": c.toolCallID,
				"arguments":    c.args,
			})
		}
		return refuse(
			fmt.Sprintf("%d %s calls are running: %s. Call %s again with tool_call_id set to the one to stop.",
				len(candidates), target, strings.Join(choices, "; "), params.FunctionName),
			map[string]any{"running": running})
	default:
		match = candidates[0]
	}

	slog.DebugContext(ctx, "canceling function call at the model's request", "service", b.Name(),
		"function", match.name, "tool_call_id", match.toolCallID)
	b.cancelFunctionCallByID(ctx, match.toolCallID)
	return b.reportCancelResult(ctx, params, map[string]any{
		cancelledKey: match.toolCallID, "function_name": match.name,
	})
}

// reportCancelResult reports what a cancel tool did, asking for inference so the
// model can carry on having been told the outcome.
func (b *Base) reportCancelResult(
	ctx context.Context, params FunctionCallParams, payload map[string]any,
) error {
	result, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	run := true
	return params.Result(ctx, string(result), &frames.FunctionCallResultProperties{RunLLM: &run})
}

// cancellableCallsOf are the in-flight calls of target the model may cancel, in
// a stable order so a refusal names them the same way twice.
func (b *Base) cancellableCallsOf(target string) []*functionCall {
	b.callsMu.Lock()
	defer b.callsMu.Unlock()
	var found []*functionCall
	for _, call := range b.calls {
		if call.name == target && b.isCancellable(call.item) {
			found = append(found, call)
		}
	}
	slices.SortFunc(found, func(a, c *functionCall) int {
		return strings.Compare(a.toolCallID, c.toolCallID)
	})
	return found
}

// CancelAsyncToolCall abandons the call with this id, reporting an error if no
// such call is running. It is what a built-in cancel tool does when the model
// asks; an application calls it directly to abandon background work on its own
// account, when it knows a result has become pointless before the model does.
func (b *Base) CancelAsyncToolCall(ctx context.Context, toolCallID string) error {
	if !b.cancelFunctionCallByID(ctx, toolCallID) {
		return fmt.Errorf("%w: %s", errCallNotRunning, toolCallID)
	}
	return nil
}

// cancelFunctionCallByID abandons one call by id, whatever it was registered
// with, reporting whether there was one to abandon. This is the model asking
// rather than a barge-in, so a tool that survives an interruption is canceled
// here too: that is the whole point of the tool.
func (b *Base) cancelFunctionCallByID(ctx context.Context, toolCallID string) bool {
	b.callsMu.Lock()
	call, running := b.calls[toolCallID]
	if running {
		// Settled before canceling, so a handler that catches its cancellation
		// and reports a result while unwinding cannot reopen a call the pipeline
		// has stopped tracking.
		call.settled = true
		delete(b.calls, toolCallID)
	}
	b.callsMu.Unlock()
	if !running {
		slog.DebugContext(ctx, "asked to cancel a call that is not running",
			"service", b.Name(), "tool_call_id", toolCallID)
		return false
	}

	slog.DebugContext(ctx, "canceling async function call by model request",
		"service", b.Name(), "function", call.name, "tool_call_id", call.toolCallID)
	call.cancel()
	// The result of the tool that asked for this cancellation runs inference
	// already, so asking again here would answer twice.
	b.broadcastFunctionCallCanceled(ctx, call, false)

	b.eventsMu.RLock()
	h := b.onCanceled
	b.eventsMu.RUnlock()
	b.notify(ctx, h, []frames.ToolCall{{ID: call.toolCallID, Name: call.name, Args: call.args}})
	return true
}
