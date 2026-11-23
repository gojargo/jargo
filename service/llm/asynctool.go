package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"

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

// buildCancelToolSchema declares the tool that cancels calls of functionName,
// answered by handler.
//
// Which work to stop is carried by the tool the model picks, so a single running
// call is stopped without arguments. A tool_call_id only has to be given to
// choose between several calls of the same tool, and the handler refuses with
// the ids when it is needed and missing.
func buildCancelToolSchema(functionName string, handler FunctionCallHandler) frames.Tool {
	return frames.Tool{
		Handler: handler,
		Name:    CancelToolName(functionName),
		Description: fmt.Sprintf(
			"Stop a running %[1]s call whose result is no longer needed: the user says to "+
				"drop it, asks for something that replaces it, or says something that makes "+
				"the pending result stale. Call it with no arguments to stop the one running "+
				"call; only when several %[1]s calls are running does it need a tool_call_id "+
				"to say which.", functionName),
		Parameters: cancelToolParameters(functionName),
	}
}

// cancelToolParameters is the schema for the one argument a cancel tool takes.
//
// It is encoded rather than written out as text with the name substituted in,
// so a function name carrying a quote cannot break out of the string it sits in
// and reshape the schema around it.
func cancelToolParameters(functionName string) json.RawMessage {
	described := fmt.Sprintf(
		"Which %s call to stop. Needed only when several are running, and carried "+
			"by the message in the conversation that reported each one running.", functionName)
	schema, err := json.Marshal(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tool_call_id": map[string]any{
				"type":        "string",
				"description": described,
			},
		},
		"required": []string{},
	})
	if err != nil {
		// Strings and maps of them always encode. Fall back to the same shape
		// without the description rather than to nothing, so the tool still
		// takes the argument.
		return json.RawMessage(`{"type":"object","properties":{"tool_call_id":{"type":"string"}},"required":[]}`)
	}
	return schema
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
// refusing the pairing that cannot work. The deprecated
// WithAsyncToolCancellation widens it here, at registration, to every
// asynchronous tool, so the registry alone says what the model may cancel.
func (b *Base) resolveCancellableByLLM(item *registryItem) {
	if !item.cancellableByLLM && (!b.asyncToolCancellation || item.cancelOnInterruption) {
		return
	}
	if item.cancelOnInterruption {
		// A synchronous call holds the model until it returns, so it is never
		// running at a moment the model could ask to cancel it.
		slog.Warn("a synchronous tool cannot be canceled by the model; pair it with WithCancelOnInterruption(false)",
			"service", b.Name(), "function", item.name)
		item.cancellableByLLM = false
		return
	}
	item.cancellableByLLM = true
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

// builtinLister is implemented by an adapter that can say which tools the service
// implements itself. adapter.Base satisfies it, so every jargo adapter does.
type builtinLister interface {
	WithBuiltins(frames.ToolsSchema) frames.ToolsSchema
}

// hasBuiltinTools reports whether the adapter this service converts through
// carries any tool the service implements itself.
func (b *Base) hasBuiltinTools() bool {
	holder, ok := b.gen.(AdapterHolder)
	if !ok {
		return false
	}
	lister, ok := holder.LLMAdapter().(builtinLister)
	return ok && len(lister.WithBuiltins(frames.ToolsSchema{}).Standard) > 0
}

// CancelAsyncToolCall abandons the call with this id, reporting an error if no
// such call is running. It is what a built-in cancel tool does when the model
// asks; an application calls it directly to abandon background work on its own
// account, when it knows a result has become pointless before the model does.
//
// It is CancelFunctionCall, reporting whether there was a call to cancel.
func (b *Base) CancelAsyncToolCall(ctx context.Context, toolCallID string) error {
	canceled := b.cancelFunctionCallTasks(ctx, func(call *functionCall) bool {
		return call.toolCallID == toolCallID
	}, "user request")
	if len(canceled) == 0 {
		slog.DebugContext(ctx, "asked to cancel a call that is not running",
			"service", b.Name(), "tool_call_id", toolCallID)
		return fmt.Errorf("%w: %s", errCallNotRunning, toolCallID)
	}
	return nil
}
