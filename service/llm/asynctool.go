package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"

	"github.com/gojargo/jargo/frames"
)

// errCallNotRunning reports a cancellation aimed at a call that has already
// finished, was already canceled, or never existed.
//
//nolint:gochecknoglobals // sentinel error
var errCallNotRunning = errors.New("no such function call is running")

// CancelAsyncToolName is the built-in tool the model calls to abandon an
// asynchronous call whose result it no longer wants. It is reserved: registering
// a handler under this name is refused, since the service owns it.
const CancelAsyncToolName = "cancel_async_tool_call"

// asyncToolCancellationInstructions tell the model how to recognize a running
// asynchronous call in the conversation and how to abandon one. They are
// appended to the system prompt only while an asynchronous tool is registered
// and cancellation is enabled, so a session with no such tool never carries
// them.
//
//nolint:misspell // prompt text, sent to the model exactly as written
const asyncToolCancellationInstructions = `ASYNC TOOL CANCELLATION:
Some tool calls run asynchronously in the background. When one starts, a tool response ` +
	`is added to the conversation whose content is a JSON object with ` +
	`"type": "async_tool", "status": "running", and a "tool_call_id" field containing the ` +
	`exact ID of that call (e.g. {"type": "async_tool", "status": "running", "tool_call_id": "..."}).

If the user changes topic, explicitly says they no longer need the result, or the pending ` +
	`result would clearly be stale, call cancel_async_tool_call. ` +
	`To find the correct tool_call_id: locate the most recent tool response in the conversation ` +
	`whose content has "status": "running" and whose call has NOT already been cancelled, ` +
	`then copy the "tool_call_id" value from that content exactly as-is. ` +
	`Never invent or guess a tool_call_id.`

// cancelAsyncToolDescription tells the model when reaching for the tool is the
// right move.
const cancelAsyncToolDescription = "Cancel a single async tool call whose results are no longer needed. " +
	"Use this when the user changes topic, indicates a pending result is " +
	"no longer relevant, or when processing the result would produce a " +
	"stale or confusing response. " +
	"The tool_call_id must be copied exactly from the 'tool_call_id' field " +
	"in the async tool's 'running' response visible in the conversation history."

// cancelAsyncToolSchema is the built-in tool's declaration.
//
//nolint:gochecknoglobals // a fixed declaration, built once
var cancelAsyncToolSchema = frames.Tool{
	Name:        CancelAsyncToolName,
	Description: cancelAsyncToolDescription,
	Parameters: json.RawMessage(`{
		"type": "object",
		"properties": {
			"tool_call_id": {
				"type": "string",
				"description": "The exact tool_call_id from the async tool's 'running' response to cancel."
			}
		},
		"required": ["tool_call_id"]
	}`),
}

// WithAsyncToolCancellation lets the model abandon an asynchronous call whose
// result it no longer wants.
//
// While at least one tool is registered with WithCancelOnInterruption(false),
// the service offers a built-in cancel_async_tool_call tool alongside the
// conversation's own, and appends instructions telling the model how to find the
// id of a call still running. Both appear only while such a tool is registered,
// so a session that has none is unaffected.
//
// It is off by default: a model given the tool will sometimes use it, and
// whether abandoning background work is right is the application's call.
func WithAsyncToolCancellation() Option {
	return func(b *Base) { b.asyncToolCancellation = true }
}

// syncAsyncToolCancellation brings the built-in tool and its instructions into
// line with what is registered, adding them once an asynchronous tool exists and
// taking them away when the last one goes. It is called wherever the registry
// changes, and does nothing unless the service was built with
// WithAsyncToolCancellation.
func (b *Base) syncAsyncToolCancellation() {
	if !b.asyncToolCancellation {
		return
	}
	want := b.hasAsyncTools()

	b.handlersMu.Lock()
	have := b.cancelToolActive
	if want == have {
		b.handlersMu.Unlock()
		return
	}
	b.cancelToolActive = want
	if want {
		if b.handlers == nil {
			b.handlers = make(map[string]registryItem)
		}
		b.handlers[CancelAsyncToolName] = registryItem{
			name:                 CancelAsyncToolName,
			handler:              b.cancelAsyncToolHandler,
			cancelOnInterruption: true,
		}
	} else {
		delete(b.handlers, CancelAsyncToolName)
	}
	b.handlersMu.Unlock()

	if want {
		slog.Debug("enabling async tool cancellation", "service", b.Name())
	} else {
		slog.Debug("disabling async tool cancellation", "service", b.Name())
	}
}

// hasAsyncTools reports whether any registered tool outlives an interruption.
// The built-in cancel tool does not count: it exists to serve the others.
func (b *Base) hasAsyncTools() bool {
	b.handlersMu.RLock()
	defer b.handlersMu.RUnlock()
	for name, item := range b.handlers {
		if name != CancelAsyncToolName && !item.cancelOnInterruption {
			return true
		}
	}
	return false
}

// cancelToolEnabled reports whether the built-in tool is currently offered.
func (b *Base) cancelToolEnabled() bool {
	b.handlersMu.RLock()
	defer b.handlersMu.RUnlock()
	return b.cancelToolActive
}

// applyAsyncToolCancellation adds the built-in tool and its instructions to what
// this inference sends, without touching the stored conversation. They belong to
// the service rather than to the conversation, so they come and go with what is
// registered instead of being written into a context the application owns.
func (b *Base) applyAsyncToolCancellation(convo *frames.LLMContext) {
	if !b.asyncToolCancellation {
		// The service never offers the tool, so it has nothing to say about the
		// conversation either way. Leaving it alone matters: another service may
		// share this conversation and have its own to add.
		return
	}
	if !b.cancelToolEnabled() {
		convo.SetServiceTools(nil)
		convo.SetServiceInstructions("")
		return
	}
	convo.SetServiceTools([]frames.Tool{cancelAsyncToolSchema})
	convo.SetServiceInstructions(asyncToolCancellationInstructions)
}

// cancelAsyncToolHandler abandons the call the model named. It reports which id
// it acted on and asks for generation to re-run, so the model can carry on
// having been told the work is off.
func (b *Base) cancelAsyncToolHandler(ctx context.Context, params FunctionCallParams) error {
	var args struct {
		ToolCallID string `json:"tool_call_id"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil || args.ToolCallID == "" {
		slog.WarnContext(ctx, "cancel_async_tool_call named no call", "service", b.Name())
		//nolint:misspell // the key the model is told to expect
		return params.Result(ctx, `{"cancelled": null}`, nil)
	}

	b.cancelFunctionCallByID(ctx, args.ToolCallID)

	run := true
	//nolint:misspell // the key the model is told to expect
	result, err := json.Marshal(map[string]string{"cancelled": args.ToolCallID})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), &frames.FunctionCallResultProperties{RunLLM: &run})
}

// CancelAsyncToolCall abandons the call with this id, reporting an error if no
// such call is running. It is what the built-in cancel_async_tool_call tool does
// when the model asks; an application calls it directly to abandon background
// work on its own account, when it knows a result has become pointless before
// the model does.
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
		call.canceled = true
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
	_ = b.Broadcast(ctx, func() frames.Frame {
		return frames.NewFunctionCallCancelFrame(call.toolCallID, call.name)
	})

	b.eventsMu.RLock()
	h := b.onCanceled
	b.eventsMu.RUnlock()
	b.notify(ctx, h, []frames.ToolCall{{ID: call.toolCallID, Name: call.name, Args: call.args}})
	return true
}
