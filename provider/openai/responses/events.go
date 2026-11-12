package responses

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// Event types the services read. The Responses API streams the same events over
// SSE and over the WebSocket, so both transports share this decoding.
const (
	// evtCreated announces the response's id, which the WebSocket service needs
	// to continue from it next turn.
	evtCreated = "response.created"
	// evtTextDelta is a chunk of generated text.
	evtTextDelta = "response.output_text.delta"
	// evtItemAdded announces an output item; a function call arrives this way,
	// carrying the call id and the function name.
	evtItemAdded = "response.output_item.added"
	// evtItemDone completes an output item; a function call's carries its
	// complete arguments.
	evtItemDone = "response.output_item.done"
	// evtArgsDelta is a chunk of a function call's arguments.
	evtArgsDelta = "response.function_call_arguments.delta"
	// evtArgsDone completes a function call's arguments.
	evtArgsDone = "response.function_call_arguments.done"
	// evtCompleted ends a successful response.
	evtCompleted = "response.completed"
	// evtFailed ends a response the model could not produce.
	evtFailed = "response.failed"
	// evtIncomplete ends a response cut short, by the output-token cap say.
	evtIncomplete = "response.incomplete"
	// evtError reports a stream-level failure.
	evtError = "error"
)

// event is the subset of a Responses stream event the services read.
type event struct {
	Type string `json:"type"`
	// StreamID is the lane of the WebSocket connection the event's response
	// runs on, empty for an event that concerns the connection.
	StreamID string `json:"stream_id"`
	// Delta carries text or argument fragments, depending on Type.
	Delta string `json:"delta"`
	// ItemID and OutputIndex identify which output item a delta belongs to.
	OutputIndex int `json:"output_index"`
	// Item is the output item announced by response.output_item.added.
	Item *outputItem `json:"item"`
	// Arguments is the completed argument JSON on response.function_call_arguments.done.
	Arguments string `json:"arguments"`
	// Response carries the response object on lifecycle events.
	Response *responseObject `json:"response"`
	// Message and Code describe a stream-level error as the HTTP stream sends
	// it, at the top level of the event.
	Message string `json:"message"`
	Code    string `json:"code"`
	// Error describes a stream-level error as the WebSocket sends it, nested
	// under the event.
	Error *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// errorMessage is the message of an error event, in either of its shapes.
func (ev event) errorMessage() string {
	if ev.Error != nil && ev.Error.Message != "" {
		return ev.Error.Message
	}
	return ev.Message
}

// errorCode is the code of an error event, in either of its shapes.
func (ev event) errorCode() string {
	if ev.Error != nil && ev.Error.Code != "" {
		return ev.Error.Code
	}
	return ev.Code
}

// outputItem is an item the model produced: a message or a function call.
type outputItem struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
	// Arguments is a function call's complete arguments, on the item that
	// response.output_item.done completes.
	Arguments string `json:"arguments"`
}

// responseObject is the response a lifecycle event reports on.
type responseObject struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	// Model is the model that answered, more specific than the one asked for:
	// an alias resolves to a dated build here.
	Model string `json:"model"`
	Error *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Usage *usage `json:"usage"`
	// Output is every item the response produced, as response.completed
	// reports it.
	Output []responseOutput `json:"output"`
}

// responseOutput is one item of a completed response's output: a message, a
// function call, a reasoning item, or a kind of item the service does not read.
type responseOutput struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	Role   string `json:"role"`
	CallID string `json:"call_id"`
	// Content holds a message's parts; its text is in the output_text ones.
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

// text is a message item's text: its output_text parts joined.
func (o responseOutput) text() string {
	var b strings.Builder
	for _, p := range o.Content {
		if p.Type == "output_text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// usage is the Responses API's per-response token accounting.
type usage struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens     int64 `json:"cached_tokens"`
		CacheWriteTokens int64 `json:"cache_write_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails *struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
	// raw is the usage object as sent, for a server that reports fields this
	// struct does not model.
	raw json.RawMessage
}

// UnmarshalJSON decodes the usage and keeps it as sent.
func (u *usage) UnmarshalJSON(data []byte) error {
	type plain usage
	var p plain
	if err := json.Unmarshal(data, &p); err != nil {
		return err
	}
	*u = usage(p)
	u.raw = slices.Clone(data)
	return nil
}

// tokenUsage converts the wire accounting into the framework's usage shape.
func (u usage) tokenUsage() frames.LLMTokenUsage {
	// The cache and reasoning counts are reported whether or not the response
	// gave them, as zero when it did not.
	var cached, written, reasoning int64
	if d := u.InputTokensDetails; d != nil {
		cached, written = d.CachedTokens, d.CacheWriteTokens
	}
	if d := u.OutputTokensDetails; d != nil {
		reasoning = d.ReasoningTokens
	}
	return frames.LLMTokenUsage{
		PromptTokens:        u.InputTokens,
		CompletionTokens:    u.OutputTokens,
		TotalTokens:         u.TotalTokens,
		CacheReadTokens:     new(cached),
		CacheCreationTokens: new(written),
		ReasoningTokens:     new(reasoning),
	}
}

// pendingCall accumulates a function call as its arguments stream in.
type pendingCall struct {
	callID string
	name   string
	args   string
	// done records that an event said the arguments are complete.
	done bool
}

// streamState turns a sequence of events into text deltas and tool calls. It is
// the transport-independent half of a turn: both services feed it events and it
// reports when the turn is over.
type streamState struct {
	// sink receives the text and tool calls. Text-only generation wraps an Emit.
	sink llm.Sink
	// calls tracks function calls by output index, since their arguments arrive
	// in fragments interleaved with other output.
	calls map[int]*pendingCall
	// responseID is the id the response announced, for continuation.
	responseID string
	// usage is the token accounting the terminal event reported.
	usage *usage
	// hasText records whether the model produced any text other than
	// whitespace, and sawCall whether it called a function.
	hasText bool
	sawCall bool
	// errorPrefix names where a stream-level error event came from, in the
	// message it is reported with.
	errorPrefix string
	// completed records that the response ended with response.completed, and
	// output is the output that event reported: what the server holds of this
	// response, which the next turn can continue from.
	completed bool
	output    []responseOutput
	// model is the model a completed response names.
	model string
}

// emptyResponse reports whether the response completed with neither text nor a
// function call: nothing reaches the TTS, and the bot stays silent until the
// user speaks again.
func (s *streamState) emptyResponse() bool {
	return s.completed && !s.hasText && !s.sawCall
}

// reportEmptyResponse reports a response that completed with no text and no
// function call: a warning listing its output items (a message with its content
// types, message[refusal] say) to show why, and a non-fatal error the
// application can react to. incremental says whether the request sent only the
// new input after a previous response; it is nil for a transport that always
// sends the whole conversation.
func reportEmptyResponse(ctx context.Context, b *llm.Base, s *streamState, incremental *bool) {
	args := []any{"service", b.Name(), "response", s.responseID, "output", outputItemKinds(s)}
	if incremental != nil {
		args = append(args, "incremental", *incremental)
	}
	slog.WarnContext(ctx, "response completed with no text or function call", args...)
	b.PushError(ctx, fmt.Sprintf("LLM response %s completed with no text or function call", s.responseID), nil, false)
}

// outputItemKinds names a response's output items by type, a message with its
// content types: message[refusal], say.
func outputItemKinds(s *streamState) []string {
	items := make([]string, 0, len(s.output))
	for _, o := range s.output {
		kind := cmp.Or(o.Type, "unknown")
		if kind == itemMessage {
			types := make([]string, 0, len(o.Content))
			for _, c := range o.Content {
				types = append(types, cmp.Or(c.Type, "unknown"))
			}
			kind = "message[" + strings.Join(types, ",") + "]"
		}
		items = append(items, kind)
	}
	return items
}

// startsFunctionCall reports whether an event announces a function call, the
// answer of a turn that calls a tool instead of replying in text.
func startsFunctionCall(ev event) bool {
	return ev.Type == evtItemAdded && ev.Item != nil && ev.Item.Type == itemFuncCall
}

// newStreamState builds a state machine feeding sink.
//
// errorPrefix names where a stream-level error event came from, in the message
// it is reported with: the HTTP stream and the WebSocket say it differently.
func newStreamState(sink llm.Sink, errorPrefix string) *streamState {
	return &streamState{sink: sink, calls: map[int]*pendingCall{}, errorPrefix: errorPrefix}
}

// handle folds one event in. It reports done when the turn has ended, and an
// error when the response failed.
func (s *streamState) handle(ev event) (done bool, err error) {
	switch ev.Type {
	case evtCreated:
		if ev.Response != nil {
			s.responseID = ev.Response.ID
		}
	case evtTextDelta:
		if ev.Delta != "" {
			s.hasText = s.hasText || strings.TrimSpace(ev.Delta) != ""
			return false, s.sink.Text(ev.Delta)
		}
	case evtItemAdded, evtArgsDelta, evtArgsDone, evtItemDone:
		s.trackCall(ev)
	case evtCompleted:
		s.finish(ev)
		s.completed = true
		if ev.Response != nil {
			s.output = ev.Response.Output
			s.model = ev.Response.Model
		}
		return true, s.flushCalls(false)
	case evtIncomplete, evtFailed:
		// A response cut short, by the output cap say, or one that failed. Either
		// is an error, but a call whose arguments finished before it is still
		// one the model made, and still runs. A call cut off part way is
		// dropped rather than run with arguments that are not all there.
		s.finish(ev)
		if err := s.flushCalls(true); err != nil {
			return true, err
		}
		return true, responseError(ev)
	case evtError:
		return true, &serverError{msg: s.errorPrefix + ": " + firstNonEmpty(ev.errorMessage(), "Unknown error")}
	}
	return false, nil
}

// carriesModelOutput reports whether an event holds output the model produced.
// The events that open a response (response.created) only acknowledge the
// request, so TTFB ends at the first event that carries output: a chunk of
// text, or the item announcing a function call for a turn that only calls tools
// and so produces no text at all.
func carriesModelOutput(ev event) bool {
	switch ev.Type {
	case evtTextDelta, evtItemAdded:
		return true
	default:
		return false
	}
}

// trackCall folds the events that describe a function call into the pending
// call for its output index: the item that announces it, then its arguments.
func (s *streamState) trackCall(ev event) {
	switch ev.Type {
	case evtItemAdded:
		if ev.Item != nil && ev.Item.Type == itemFuncCall {
			s.calls[ev.OutputIndex] = &pendingCall{callID: ev.Item.CallID, name: ev.Item.Name}
		}
	case evtArgsDelta:
		if call, ok := s.calls[ev.OutputIndex]; ok {
			call.args += ev.Delta
		}
	case evtArgsDone:
		if call, ok := s.calls[ev.OutputIndex]; ok {
			call.done = true
			if ev.Arguments != "" {
				// The done event carries the complete arguments, so it supersedes
				// the fragments rather than appending to them.
				call.args = ev.Arguments
			}
		}
	case evtItemDone:
		// The completed item carries the complete arguments too, which is where
		// a call whose announcement was held back until now gets them.
		if call, ok := s.calls[ev.OutputIndex]; ok && ev.Item != nil {
			call.done = true
			if ev.Item.Arguments != "" {
				call.args = ev.Item.Arguments
			}
		}
	}
}

// finish records what a terminal event reported.
func (s *streamState) finish(ev event) {
	if ev.Response == nil {
		return
	}
	if ev.Response.ID != "" {
		s.responseID = ev.Response.ID
	}
	if ev.Response.Usage != nil {
		s.usage = ev.Response.Usage
	}
}

// flushCalls reports the function calls the model requested, in output order,
// or with onlyDone only those whose arguments an event said were complete. The
// calls are keyed by output index because their argument fragments arrive
// interleaved, so they are sorted back into the order the model emitted them.
func (s *streamState) flushCalls(onlyDone bool) error {
	indices := slices.Sorted(maps.Keys(s.calls))
	for _, i := range indices {
		call := s.calls[i]
		if onlyDone && !call.done {
			delete(s.calls, i)
			continue
		}
		args := call.args
		if args == "" {
			args = "{}"
		}
		delete(s.calls, i)
		s.sawCall = true
		if err := s.sink.Tool(frames.ToolCall{
			ID:   call.callID,
			Name: call.name,
			Args: json.RawMessage(args),
		}); err != nil {
			return err
		}
	}
	return nil
}

// serverError is a failure the API reported on the stream, worded as it is
// reported to the pipeline. It wraps errServer.
type serverError struct{ msg string }

func (e *serverError) Error() string { return e.msg }

func (e *serverError) Unwrap() error { return errServer }

// responseError describes a response the model failed to finish: the server's
// message for one that failed, the reason for one cut short, or else a generic
// account of which it was.
func responseError(ev event) error {
	reason := "Response failed"
	if ev.Type == evtIncomplete {
		reason = "Response incomplete"
	}
	if r := ev.Response; r != nil {
		switch {
		case r.Error != nil && r.Error.Message != "":
			reason = r.Error.Message
		case r.IncompleteDetails != nil && r.IncompleteDetails.Reason != "":
			reason = r.IncompleteDetails.Reason
		}
	}
	return &serverError{msg: "LLM response error: " + reason}
}

// textSink adapts an Emit to the Sink interface, for generation without tools.
type textSink struct{ emit llm.Emit }

func (s textSink) Text(text string) error { return s.emit(text) }

// Tool discards a call: a conversation with no tools cannot receive one, and the
// model was never told about any.
func (textSink) Tool(frames.ToolCall) error { return nil }

// firstNonEmpty returns the first non-empty string, or "" when both are empty.
func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// decodeEvent parses one stream event, reporting whether it was well-formed.
// A malformed event is skipped rather than ending the turn.
func decodeEvent(data []byte) (event, bool) {
	var ev event
	if json.Unmarshal(data, &ev) != nil {
		return event{}, false
	}
	return ev, true
}
