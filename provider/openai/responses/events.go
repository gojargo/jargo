package responses

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// Event types the services read. The Responses API streams the same events over
// SSE and over the WebSocket, so both transports share this decoding.
const (
	// evtCreated announces the response's id, which the WebSocket service needs
	// both to cancel it and to continue from it next turn.
	evtCreated = "response.created"
	// evtTextDelta is a chunk of generated text.
	evtTextDelta = "response.output_text.delta"
	// evtItemAdded announces an output item; a function call arrives this way,
	// carrying the call id and the function name.
	evtItemAdded = "response.output_item.added"
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
	// Message and Code describe a stream-level error.
	Message string `json:"message"`
	Code    string `json:"code"`
}

// outputItem is an item the model produced: a message or a function call.
type outputItem struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
}

// responseObject is the response a lifecycle event reports on.
type responseObject struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Error  *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Usage *usage `json:"usage"`
}

// usage is the Responses API's per-response token accounting.
type usage struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
}

// tokenUsage converts the wire accounting into the framework's usage shape.
func (u usage) tokenUsage() frames.LLMTokenUsage {
	out := frames.LLMTokenUsage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.InputTokensDetails != nil {
		out.CacheReadTokens = new(u.InputTokensDetails.CachedTokens)
	}
	return out
}

// pendingCall accumulates a function call as its arguments stream in.
type pendingCall struct {
	callID string
	name   string
	args   string
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
	// responseID is the id the response announced, for continuation and cancel.
	responseID string
	// usage is the token accounting the terminal event reported.
	usage *usage
	// sawText records whether the model produced any text, and callCount how
	// many function calls it requested. Together they say how many input items
	// this response will add to the conversation, which is what the next turn
	// must skip when continuing from it.
	sawText   bool
	callCount int
}

// outputItems is how many input items this response contributes to the
// conversation: one message when it produced text, plus one per function call.
// It mirrors how buildInput renders an assistant turn.
func (s *streamState) outputItems() int {
	n := s.callCount
	if s.sawText {
		n++
	}
	return n
}

// newStreamState builds a state machine feeding sink.
func newStreamState(sink llm.Sink) *streamState {
	return &streamState{sink: sink, calls: map[int]*pendingCall{}}
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
			s.sawText = true
			return false, s.sink.Text(ev.Delta)
		}
	case evtItemAdded, evtArgsDelta, evtArgsDone:
		s.trackCall(ev)
	case evtCompleted, evtIncomplete:
		s.finish(ev)
		return true, s.flushCalls()
	case evtFailed:
		s.finish(ev)
		return true, responseError(ev)
	case evtError:
		return true, fmt.Errorf("%w: %s", errServer, firstNonEmpty(ev.Message, ev.Code))
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
			s.callCount++
		}
	case evtArgsDelta:
		if call, ok := s.calls[ev.OutputIndex]; ok {
			call.args += ev.Delta
		}
	case evtArgsDone:
		if call, ok := s.calls[ev.OutputIndex]; ok && ev.Arguments != "" {
			// The done event carries the complete arguments, so it supersedes the
			// fragments rather than appending to them.
			call.args = ev.Arguments
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

// flushCalls reports every function call the model requested, in output order.
// The calls are keyed by output index because their argument fragments arrive
// interleaved, so they are sorted back into the order the model emitted them.
func (s *streamState) flushCalls() error {
	indices := slices.Sorted(maps.Keys(s.calls))
	for _, i := range indices {
		call := s.calls[i]
		args := call.args
		if args == "" {
			args = "{}"
		}
		delete(s.calls, i)
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

// responseError describes a response the model failed to produce.
func responseError(ev event) error {
	if ev.Response == nil {
		return fmt.Errorf("%w: response failed", errServer)
	}
	if ev.Response.Error != nil {
		return fmt.Errorf("%w: %s", errServer, firstNonEmpty(ev.Response.Error.Message, ev.Response.Error.Code))
	}
	return fmt.Errorf("%w: response failed with status %q", errServer, ev.Response.Status)
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
