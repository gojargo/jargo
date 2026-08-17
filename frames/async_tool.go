package frames

import "encoding/json"

// This file is the single source of truth for the on-the-wire shape of the
// async-tool message protocol.
//
// When a tool is registered to survive interruptions (CancelOnInterruption
// false), the assistant context aggregator appends async-tool messages to the
// conversation as the underlying work progresses:
//
//   - a started message (a tool-result message) as soon as the tool starts,
//   - an intermediate message (a developer message) for each intermediate result
//     the handler reports,
//   - a final message (a developer message) when the work finishes.
//
// AsyncToolMessage is the canonical structured form; the JSON string carried in
// the message is always derived from it and never stored separately, so the two
// representations cannot drift.

// AsyncToolKind identifies which stage of the async-tool protocol a message is.
type AsyncToolKind string

const (
	// AsyncToolStarted marks the message appended when the tool starts running.
	AsyncToolStarted AsyncToolKind = "started"
	// AsyncToolIntermediate marks an intermediate result reported while the tool
	// is still running.
	AsyncToolIntermediate AsyncToolKind = "intermediate"
	// AsyncToolFinal marks the tool's final result.
	AsyncToolFinal AsyncToolKind = "final"
)

const (
	// AsyncToolPayloadType identifies an async-tool payload. Both the builders and
	// the parser use it; the literal is not duplicated anywhere else.
	AsyncToolPayloadType = "async_tool"
	// AsyncToolStatusRunning is the status of a started or intermediate message.
	AsyncToolStatusRunning = "running"
	// AsyncToolStatusFinished is the status of a final message.
	AsyncToolStatusFinished = "finished"
)

// ToolResultInProgress is the placeholder written as a tool call's result the
// moment the call starts, so the tool-use block is never left unanswered while
// the call runs. It is replaced in place once the call reports.
const ToolResultInProgress = "IN_PROGRESS"

// The descriptions shipped on each stage are deliberately self-explanatory, so a
// model reading the context can tell what is happening without out-of-band
// knowledge of the protocol.
const (
	asyncToolStartedDescription = "An asynchronous task associated with this tool_call_id has started " +
		"running. Expect results to arrive later as developer messages that look " +
		"roughly like this one (with 'type=async_tool' and a matching tool_call_id) " +
		"but with a 'result' field. Note that there *may* be more than one result " +
		"(i.e., a stream of results), but there doesn't have to be (there may be " +
		"only one). The last result will come in a message with 'status=finished'."

	asyncToolIntermediateDescription = "This is an intermediate result for the asynchronous task associated with " +
		"this tool_call_id. The task is still running. More intermediate results " +
		"may follow, or the next result may be the final one with " +
		"'status=finished'."

	asyncToolFinalDescription = "This is the final result for the asynchronous task associated with " +
		"this tool_call_id. The task has completed. No further results will arrive for " +
		"this tool_call_id."

	asyncToolCanceledDescription = "The asynchronous task associated with this tool_call_id was canceled " +
		"before it produced a result, either because it ran past its deadline or " +
		"because cancellation was requested. No further results will arrive for " +
		"this tool_call_id. If the user is still waiting on it, tell them it did " +
		"not complete rather than leaving it unanswered."
)

// asyncToolCanceledResult is the result a canceled call settles with. It names
// the tool call as the thing that ended: a bare marker says nothing about
// whatever the tool looks up, and a model relaying one will tell the user their
// flight, order or booking is off. The spelling is the protocol's, not prose.
//
//nolint:misspell // the literal written to the conversation
const asyncToolCanceledResult = "CANCELLED: this tool call was cancelled before it returned a result"

// AsyncToolMessage is the structured contents of an async-tool message.
type AsyncToolMessage struct {
	// Kind is which of the three stages this message is.
	Kind AsyncToolKind
	// ToolCallID is the id of the tool invocation the message relates to.
	ToolCallID string
	// Status is "running" for started and intermediate messages, "finished" for
	// the final one.
	Status string
	// Description is the human-readable description carried in the payload. It
	// may be empty.
	Description string
	// Result is the result string for intermediate and final messages, and empty
	// for a started message. HasResult distinguishes the two.
	Result string
	// HasResult reports whether the payload carried a result field at all.
	HasResult bool
}

// asyncToolPayload is the JSON form of an AsyncToolMessage. Result is a pointer
// so a started message omits it rather than carrying an empty string.
type asyncToolPayload struct {
	Type        string  `json:"type"`
	Status      string  `json:"status"`
	ToolCallID  string  `json:"tool_call_id"`
	Description string  `json:"description"`
	Result      *string `json:"result,omitempty"`
}

// json renders the message as its on-the-wire JSON string. Fields that do not
// apply to the message's kind are omitted, notably the result of a started
// message, whose task has not produced one yet.
func (m AsyncToolMessage) json() string {
	p := asyncToolPayload{
		Type:        AsyncToolPayloadType,
		Status:      m.Status,
		ToolCallID:  m.ToolCallID,
		Description: m.Description,
	}
	if m.HasResult {
		result := m.Result
		p.Result = &result
	}
	data, err := json.Marshal(p)
	if err != nil {
		return ""
	}
	return string(data)
}

// message wraps the payload in the conversation message shape its kind calls
// for. A started message is a tool-result message carrying the tool call's id,
// so it sits alongside ordinary tool results; an intermediate or final message
// is a developer message, whose id lives only inside the payload.
func (m AsyncToolMessage) message() Message {
	content := m.json()
	if m.Kind == AsyncToolStarted {
		return Message{
			Role:        RoleUser,
			ToolResults: []ToolResult{{ID: m.ToolCallID, Content: content}},
		}
	}
	return Message{Role: RoleDeveloper, Text: content}
}

// NewAsyncToolStartedMessage builds the message appended to the context as soon
// as an asynchronous tool call starts running. It tells the model that work is in
// flight and that its results arrive later as developer messages.
func NewAsyncToolStartedMessage(toolCallID string) Message {
	return AsyncToolMessage{
		Kind:        AsyncToolStarted,
		ToolCallID:  toolCallID,
		Status:      AsyncToolStatusRunning,
		Description: asyncToolStartedDescription,
	}.message()
}

// NewAsyncToolIntermediateMessage builds the message appended each time a running
// asynchronous tool reports a non-final result.
func NewAsyncToolIntermediateMessage(toolCallID, result string) Message {
	return AsyncToolMessage{
		Kind:        AsyncToolIntermediate,
		ToolCallID:  toolCallID,
		Status:      AsyncToolStatusRunning,
		Description: asyncToolIntermediateDescription,
		Result:      result,
		HasResult:   true,
	}.message()
}

// NewAsyncToolFinalMessage builds the message appended when an asynchronous tool
// finishes. No further async-tool messages arrive for the call after it. result
// is the tool's result, or "COMPLETED" when the handler produced none, which is
// the convention a synchronous call uses too.
func NewAsyncToolFinalMessage(toolCallID, result string) Message {
	return AsyncToolMessage{
		Kind:        AsyncToolFinal,
		ToolCallID:  toolCallID,
		Status:      AsyncToolStatusFinished,
		Description: asyncToolFinalDescription,
		Result:      result,
		HasResult:   true,
	}.message()
}

// NewAsyncToolCanceledMessage builds the message that settles an asynchronous
// call canceled before it returned a result, whether by its own deadline or at
// the model's request. It settles the tool call the same way a final result
// does, carrying a cancellation notice in place of one.
//
//nolint:misspell // the protocol's own spelling
func NewAsyncToolCanceledMessage(toolCallID string) Message {
	return AsyncToolMessage{
		Kind:        AsyncToolFinal,
		ToolCallID:  toolCallID,
		Status:      AsyncToolStatusFinished,
		Description: asyncToolCanceledDescription,
		Result:      asyncToolCanceledResult,
		HasResult:   true,
	}.message()
}

// ParseAsyncToolMessage decodes an async-tool payload out of a conversation
// message, reporting false when the message is not one. A realtime LLM service
// uses it to spot async-tool messages as it walks the context, so it can deliver
// the result through its own tool-result channel.
func ParseAsyncToolMessage(m Message) (AsyncToolMessage, bool) {
	content, ok := asyncToolContent(m)
	if !ok {
		return AsyncToolMessage{}, false
	}
	var p asyncToolPayload
	if err := json.Unmarshal([]byte(content), &p); err != nil {
		return AsyncToolMessage{}, false
	}
	if p.Type != AsyncToolPayloadType {
		return AsyncToolMessage{}, false
	}
	if p.ToolCallID == "" {
		return AsyncToolMessage{}, false
	}
	if p.Status != AsyncToolStatusRunning && p.Status != AsyncToolStatusFinished {
		return AsyncToolMessage{}, false
	}
	out := AsyncToolMessage{
		ToolCallID:  p.ToolCallID,
		Status:      p.Status,
		Description: p.Description,
	}
	switch {
	case p.Result == nil:
		out.Kind = AsyncToolStarted
	case p.Status == AsyncToolStatusFinished:
		out.Kind = AsyncToolFinal
		out.Result = *p.Result
		out.HasResult = true
	default:
		out.Kind = AsyncToolIntermediate
		out.Result = *p.Result
		out.HasResult = true
	}
	return out, true
}

// asyncToolContent returns the payload string a message may carry: the content of
// its single tool result for a started message, or the text of a developer
// message. Any other message shape cannot hold an async-tool payload.
func asyncToolContent(m Message) (string, bool) {
	if m.Role == RoleDeveloper {
		return m.Text, m.Text != ""
	}
	if len(m.ToolResults) == 1 {
		return m.ToolResults[0].Content, m.ToolResults[0].Content != ""
	}
	return "", false
}
