package mistral

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// answeredConvo is a conversation in which call_a has been made and answered
// and call_b has not been made at all.
func answeredConvo() *frames.LLMContext {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("weather in Paris?")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "call_a", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "call_a", Name: "get_weather", Content: "sunny"})
	return convo
}

func toolCall(id, name string) frames.ToolCall {
	return frames.ToolCall{ID: id, Name: name, Args: json.RawMessage(`{}`)}
}

// TestDropAnsweredCalls checks the completion that answers a tool result does
// not run that tool a second time, while a call the conversation has no result
// for is left alone.
func TestDropAnsweredCalls(t *testing.T) {
	convo := answeredConvo()

	kept := dropAnsweredCalls(convo, []frames.ToolCall{
		toolCall("call_a", "get_weather"),
		toolCall("call_b", "get_time"),
	})
	if len(kept) != 1 || kept[0].ID != "call_b" {
		t.Errorf("kept = %+v, want only the call the conversation has no result for", kept)
	}

	if kept := dropAnsweredCalls(convo, []frames.ToolCall{toolCall("call_a", "get_weather")}); len(kept) != 0 {
		t.Errorf("kept = %+v, want nothing: every call was answered already", kept)
	}

	fresh := []frames.ToolCall{toolCall("call_c", "get_time")}
	if kept := dropAnsweredCalls(frames.NewLLMContext(""), fresh); len(kept) != 1 {
		t.Errorf("kept = %+v, want the call: an empty conversation has answered nothing", kept)
	}

	// An id is what pairs a call to its result. Without one there is nothing to
	// match, so the call runs rather than being swallowed by the first result.
	if kept := dropAnsweredCalls(convo, []frames.ToolCall{toolCall("", "get_time")}); len(kept) != 1 {
		t.Errorf("kept = %+v, want a call with no id to run", kept)
	}
}

// TestShapeMessagesFollowsATooResultWithAnAssistantMessage checks the first of
// Mistral's constraints: a tool result that ends the conversation, which is
// exactly what the completion answering a tool call sees.
func TestShapeMessagesFollowsATooResultWithAnAssistantMessage(t *testing.T) {
	out := shapeMessages([]chat.Message{
		{Role: chat.RoleUser, Content: "weather?"},
		{Role: chat.RoleAssistant, ToolCalls: []chat.ToolCall{{ID: "call_a"}}},
		{Role: chat.RoleTool, ToolCallID: "call_a", Content: "sunny"},
	})

	if len(out) != 4 {
		t.Fatalf("messages = %+v, want an assistant message inserted after the tool result", out)
	}
	if out[3].Role != chat.RoleAssistant || out[3].Content != " " {
		t.Errorf("inserted message = %+v, want a minimal assistant message", out[3])
	}
	// It now trails the conversation, so it is a partial reply to continue.
	if out[3].Extra["prefix"] != true {
		t.Errorf("trailing assistant message = %+v, want it marked as a prefix", out[3])
	}

	// A tool result already followed by an assistant message needs no insertion.
	out = shapeMessages([]chat.Message{
		{Role: chat.RoleTool, ToolCallID: "call_a", Content: "sunny"},
		{Role: chat.RoleAssistant, Content: "it is sunny"},
		{Role: chat.RoleUser, Content: "thanks"},
	})
	if len(out) != 3 {
		t.Errorf("messages = %+v, want no insertion", out)
	}
}

// TestShapeMessagesDemotesLateSystemMessages checks the second constraint: only
// the leading run of system messages is accepted.
func TestShapeMessagesDemotesLateSystemMessages(t *testing.T) {
	out := shapeMessages([]chat.Message{
		{Role: chat.RoleSystem, Content: "be brief"},
		{Role: chat.RoleSystem, Content: "and polite"},
		{Role: chat.RoleUser, Content: "hello"},
		{Role: chat.RoleSystem, Content: "the user is in a hurry"},
	})

	if out[0].Role != chat.RoleSystem || out[1].Role != chat.RoleSystem {
		t.Errorf("leading messages = %+v, want the opening system block kept", out[:2])
	}
	if out[3].Role != chat.RoleUser {
		t.Errorf("late system message = %+v, want it sent as a user message", out[3])
	}
	if out[3].Content != "the user is in a hurry" {
		t.Errorf("content = %q, want what the system message carried", out[3].Content)
	}
}

// TestShapeMessagesMarksATrailingAssistantMessage checks the third constraint,
// and that a conversation ending any other way is left as it is.
func TestShapeMessagesMarksATrailingAssistantMessage(t *testing.T) {
	out := shapeMessages([]chat.Message{
		{Role: chat.RoleUser, Content: "hello"},
		{Role: chat.RoleAssistant, Content: "hi, I was saying"},
	})
	if out[1].Extra["prefix"] != true {
		t.Errorf("trailing assistant message = %+v, want it marked as a prefix", out[1])
	}

	out = shapeMessages([]chat.Message{
		{Role: chat.RoleAssistant, Content: "hi"},
		{Role: chat.RoleUser, Content: "hello"},
	})
	if _, ok := out[1].Extra["prefix"]; ok {
		t.Errorf("trailing user message = %+v, want nothing added to it", out[1])
	}
	if _, ok := out[0].Extra["prefix"]; ok {
		t.Errorf("assistant message mid-conversation = %+v, want nothing added to it", out[0])
	}

	if got := shapeMessages(nil); got != nil {
		t.Errorf("shapeMessages(nil) = %+v, want nil", got)
	}
}
