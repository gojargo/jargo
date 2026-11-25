package mistral

import (
	"testing"

	"github.com/gojargo/jargo/provider/openai/chat"
)

// TestTransformMessagesFollowsATooResultWithAnAssistantMessage checks the first of
// Mistral's constraints: a tool result that ends the conversation, which is
// exactly what the completion answering a tool call sees.
func TestTransformMessagesFollowsATooResultWithAnAssistantMessage(t *testing.T) {
	out := TransformMessages([]chat.Message{
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
	out = TransformMessages([]chat.Message{
		{Role: chat.RoleTool, ToolCallID: "call_a", Content: "sunny"},
		{Role: chat.RoleAssistant, Content: "it is sunny"},
		{Role: chat.RoleUser, Content: "thanks"},
	})
	if len(out) != 3 {
		t.Errorf("messages = %+v, want no insertion", out)
	}
}

// TestTransformMessagesDemotesLateSystemMessages checks the second constraint: only
// the leading run of system messages is accepted.
func TestTransformMessagesDemotesLateSystemMessages(t *testing.T) {
	out := TransformMessages([]chat.Message{
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

// TestTransformMessagesMarksATrailingAssistantMessage checks the third constraint,
// and that a conversation ending any other way is left as it is.
func TestTransformMessagesMarksATrailingAssistantMessage(t *testing.T) {
	out := TransformMessages([]chat.Message{
		{Role: chat.RoleUser, Content: "hello"},
		{Role: chat.RoleAssistant, Content: "hi, I was saying"},
	})
	if out[1].Extra["prefix"] != true {
		t.Errorf("trailing assistant message = %+v, want it marked as a prefix", out[1])
	}

	out = TransformMessages([]chat.Message{
		{Role: chat.RoleAssistant, Content: "hi"},
		{Role: chat.RoleUser, Content: "hello"},
	})
	if _, ok := out[1].Extra["prefix"]; ok {
		t.Errorf("trailing user message = %+v, want nothing added to it", out[1])
	}
	if _, ok := out[0].Extra["prefix"]; ok {
		t.Errorf("assistant message mid-conversation = %+v, want nothing added to it", out[0])
	}

	if got := TransformMessages(nil); got != nil {
		t.Errorf("TransformMessages(nil) = %+v, want nil", got)
	}
}
