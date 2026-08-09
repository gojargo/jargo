package perplexity

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
)

// msg builds a message with plain text content.
func msg(role, content string) openai.Message {
	return openai.Message{Role: role, Content: content}
}

// roles returns the role of each message, which is what most of these tests
// compare.
func roles(msgs []openai.Message) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Role)
	}
	return out
}

// wantRoles fails unless the messages carry exactly the given roles in order.
func wantRoles(t *testing.T, msgs []openai.Message, want ...string) {
	t.Helper()
	got := roles(msgs)
	if len(got) != len(want) {
		t.Fatalf("roles = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("message %d role = %q, want %q", i, got[i], want[i])
		}
	}
}

// texts returns the text a message carries, whichever way it carries it.
func texts(m openai.Message) []string {
	out := make([]string, 0, len(m.Parts()))
	for _, p := range m.Parts() {
		out = append(out, p.Text)
	}
	return out
}

// wantTexts fails unless the message carries exactly the given texts in order.
func wantTexts(t *testing.T, m openai.Message, want ...string) {
	t.Helper()
	got := texts(m)
	if len(got) != len(want) {
		t.Fatalf("texts = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("text %d = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestStandardMessagesPassThrough checks a conversation that already alternates
// is left as it is.
func TestStandardMessagesPassThrough(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleSystem, "You are helpful."),
		msg(openai.RoleUser, "Hello"),
		msg(openai.RoleAssistant, "Hi"),
		msg(openai.RoleUser, "How are you?"),
	})
	wantRoles(t, out, openai.RoleSystem, openai.RoleUser, openai.RoleAssistant, openai.RoleUser)
	if out[1].Content != "Hello" {
		t.Errorf("content = %q, want it untouched", out[1].Content)
	}
}

// TestInitialSystemMessagePreserved checks the leading system message stays the
// system message it was.
func TestInitialSystemMessagePreserved(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleSystem, "You are helpful."),
		msg(openai.RoleUser, "Hello"),
	})
	wantRoles(t, out, openai.RoleSystem, openai.RoleUser)
}

// TestMultipleInitialSystemMessagesPreserved checks the leading run of system
// messages is neither demoted nor merged: Perplexity accepts several, and
// merging them would rewrite a conversation it already reads.
func TestMultipleInitialSystemMessagesPreserved(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleSystem, "Be brief."),
		msg(openai.RoleSystem, "Be polite."),
		msg(openai.RoleUser, "Hello"),
	})
	wantRoles(t, out, openai.RoleSystem, openai.RoleSystem, openai.RoleUser)
	if out[0].Content != "Be brief." || out[1].Content != "Be polite." {
		t.Errorf("messages = %+v, want the opening system block kept apart", out[:2])
	}
}

// TestNonInitialSystemConvertedToUser checks a system message after any other
// message is sent as a user message, which is the only way Perplexity takes one.
func TestNonInitialSystemConvertedToUser(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleSystem, "You are helpful."),
		msg(openai.RoleUser, "Hello"),
		msg(openai.RoleAssistant, "Hi"),
		msg(openai.RoleSystem, "Be concise."),
		msg(openai.RoleUser, "Tell me more"),
	})
	wantRoles(t, out, openai.RoleSystem, openai.RoleUser, openai.RoleAssistant, openai.RoleUser)
	// The demoted system message merges with the user message following it.
	wantTexts(t, out[3], "Be concise.", "Tell me more")
}

// TestConsecutiveSameRoleMerged checks two messages of one role become one
// carrying both contents, which is what strict alternation requires.
func TestConsecutiveSameRoleMerged(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleSystem, "You are helpful."),
		msg(openai.RoleUser, "Hello"),
		msg(openai.RoleUser, "Are you there?"),
		msg(openai.RoleAssistant, "Hi"),
		msg(openai.RoleUser, "Good"),
	})
	wantRoles(t, out, openai.RoleSystem, openai.RoleUser, openai.RoleAssistant, openai.RoleUser)
	wantTexts(t, out[1], "Hello", "Are you there?")
}

// TestConsecutiveAssistantsMergedThenTrailingRemoved checks the two rewrites
// compose: the run is merged first, and the merged message is then dropped for
// ending the conversation.
func TestConsecutiveAssistantsMergedThenTrailingRemoved(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleUser, "Hello"),
		msg(openai.RoleAssistant, "Hi"),
		msg(openai.RoleAssistant, "How can I help?"),
	})
	wantRoles(t, out, openai.RoleUser)
}

// TestTrailingAssistantRemoved checks a conversation ending on an assistant
// message has it dropped, which is what OpenAI does with one server-side.
func TestTrailingAssistantRemoved(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleUser, "Hello"),
		msg(openai.RoleAssistant, "Hi"),
	})
	wantRoles(t, out, openai.RoleUser)
}

// TestSystemExposedAfterTrailingAssistantRemoved checks the system message a
// dropped assistant message was hiding is left as a system message, not demoted.
// Demoting it would depend on how much of the conversation had happened, and
// Perplexity rejects a message whose role changes between turns.
func TestSystemExposedAfterTrailingAssistantRemoved(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleSystem, "You are helpful."),
		msg(openai.RoleAssistant, "Hi"),
	})
	wantRoles(t, out, openai.RoleSystem)
}

// TestOnlySystemMessagesPreserved checks a conversation of nothing but system
// messages is left alone. The API refuses it, which reports the mistake straight
// away rather than the rewrite hiding it.
func TestOnlySystemMessagesPreserved(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleSystem, "Be brief."),
		msg(openai.RoleSystem, "Be polite."),
	})
	wantRoles(t, out, openai.RoleSystem, openai.RoleSystem)
}

// TestToolMessagesPreserved checks a tool turn survives: a tool message may end
// the conversation, and it does not merge with the assistant message that asked
// for it.
func TestToolMessagesPreserved(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleUser, "weather?"),
		{Role: openai.RoleAssistant, ToolCalls: []openai.ToolCall{{ID: "call_a"}}},
		{Role: openai.RoleTool, ToolCallID: "call_a", Content: "sunny"},
	})
	wantRoles(t, out, openai.RoleUser, openai.RoleAssistant, openai.RoleTool)
	if out[2].ToolCallID != "call_a" || out[2].Content != "sunny" {
		t.Errorf("tool message = %+v, want it untouched", out[2])
	}
}

// TestEmptyMessages checks an empty conversation is returned as it came.
func TestEmptyMessages(t *testing.T) {
	if got := TransformMessages(nil); got != nil {
		t.Errorf("TransformMessages(nil) = %+v, want nil", got)
	}
	if got := TransformMessages([]openai.Message{}); len(got) != 0 {
		t.Errorf("TransformMessages(empty) = %+v, want empty", got)
	}
}

// TestSourceMessagesNotMutated checks the rewrite leaves the caller's messages
// as they were: the demotion writes roles, and the slice is shared with whoever
// converted the conversation.
func TestSourceMessagesNotMutated(t *testing.T) {
	in := []openai.Message{
		msg(openai.RoleSystem, "You are helpful."),
		msg(openai.RoleUser, "Hello"),
		msg(openai.RoleSystem, "Be concise."),
	}
	TransformMessages(in)
	if in[2].Role != openai.RoleSystem {
		t.Errorf("source message role = %q, want it left as the system role", in[2].Role)
	}
}

// TestDeveloperMessageConvertedToUser checks the developer role is settled by
// the embedded adapter, which the service tells that this endpoint has none.
func TestDeveloperMessageConvertedToUser(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddUserMessage("Hello")

	p, err := (&Adapter{}).LLMInvocationParams(
		convo, adapter.Options{ConvertDeveloperToUser: true},
	)
	if err != nil {
		t.Fatalf("LLMInvocationParams: %v", err)
	}
	// Both became user messages, so they merge into one.
	wantRoles(t, p.Messages, openai.RoleUser)
	wantTexts(t, p.Messages[0], "Extra context.", "Hello")
}

// TestMergedContentGoesOutAsParts checks a merged message reaches the wire as a
// list of parts, so what each of the two messages said stays distinct rather
// than their text running together.
func TestMergedContentGoesOutAsParts(t *testing.T) {
	out := TransformMessages([]openai.Message{
		msg(openai.RoleUser, "Hello"),
		msg(openai.RoleUser, "Are you there?"),
	})
	raw, err := json.Marshal(out[0])
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	want := `{"content":[{"type":"text","text":"Hello"},` +
		`{"type":"text","text":"Are you there?"}],"role":"user"}`
	if string(raw) != want {
		t.Errorf("encoded = %s, want %s", raw, want)
	}
}

// TestConversionThroughAdapter checks the whole path: a conversation the
// endpoint would reject is converted and rewritten into one it takes.
func TestConversionThroughAdapter(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	convo.AddUserMessage("Are you there?")
	convo.AddAssistantMessage("Hi")

	p, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("LLMInvocationParams: %v", err)
	}
	wantRoles(t, p.Messages, openai.RoleSystem, openai.RoleUser)
	wantTexts(t, p.Messages[1], "Hello", "Are you there?")
}
