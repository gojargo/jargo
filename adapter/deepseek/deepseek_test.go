package deepseek

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
)

// toolCall is the call the tool-call tests have the model make.
func toolCall() frames.ToolCall {
	return frames.ToolCall{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"location": "LA"}`)}
}

// invoke converts convo through the adapter, failing the test on an error.
func invoke(t *testing.T, convo *frames.LLMContext) []openai.Message {
	t.Helper()
	// DeepSeek has no developer role, so the service converts developer
	// messages to user ones. The tests convert the same way to match what is
	// sent.
	p, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{ConvertDeveloperToUser: true})
	if err != nil {
		t.Fatalf("LLMInvocationParams: %v", err)
	}
	return p.Messages
}

// reasoning reports the reasoning_content a message carries, and whether it
// carries one at all.
func reasoning(m openai.Message) (any, bool) {
	v, ok := m.Extra[reasoningContent]
	return v, ok
}

// An assistant tool-call message without reasoning_content gets an empty one.
func TestToolCallMessageGetsEmptyReasoningContent(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Weather in LA?")
	convo.AddAssistantToolCall(toolCall())
	convo.AddToolResult(frames.ToolResult{ID: "call_1", Name: "get_weather", Content: `{"temperature": "75"}`})

	msgs := invoke(t, convo)

	assistant := msgs[1]
	if assistant.Role != openai.RoleAssistant {
		t.Fatalf("message 1 role = %q, want assistant", assistant.Role)
	}
	if v, ok := reasoning(assistant); !ok || v != "" {
		t.Errorf("reasoning_content = %v (present %t), want an empty one", v, ok)
	}
	if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].ID != "call_1" ||
		assistant.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("tool_calls = %+v, want the call kept", assistant.ToolCalls)
	}
}

// Assistant text recorded before a tool call (for example spoken through TTS) is
// stamped too.
func TestSpokenTextBeforeToolCallAlsoStamped(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Weather in LA?")
	convo.AddAssistantMessage("Let me check on that.")
	convo.AddAssistantToolCall(toolCall())
	convo.AddToolResult(frames.ToolResult{ID: "call_1", Name: "get_weather", Content: `{"temperature": "75"}`})

	msgs := invoke(t, convo)

	if msgs[1].Content != "Let me check on that." {
		t.Errorf("message 1 content = %q, want the spoken text", msgs[1].Content)
	}
	for _, i := range []int{1, 2} {
		if v, ok := reasoning(msgs[i]); !ok || v != "" {
			t.Errorf("message %d reasoning_content = %v (present %t), want an empty one", i, v, ok)
		}
	}
}

// An assistant message that already carries reasoning_content is left alone.
func TestExistingReasoningContentPreserved(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Weather in LA?")
	convo.AddMessage(frames.NewLLMSpecificMessage("openai", openai.Message{
		Role:      openai.RoleAssistant,
		ToolCalls: []openai.ToolCall{{ID: "call_1", Type: "function"}},
		Extra:     map[string]any{reasoningContent: "Need the weather tool."},
	}))

	msgs := invoke(t, convo)

	if v, _ := reasoning(msgs[1]); v != "Need the weather tool." {
		t.Errorf("reasoning_content = %v, want the one the message carried", v)
	}
}

// System, user and tool messages never get the field.
func TestNonAssistantMessagesUntouched(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Weather in LA?")
	convo.AddAssistantToolCall(toolCall())
	convo.AddToolResult(frames.ToolResult{ID: "call_1", Name: "get_weather", Content: `{"temperature": "75"}`})
	convo.AddUserMessage("Thanks.")

	msgs := invoke(t, convo)

	for _, i := range []int{0, 1, 3, 4} {
		if v, ok := reasoning(msgs[i]); ok {
			t.Errorf("message %d (%s) reasoning_content = %v, want none", i, msgs[i].Role, v)
		}
	}
	if v, ok := reasoning(msgs[2]); !ok || v != "" {
		t.Errorf("assistant reasoning_content = %v (present %t), want an empty one", v, ok)
	}
}

// Stamping produces copies: the conversation's own messages are unchanged.
func TestContextMessagesNotMutated(t *testing.T) {
	native := openai.Message{
		Role:    openai.RoleAssistant,
		Content: "Hi!",
		Extra:   map[string]any{"name": "assistant"},
	}
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	convo.AddMessage(frames.NewLLMSpecificMessage("openai", native))

	msgs := invoke(t, convo)

	if v, ok := reasoning(msgs[1]); !ok || v != "" {
		t.Errorf("reasoning_content = %v (present %t), want an empty one", v, ok)
	}
	if _, ok := native.Extra[reasoningContent]; ok {
		t.Error("the message's own Extra map was written to")
	}
	held, ok := convo.Messages()[1].Native.(openai.Message)
	if !ok {
		t.Fatalf("context message holds %T, want an openai.Message", convo.Messages()[1].Native)
	}
	if _, ok := held.Extra[reasoningContent]; ok {
		t.Error("the conversation's message was written to")
	}
}

// The field reaches the wire, which is what DeepSeek reads it from.
func TestReasoningContentIsEncoded(t *testing.T) {
	raw, err := json.Marshal(AddReasoningContent([]openai.Message{
		{Role: openai.RoleAssistant, Content: "Hi!"},
	})[0])
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if v, ok := got[reasoningContent]; !ok || v != "" {
		t.Errorf("encoded message = %s, want an empty reasoning_content", raw)
	}
}
