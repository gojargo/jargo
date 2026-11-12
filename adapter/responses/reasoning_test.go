package responses

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// itemJSON is an input item as it goes on the wire, decoded back into a map.
func itemJSON(t *testing.T, item InputItem) map[string]any {
	t.Helper()
	raw, err := json.Marshal(item)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return got
}

// Ported from upstream (test_reasoning_message_becomes_reasoning_item).
func TestReasoningMessageBecomesReasoningItem(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(NewReasoningMessage("rs_1",
		[]SummaryPart{{Type: SummaryTypeText, Text: "Let me think."}}, "ENCRYPTED"))

	got := itemJSON(t, paramsOf(t, convo, adapter.Options{}).Input[0])
	want := map[string]any{
		"type":              "reasoning",
		"id":                "rs_1",
		"summary":           []any{map[string]any{"type": "summary_text", "text": "Let me think."}},
		"encrypted_content": "ENCRYPTED",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("item = %v, want %v", got, want)
	}
}

// Ported from upstream (test_reasoning_precedes_assistant_message).
func TestReasoningPrecedesAssistantMessage(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(NewReasoningMessage("rs_1", nil, "ENCRYPTED"))
	convo.AddAssistantMessage("Hello!")

	items := paramsOf(t, convo, adapter.Options{}).Input
	if len(items) != 2 || items[0].Type != ItemReasoning || items[1].Role != RoleAssistant {
		t.Errorf("input = %+v, want the reasoning item then the assistant message", items)
	}
}

// Ported from upstream (test_reasoning_without_encrypted_content_omits_field).
// The summary is sent even when empty: the API requires one.
func TestReasoningWithoutEncryptedContentOmitsField(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(NewReasoningMessage("rs_2", nil, ""))

	got := itemJSON(t, paramsOf(t, convo, adapter.Options{}).Input[0])
	if got["type"] != "reasoning" || got["id"] != "rs_2" {
		t.Errorf("item = %v, want reasoning rs_2", got)
	}
	if _, ok := got["encrypted_content"]; ok {
		t.Errorf("item = %v carries encrypted_content", got)
	}
	if summary, ok := got["summary"].([]any); !ok || len(summary) != 0 {
		t.Errorf("summary = %v, want an empty list", got["summary"])
	}
}

// reasoningConvo is a turn whose reply came with a reasoning item.
func reasoningConvo(encrypted string) *frames.LLMContext {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(NewReasoningMessage("rs_123",
		[]SummaryPart{{Type: SummaryTypeText, Text: "thinking about the weather"}}, encrypted))
	convo.AddAssistantMessage("hi")
	return convo
}

// Ported from upstream (test_reasoning_encrypted_content_is_elided,
// test_messages_for_logging_are_json_serializable and
// test_standard_messages_pass_through).
func TestReasoningEncryptedContentIsElidedFromLogs(t *testing.T) {
	encrypted := strings.Repeat("gAAAA", 300)
	convo := reasoningConvo(encrypted)
	msgs := (&Adapter{}).MessagesForLogging(convo)
	if len(msgs) != 3 {
		t.Fatalf("messages = %v, want 3", msgs)
	}
	reasoning := msgs[1]
	if reasoning["type"] != "reasoning" || reasoning["id"] != "rs_123" ||
		reasoning["encrypted_content"] != "..." {
		t.Errorf("reasoning = %v, want rs_123 with its encrypted content elided", reasoning)
	}
	if msgs[0]["role"] != RoleUser || msgs[2]["role"] != RoleAssistant {
		t.Errorf("messages = %v, want the user and assistant messages around it", msgs)
	}
	if _, err := json.Marshal(msgs); err != nil {
		t.Errorf("messages are not serializable: %v", err)
	}

	// Ported from upstream (test_context_not_mutated_by_elision).
	stored, ok := convo.MessagesFor(LLMSpecificID)[1].Native.(InputItem)
	if !ok || stored.EncryptedContent != encrypted {
		t.Error("eliding for the log changed the conversation")
	}
}

// Ported from upstream (test_reasoning_without_encrypted_content_is_untouched).
func TestReasoningWithoutEncryptedContentLoggedAsIs(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(NewReasoningMessage("rs_9", nil, ""))
	msgs := (&Adapter{}).MessagesForLogging(convo)
	want := map[string]any{"type": "reasoning", "id": "rs_9", "summary": []any{}}
	if len(msgs) != 1 || !reflect.DeepEqual(msgs[0], want) {
		t.Errorf("messages = %v, want [%v]", msgs, want)
	}
}
