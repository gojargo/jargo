package responses

import (
	"reflect"
	"strings"
	"testing"

	responsesadapter "github.com/gojargo/jargo/adapter/responses"
	"github.com/gojargo/jargo/frames"
)

// feedPushing folds events into a state machine that records what it pushes.
func feedPushing(t *testing.T, events ...string) []frames.Frame {
	t.Helper()
	var pushed []frames.Frame
	state := newStreamState(&recordingSink{}, "Responses API error")
	state.push = func(f frames.Frame) error {
		pushed = append(pushed, f)
		return nil
	}
	for _, raw := range events {
		if _, err := state.handle(decodeTestEvent(t, raw)); err != nil {
			t.Fatalf("handle: %v", err)
		}
	}
	return pushed
}

const reasoningCompleted = `{"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.4-mini"}}`

// Ported from upstream (TestHttpReasoningCapture and
// TestReceiveResponseEventsReasoning::test_summary_streamed_and_reasoning_item_persisted).
// The streamed summary becomes a thought, and the reasoning item is appended to
// the conversation, encrypted reasoning and all, so it goes back to the model.
func TestReasoningSummaryStreamedAndItemPersisted(t *testing.T) {
	pushed := feedPushing(t,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}`,
		`{"type":"response.reasoning_summary_text.delta","delta":"Think"}`,
		`{"type":"response.reasoning_summary_text.delta","delta":"ing..."}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1",`+
			`"summary":[{"type":"summary_text","text":"Thinking..."}],"encrypted_content":"ENCRYPTED"}}`,
		reasoningCompleted,
	)

	var starts, ends int
	var texts []string
	var appends []*frames.LLMMessagesAppendFrame
	for _, f := range pushed {
		switch f := f.(type) {
		case *frames.LLMThoughtStartFrame:
			starts++
		case *frames.LLMThoughtTextFrame:
			texts = append(texts, f.Text)
		case *frames.LLMThoughtEndFrame:
			ends++
		case *frames.LLMMessagesAppendFrame:
			appends = append(appends, f)
		}
	}
	if starts != 1 || ends != 1 || strings.Join(texts, "|") != "Think|ing..." {
		t.Errorf("thought start=%d text=%v end=%d, want 1, [Think ing...], 1", starts, texts, ends)
	}
	if len(appends) != 1 || len(appends[0].Messages) != 1 {
		t.Fatalf("appended %d frames, want the reasoning item once", len(appends))
	}
	msg := appends[0].Messages[0]
	want := responsesadapter.InputItem{
		Type: "reasoning", ID: "rs_1",
		Summary:          []responsesadapter.SummaryPart{{Type: "summary_text", Text: "Thinking..."}},
		EncryptedContent: "ENCRYPTED",
	}
	if msg.LLM != "openai_responses" || !reflect.DeepEqual(msg.Native, want) {
		t.Errorf("appended %+v, want %+v as the adapter's own message", msg, want)
	}
}

// Ported from upstream (test_reasoning_without_encrypted_content_not_persisted).
// With no encrypted reasoning (effort none, say) there is nothing to carry
// over.
func TestReasoningWithoutEncryptedContentNotPersisted(t *testing.T) {
	pushed := feedPushing(t,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
		reasoningCompleted,
	)
	for _, f := range pushed {
		if _, ok := f.(*frames.LLMMessagesAppendFrame); ok {
			t.Error("a reasoning item with no encrypted reasoning was appended")
		}
	}
}

// Ported from upstream (TestStartsWithResponseOutputReasoning). A reasoning
// item in the output is matched on its id, so a turn that reasoned can still
// be continued from.
func TestStartsWithResponseOutputReasoning(t *testing.T) {
	output := outputOf(t, `[{"type":"reasoning","id":"rs_1","encrypted_content":"ENC"}]`)
	items := []inputItem{
		{Type: itemReasoning, ID: "rs_1", EncryptedContent: "ENC"},
		{Type: itemMessage, Role: "user", Content: "next"},
	}
	if !startsWithResponseOutput(items, output) {
		t.Error("a reasoning item with the same id did not match")
	}
	if startsWithResponseOutput([]inputItem{{Type: itemReasoning, ID: "rs_2"}}, output) {
		t.Error("a reasoning item with another id matched")
	}
}

// Ported from upstream (test_gpt_6_1_sol_explicit_effort_honored,
// test_explicit_reasoning_overrides_default and
// test_empty_reasoning_config_falls_back_to_default). Configured reasoning asks
// for the encrypted reasoning; the default effort does not.
func TestConfiguredReasoningIncludesEncryptedContent(t *testing.T) {
	req := mustRequest(t, Config{Model: "gpt-6.1-sol", Reasoning: &ReasoningConfig{Effort: "low"}}, convo())
	if !reflect.DeepEqual(req.Include, []string{"reasoning.encrypted_content"}) {
		t.Errorf("include = %v, want [reasoning.encrypted_content]", req.Include)
	}
	for _, cfg := range []Config{
		{Model: "gpt-5-mini"},
		{Model: "gpt-5-mini", Reasoning: &ReasoningConfig{}},
		{Model: "gpt-5-chat-latest"},
	} {
		if req := mustRequest(t, cfg, convo()); req.Include != nil {
			t.Errorf("%s with reasoning %+v: include = %v, want none", cfg.Model, cfg.Reasoning, req.Include)
		}
	}
}
