package anthropic_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/service/llm"
)

// TestRunInferenceAnswersOnce checks the one-shot path: a single request that
// does not stream, the instruction it was given in place of the conversation's
// own, its own bound on the answer, and the text handed straight back.
func TestRunInferenceAnswersOnce(t *testing.T) {
	var body map[string]any
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
			"content": [{"type": "text", "text": "a short summary"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 4}
		}`))
	}))
	t.Cleanup(srv.Close)

	svc := anthropic.NewLLM(anthropic.Config{APIKey: "k", BaseURL: srv.URL})
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("what did we agree?")

	got, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{
		MaxTokens:         64,
		SystemInstruction: "Summarize the conversation.",
	})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got != "a short summary" {
		t.Errorf("answer = %q, want the message's text", got)
	}
	if requests != 1 {
		t.Errorf("requests = %d, want one", requests)
	}
	if stream, ok := body["stream"]; ok && stream == true {
		t.Errorf("body streams the one-shot path: %v", body)
	}
	if body["max_tokens"] != float64(64) {
		t.Errorf("max_tokens = %v, want the bound this inference was given", body["max_tokens"])
	}
	system, ok := body["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system = %v, want the one instruction this inference was given", body["system"])
	}
	block, _ := system[0].(map[string]any)
	if block["text"] != "Summarize the conversation." {
		t.Errorf("system = %v, want the instruction to stand in for the conversation's own", block)
	}
}

// TestServiceRunsOneShotInferences checks the service satisfies the interface a
// summarizer or a judge asks for.
func TestServiceRunsOneShotInferences(t *testing.T) {
	var _ llm.Inferencer = (*anthropic.Service)(nil)
}
