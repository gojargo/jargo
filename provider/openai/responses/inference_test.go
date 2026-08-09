package responses

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// TestRunInferenceAnswersOnce checks the one-shot path: a plain request that
// does not stream, the instruction and bound this inference was given, and the
// output text handed straight back.
func TestRunInferenceAnswersOnce(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		_, _ = w.Write([]byte(
			`{"output":[{"content":[{"type":"output_text","text":"a short summary"}]}]}`,
		))
	}))
	t.Cleanup(srv.Close)

	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL})
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
		t.Errorf("answer = %q, want the output text", got)
	}
	if body["stream"] != false {
		t.Errorf("stream = %v, want the one-shot path not to stream", body["stream"])
	}
	if body["max_output_tokens"] != float64(64) {
		t.Errorf("max_output_tokens = %v, want the bound this inference was given", body["max_output_tokens"])
	}
	if body["instructions"] != "Summarize the conversation." {
		t.Errorf("instructions = %v, want it to stand in for the conversation's own", body["instructions"])
	}
}

// TestBothServicesRunOneShotInferences checks each satisfies the interface a
// summarizer or a judge asks for. The WebSocket service answers over HTTP like
// the other one: a one-shot inference is a plain request, so the connection it
// holds open for its turns is left to them.
func TestBothServicesRunOneShotInferences(t *testing.T) {
	var _ llm.Inferencer = (*HTTPService)(nil)
	var _ llm.Inferencer = (*Service)(nil)
}

// TestWebSocketServiceAnswersOneShotOverHTTP checks the WebSocket service does
// not need its connection to answer a one-shot inference.
func TestWebSocketServiceAnswersOneShotOverHTTP(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"output":[{"content":[{"type":"output_text","text":"answered"}]}]}`))
	}))
	t.Cleanup(srv.Close)

	svc := NewLLM(Config{APIKey: "k", BaseURL: srv.URL, WSURL: "ws://127.0.0.1:1"})
	got, err := svc.RunInference(t.Context(), frames.NewLLMContext(""), llm.InferenceOptions{})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got != "answered" {
		t.Errorf("answer = %q, want the output text", got)
	}
}
