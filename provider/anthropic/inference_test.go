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

// TestRunInferenceSendsTheServiceInstructionAndTools checks an inference given
// no instruction of its own sends the service's, and carries the conversation's
// tools: the API refuses a conversation holding tool_use or tool_result blocks
// without them.
func TestRunInferenceSendsTheServiceInstructionAndTools(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
			"content": [{"type": "text", "text": "ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 1}
		}`))
	}))
	t.Cleanup(srv.Close)

	svc := anthropic.NewLLM(anthropic.Config{APIKey: "k", BaseURL: srv.URL})
	svc.SetSystemInstruction("You are a booking assistant.")
	convo := frames.NewLLMContext("")
	convo.SetTools([]frames.Tool{{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object"}`)}})
	convo.AddUserMessage("weather?")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{}`)})
	convo.AddToolResult(frames.ToolResult{ID: "c1", Content: "sunny"})

	if _, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{}); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	system, _ := body["system"].([]any)
	if len(system) != 1 {
		t.Fatalf("system = %v, want the service's instruction", body["system"])
	}
	if block, _ := system[0].(map[string]any); block["text"] != "You are a booking assistant." {
		t.Errorf("system = %v, want the service's instruction", block)
	}
	if tools, _ := body["tools"].([]any); len(tools) != 1 {
		t.Errorf("tools = %v, want the conversation's tool", body["tools"])
	}
}
