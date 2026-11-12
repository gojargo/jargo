package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// TestSettingsUpdateReachesTheNextRequest checks a settings update changes what
// the next request asks for: the model, the token bound, the sampling
// controls, thinking, the effort and a field the service does not model, and
// that a cleared control is omitted again.
func TestSettingsUpdateReachesTheNextRequest(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body = nil
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	}))
	defer srv.Close()

	temp := 0.3
	svc := NewLLM(Config{APIKey: "k", BaseURL: srv.URL, Model: "claude-sonnet-4-5", Temperature: &temp})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	generate := func() {
		t.Helper()
		if err := svc.Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
			t.Fatalf("Generate: %v", err)
		}
	}
	update := func(f *frames.LLMUpdateSettingsFrame) {
		t.Helper()
		if err := svc.Base.ProcessFrame(context.Background(), f, processor.Downstream); err != nil {
			t.Fatalf("update: %v", err)
		}
	}

	generate()
	if body["model"] != "claude-sonnet-4-5" || body["temperature"] != 0.3 {
		t.Fatalf("first request = %v, want the configured model and temperature", body)
	}

	delta := &LLMSettings{}
	delta.Model = settings.Set("claude-opus-4-1")
	delta.MaxTokens = settings.Set(300)
	delta.Temperature = settings.Cleared[float64]()
	delta.TopK = settings.Set(20)
	delta.Thinking = settings.Set(ThinkingConfig{Type: "enabled", BudgetTokens: 2048})
	delta.Effort = settings.Set("low")
	update(frames.NewLLMUpdateSettingsFrame(delta))
	plain := frames.NewLLMUpdateSettingsFrame(nil)
	plain.Settings = map[string]any{"metadata": map[string]any{"user_id": "u1"}}
	update(plain)

	generate()
	if body["model"] != "claude-opus-4-1" || body["max_tokens"] != float64(300) || body["top_k"] != float64(20) {
		t.Errorf("request = %v, want the updated model, max_tokens and top_k", body)
	}
	if _, ok := body["temperature"]; ok {
		t.Errorf("temperature = %v, want it omitted once cleared", body["temperature"])
	}
	if th, _ := body["thinking"].(map[string]any); th["type"] != "enabled" || th["budget_tokens"] != float64(2048) {
		t.Errorf("thinking = %v, want the updated configuration", body["thinking"])
	}
	if oc, _ := body["output_config"].(map[string]any); oc["effort"] != "low" {
		t.Errorf("output_config = %v, want the updated effort", body["output_config"])
	}
	if md, _ := body["metadata"].(map[string]any); md["user_id"] != "u1" {
		t.Errorf("metadata = %v, want the unmodeled field set on the request", body["metadata"])
	}
	if got := svc.Model(); got != "claude-opus-4-1" {
		t.Errorf("Model() = %q, want the updated model", got)
	}
}

// TestSettingsStartFromTheConfig checks the settings start as the service was
// built, defaults filled in.
func TestSettingsStartFromTheConfig(t *testing.T) {
	s := newLLMSettings(Config{})
	if s.Model.Or("") == "" || s.MaxTokens.Or(0) != defaultMaxTokens || !s.EnablePromptCaching.Or(false) {
		t.Errorf("settings = %+v, want the default model, token bound and prompt caching", s)
	}
	off, topK := false, int64(5)
	s = newLLMSettings(Config{
		Model: "m", MaxTokens: 9, TopK: &topK, EnablePromptCaching: &off,
		SystemPromptCacheTTL: "1h", Effort: "high", Extra: map[string]any{"x": 1},
	})
	if s.Model.Or("") != "m" || s.MaxTokens.Or(0) != 9 || s.TopK.Or(0) != 5 || s.EnablePromptCaching.Or(true) ||
		s.SystemPromptCacheTTL.Or("") != "1h" || s.Effort.Or("") != "high" || s.Extra["x"] != 1 {
		t.Errorf("settings = %+v, want them seeded from the config", s)
	}
}
