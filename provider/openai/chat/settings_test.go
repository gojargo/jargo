package chat

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// updateSettings sends a settings update the way the pipeline does: a typed
// delta, or plain names and values as an RTVI client sends them.
func updateSettings(t *testing.T, svc *LLMService, delta any) {
	t.Helper()
	f := frames.NewLLMUpdateSettingsFrame(delta)
	if m, ok := delta.(map[string]any); ok {
		f = frames.NewLLMUpdateSettingsFrame(nil)
		f.Settings = m
	}
	if err := svc.Base.ProcessFrame(context.Background(), f, processor.Downstream); err != nil {
		t.Fatalf("update: %v", err)
	}
}

// generateWith runs one plain generation through svc against srv.
func generateWith(t *testing.T, svc *LLMService) {
	t.Helper()
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	if err := svc.Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
}

// TestSettingsUpdateReachesTheNextRequest checks a settings update changes what
// the next request asks for: the model, the sampling controls, the token bounds
// and a field the service does not model, and that a cleared control is
// omitted again.
func TestSettingsUpdateReachesTheNextRequest(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("hi")))
	temp := 0.2
	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL, Model: "gpt-4o", Temperature: &temp})

	generateWith(t, svc)
	if srv.body["model"] != "gpt-4o" || srv.body["temperature"] != 0.2 {
		t.Fatalf("first request = %v, want the configured model and temperature", srv.body)
	}

	delta := &LLMSettings{}
	delta.Model = settings.Set("gpt-4.1")
	delta.Temperature = settings.Cleared[float64]()
	delta.TopP = settings.Set(0.5)
	delta.MaxTokens = settings.Set(64)
	delta.MaxCompletionTokens = settings.Set(128)
	delta.Seed = settings.Set(7)
	updateSettings(t, svc, delta)
	updateSettings(t, svc, map[string]any{"user": "abc"})

	srv.body = nil
	generateWith(t, svc)
	want := map[string]any{
		"model": "gpt-4.1", "top_p": 0.5, "max_tokens": float64(64),
		"max_completion_tokens": float64(128), "seed": float64(7), "user": "abc",
	}
	for k, v := range want {
		if srv.body[k] != v {
			t.Errorf("%s = %v, want %v", k, srv.body[k], v)
		}
	}
	if _, ok := srv.body["temperature"]; ok {
		t.Errorf("temperature = %v, want it omitted once cleared", srv.body["temperature"])
	}
	if got := svc.Model(); got != "gpt-4.1" {
		t.Errorf("Model() = %q, want the updated model labeling what follows", got)
	}
}

// TestSettingsStartFromTheConfig checks the settings an update is merged into
// start as the service was built, so an update to one field leaves the rest of
// the configuration in force.
func TestSettingsStartFromTheConfig(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("hi")))
	topP, seed := 0.9, 3
	svc := NewLLM(LLMConfig{
		APIKey: "k", BaseURL: srv.URL, MaxTokens: 50, TopP: &topP, Seed: &seed,
		Extra: map[string]any{"user": "u1"},
	})
	updateSettings(t, svc, map[string]any{"temperature": 0.7})

	generateWith(t, svc)
	want := map[string]any{
		"model": defaultLLMModel, "max_tokens": float64(50), "top_p": 0.9,
		"seed": float64(3), "user": "u1", "temperature": 0.7,
	}
	for k, v := range want {
		if srv.body[k] != v {
			t.Errorf("%s = %v, want %v", k, srv.body[k], v)
		}
	}
}
