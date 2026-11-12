package gemini

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// TestSettingsUpdateReachesTheNextRequest checks a settings update changes what
// the next request asks for: the model it is addressed to, the token bound, the
// sampling controls, thinking, the safety filters and a field the service does
// not model, and that a cleared control is omitted again.
func TestSettingsUpdateReachesTheNextRequest(t *testing.T) {
	temp := 0.3
	svc := NewLLM(Config{APIKey: "k", Model: "gemini-2.5-flash", Temperature: &temp})
	update := func(f *frames.LLMUpdateSettingsFrame) {
		t.Helper()
		if err := svc.Base.ProcessFrame(context.Background(), f, processor.Downstream); err != nil {
			t.Fatalf("update: %v", err)
		}
	}

	delta := &LLMSettings{}
	delta.Model = settings.Set("gemini-2.5-pro")
	delta.MaxTokens = settings.Set(300)
	delta.Temperature = settings.Cleared[float64]()
	delta.TopK = settings.Set(20)
	delta.Thinking = settings.Set(ThinkingConfig{Budget: new(512)})
	delta.SafetySettings = settings.Set([]SafetySetting{{
		Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_NONE",
	}})
	update(frames.NewLLMUpdateSettingsFrame(delta))
	plain := frames.NewLLMUpdateSettingsFrame(nil)
	plain.Settings = map[string]any{"candidateCount": 1}
	update(plain)

	cfg := svc.requestConfig()
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	body, err := svc.requestBodyFor(cfg, convo, adapter.Options{}, false)
	if err != nil {
		t.Fatalf("requestBody: %v", err)
	}
	req, err := svc.newRequestTo(t.Context(), cfg.Model, body, true)
	if err != nil {
		t.Fatalf("newRequest: %v", err)
	}
	if want := apiBase + "/gemini-2.5-pro:streamGenerateContent?alt=sse"; req.URL.String() != want {
		t.Errorf("url = %s, want the updated model's %s", req.URL, want)
	}
	g, _ := body["generationConfig"].(map[string]any)
	if g["maxOutputTokens"] != 300 || g["topK"] != 20 || g["candidateCount"] != 1 {
		t.Errorf("generationConfig = %v, want the updated bound, topK and extra field", g)
	}
	if _, ok := g["temperature"]; ok {
		t.Errorf("temperature = %v, want it omitted once cleared", g["temperature"])
	}
	if th, _ := g[keyThinkingConfig].(map[string]any); th["thinkingBudget"] != 512 {
		t.Errorf("thinkingConfig = %v, want the updated budget", g[keyThinkingConfig])
	}
	if ss, _ := body["safetySettings"].([]SafetySetting); len(ss) != 1 {
		t.Errorf("safetySettings = %v, want the updated filter", body["safetySettings"])
	}
	if got := svc.Model(); got != "gemini-2.5-pro" {
		t.Errorf("Model() = %q, want the updated model", got)
	}
}
