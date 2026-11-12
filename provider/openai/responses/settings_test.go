package responses

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

// sendSettings sends a settings update to a service the way the pipeline does.
func sendSettings(t *testing.T, p processor.Processor, delta any) {
	t.Helper()
	f := frames.NewLLMUpdateSettingsFrame(delta)
	if m, ok := delta.(map[string]any); ok {
		f = frames.NewLLMUpdateSettingsFrame(nil)
		f.Settings = m
	}
	if err := p.ProcessFrame(context.Background(), f, processor.Downstream); err != nil {
		t.Fatalf("update: %v", err)
	}
}

// liveUpdate is the update both transports are checked against: a new model,
// a cleared temperature, an output bound, reasoning, and a field the service
// does not model.
func liveUpdate(t *testing.T, p processor.Processor) {
	t.Helper()
	delta := &LLMSettings{}
	delta.Model = settings.Set("o4-mini")
	delta.Temperature = settings.Cleared[float64]()
	delta.MaxCompletionTokens = settings.Set(256)
	delta.Reasoning = settings.Set(ReasoningConfig{Effort: "high"})
	sendSettings(t, p, delta)
	sendSettings(t, p, map[string]any{"user": "abc"})
}

// checkLiveUpdate checks a request carries what liveUpdate asked for.
func checkLiveUpdate(t *testing.T, body map[string]any) {
	t.Helper()
	if body["model"] != "o4-mini" {
		t.Errorf("model = %v, want the updated o4-mini", body["model"])
	}
	if _, ok := body["temperature"]; ok {
		t.Errorf("temperature = %v, want it omitted once cleared", body["temperature"])
	}
	if body["max_output_tokens"] != float64(256) {
		t.Errorf("max_output_tokens = %v, want 256", body["max_output_tokens"])
	}
	if r, _ := body["reasoning"].(map[string]any); r["effort"] != "high" {
		t.Errorf("reasoning = %v, want the updated effort", body["reasoning"])
	}
	if body["user"] != "abc" {
		t.Errorf("user = %v, want the unmodeled field merged in", body["user"])
	}
}

// TestHTTPSettingsUpdateReachesTheNextRequest checks a settings update changes
// what the next HTTP request asks for.
func TestHTTPSettingsUpdateReachesTheNextRequest(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body = nil
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: " + `{"type":"response.output_text.delta","delta":"hi"}` + "\n\n"))
		_, _ = w.Write([]byte("data: " + `{"type":"response.completed","response":{"id":"r"}}` + "\n\n"))
	}))
	defer srv.Close()

	temp := 0.3
	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL, Model: "gpt-4.1", Temperature: &temp})
	svc.http = srv.Client()
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	discard := func(string) error { return nil }

	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if body["model"] != "gpt-4.1" || body["temperature"] != 0.3 {
		t.Fatalf("first request = %v, want the configured model and temperature", body)
	}

	liveUpdate(t, svc)
	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	checkLiveUpdate(t, body)
	if got := svc.Model(); got != "o4-mini" {
		t.Errorf("Model() = %q, want the updated model", got)
	}
}

// TestWSSettingsUpdateReachesTheNextRequest checks the same over the
// WebSocket.
func TestWSSettingsUpdateReachesTheNextRequest(t *testing.T) {
	turn := []string{
		`{"type":"response.output_text.delta","delta":"hi"}`,
		`{"type":"response.completed","response":{"id":"resp_1"}}`,
	}
	srv := newFakeResponsesWS(t, [][]string{turn, turn})
	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL(), Model: "gpt-4.1"})
	t.Cleanup(svc.disconnect)
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	discard := func(string) error { return nil }

	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	liveUpdate(t, svc)
	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	checkLiveUpdate(t, srv.turn(1))
}

// TestSettingsStartFromTheConfig checks the settings start as the service was
// built, so an update to one field leaves the rest of the configuration in
// force.
func TestSettingsStartFromTheConfig(t *testing.T) {
	topP := 0.9
	s := newLLMSettings(Config{
		Model: "gpt-4.1", TopP: &topP, MaxOutputTokens: 50,
		Reasoning: &ReasoningConfig{Effort: "low"}, Extra: map[string]any{"user": "u1"},
	})
	if s.Model.Or("") != "gpt-4.1" || s.TopP.Or(0) != 0.9 || s.MaxCompletionTokens.Or(0) != 50 ||
		s.Reasoning.Or(ReasoningConfig{}).Effort != "low" || s.Extra["user"] != "u1" {
		t.Errorf("settings = %+v, want them seeded from the config", s)
	}
	if s.Temperature.IsGiven() {
		t.Error("temperature was given for a config that set none")
	}
}
