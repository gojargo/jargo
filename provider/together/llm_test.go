package together_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/together"
)

// TestReasoningControlsReachTheRequest checks the toggle and the effort
// Together adds to the OpenAI body.
func TestReasoningControlsReachTheRequest(t *testing.T) {
	body := generate(t, together.LLMConfig{
		Reasoning:       map[string]any{"enabled": false},
		ReasoningEffort: "max",
	})
	if !reflect.DeepEqual(body["reasoning"], map[string]any{"enabled": false}) {
		t.Errorf("reasoning = %v, want {enabled: false}", body["reasoning"])
	}
	if body["reasoning_effort"] != "max" {
		t.Errorf("reasoning_effort = %v, want max", body["reasoning_effort"])
	}
}

// TestUnsetReasoningControlsAreOmitted checks neither field is sent when nobody
// set it, which is what leaves Together's per-model defaults in force.
func TestUnsetReasoningControlsAreOmitted(t *testing.T) {
	body := generate(t, together.LLMConfig{})
	for _, key := range []string{"reasoning", "reasoning_effort"} {
		if v, present := body[key]; present {
			t.Errorf("%s = %v, want it omitted", key, v)
		}
	}
}

// TestTheConfiguredExtraWins checks the caller's own Extra wins key by key, as
// it does over every other request parameter, and is not written to.
func TestTheConfiguredExtraWins(t *testing.T) {
	extra := map[string]any{"reasoning_effort": "low", "service_tier": "flex"}
	cfg := together.LLMConfig{
		Extra:           extra,
		ReasoningEffort: "high",
		Reasoning:       map[string]any{"enabled": true},
	}
	body := generate(t, cfg)

	if body["reasoning_effort"] != "low" || body["service_tier"] != "flex" {
		t.Errorf("body carried %v and %v, want the caller's extra", body["reasoning_effort"], body["service_tier"])
	}
	if !reflect.DeepEqual(body["reasoning"], map[string]any{"enabled": true}) {
		t.Errorf("reasoning = %v, want {enabled: true}", body["reasoning"])
	}
	if _, written := extra["reasoning"]; written {
		t.Error("the caller's Extra map was written to")
	}
}

// generate runs one generation against a fake endpoint and reports the request
// body it received.
func generate(t *testing.T, cfg together.LLMConfig) map[string]any {
	t.Helper()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	cfg.APIKey = "k"
	cfg.BaseURL = srv.URL
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")

	if err := together.NewLLM(cfg).Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return body
}
