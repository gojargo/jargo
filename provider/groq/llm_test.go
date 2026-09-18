package groq_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/groq"
)

// TestConfigValidateLLM pins the reasoning levels Groq documents. A level it
// does not serve is a request that fails, so it is refused when the service is
// built rather than on the first turn.
func TestConfigValidateLLM(t *testing.T) {
	for _, tc := range []struct {
		effort string
		valid  bool
	}{
		{"", true},
		{"none", true},
		{"default", true},
		{"low", true},
		{"medium", true},
		{"high", true},
		{"minimal", false},
	} {
		err := groq.LLMConfig{
			APIKey:          "k",
			ReasoningEffort: tc.effort,
		}.Validate()
		if (err == nil) != tc.valid {
			t.Errorf("Validate() with effort %q = %v, want valid %v", tc.effort, err, tc.valid)
		}
	}
}

// TestReasoningEffortReachesTheRequest checks the control Groq adds to the
// OpenAI body. "none" is the one that matters on a voice call: Groq's reasoning
// format is raw by default, so a Qwen model left to think streams its chain of
// thought inline and the whole of it would be spoken.
func TestReasoningEffortReachesTheRequest(t *testing.T) {
	body := generate(t, groq.LLMConfig{ReasoningEffort: "none"})
	if body["reasoning_effort"] != "none" {
		t.Errorf("reasoning_effort = %v, want none", body["reasoning_effort"])
	}
}

// TestUnsetReasoningEffortIsOmitted checks the field is absent rather than empty
// when nobody chose a level, which is what leaves Groq's per-model default in
// force.
func TestUnsetReasoningEffortIsOmitted(t *testing.T) {
	body := generate(t, groq.LLMConfig{})
	if v, present := body["reasoning_effort"]; present {
		t.Errorf("reasoning_effort = %v, want it omitted", v)
	}
}

// TestReasoningEffortLeavesTheConfiguredExtraAlone checks the control is merged
// into a copy: a caller's own Extra map is not written to behind its back.
func TestReasoningEffortLeavesTheConfiguredExtraAlone(t *testing.T) {
	extra := map[string]any{"service_tier": "flex"}
	cfg := groq.LLMConfig{
		Extra:           extra,
		ReasoningEffort: "low",
	}
	body := generate(t, cfg)

	if body["reasoning_effort"] != "low" || body["service_tier"] != "flex" {
		t.Errorf("body carried %v and %v, want both the control and the caller's extra",
			body["reasoning_effort"], body["service_tier"])
	}
	if _, written := extra["reasoning_effort"]; written {
		t.Error("the caller's Extra map was written to")
	}
}

// generate runs one generation against a fake endpoint and reports the request
// body it received.
func generate(t *testing.T, cfg groq.LLMConfig) map[string]any {
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

	if err := groq.NewLLM(cfg).Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return body
}
