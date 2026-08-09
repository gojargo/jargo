package responses

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
)

// mustRequest builds a request and fails the test if the conversion did.
func mustRequest(t *testing.T, cfg Config, convo *frames.LLMContext) request {
	t.Helper()
	req, err := cfg.newRequest(convo, adapter.Options{}, false)
	if err != nil {
		t.Fatalf("newRequest: %v", err)
	}
	return req
}

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: Config{}, Valid: false},
		{Name: "API key only", Cfg: Config{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "OpenAIResponsesLLM", NewLLM(Config{APIKey: "k"}))
	providertest.Service(t, "OpenAIResponsesHTTPLLM", NewHTTPLLM(Config{APIKey: "k"}))
}

// TestBuildInputConfiguredInstructions checks the configured prompt is used when
// the context carries no system message, and yields to one that does.
func TestBuildInputConfiguredInstructions(t *testing.T) {
	cfg := Config{Instructions: "from config"}

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	if got := mustRequest(t, cfg, convo).Instructions; got != "from config" {
		t.Errorf("instructions = %q, want the configured prompt", got)
	}

	withSystem := frames.NewLLMContext("from context")
	withSystem.AddUserMessage("hi")
	if got := mustRequest(t, cfg, withSystem).Instructions; got != "from context" {
		t.Errorf("instructions = %q, want the conversation's own prompt to win", got)
	}
}

// TestEncodeBodyExtraOverrides checks configured extra fields reach the request
// and win over the modeled ones.
func TestEncodeBodyExtraOverrides(t *testing.T) {
	req := request{Model: "gpt-4.1", Stream: true}
	raw, err := encodeBody(req, map[string]any{"model": "gpt-5", "custom": 7})
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["model"] != "gpt-5" {
		t.Errorf("model = %v, want the extra field to win", m["model"])
	}
	if m["custom"] != float64(7) {
		t.Errorf("custom = %v, want the extra field carried through", m["custom"])
	}
}

// TestNewRequestOmitsUnset checks the optional controls are absent from the wire
// when unset, leaving the API defaults in place.
func TestNewRequestOmitsUnset(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")

	raw, err := encodeBody(mustRequest(t, Config{Model: "gpt-4.1"}, convo), nil)
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, key := range []string{
		"temperature", "top_p", "max_output_tokens", "service_tier", "tools",
		"instructions", "previous_response_id",
	} {
		if _, ok := m[key]; ok {
			t.Errorf("%s is present when unset, want it omitted", key)
		}
	}
	if m["store"] != false {
		t.Errorf("store = %v, want false so conversations are not retained", m["store"])
	}
	if m["stream"] != true {
		t.Errorf("stream = %v, want true", m["stream"])
	}
}
