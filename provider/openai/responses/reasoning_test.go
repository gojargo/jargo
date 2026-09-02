package responses

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// convo is a one-turn conversation, which is all a request needs to be built.
func convo() *frames.LLMContext {
	c := frames.NewLLMContext("")
	c.AddUserMessage("bonjour")
	return c
}

// TestModelSupportsReasoning pins the classification the default rests on.
// Unknown is its own answer: a name matching neither shape is not assumed
// either way.
func TestModelSupportsReasoning(t *testing.T) {
	for _, tc := range []struct {
		model          string
		reasons, known bool
	}{
		{"gpt-5.6-terra", true, true},
		{"gpt-5", true, true},
		{"gpt-6", true, true},
		{"o3", true, true},
		{"o4-mini", true, true},
		{"gpt-5-chat-latest", false, true},
		{"gpt-4.1", false, true},
		{"gpt-4o", false, true},
		{"llama-3", false, false},
		{"", false, false},
	} {
		reasons, known := modelSupportsReasoning(tc.model)
		if reasons != tc.reasons || known != tc.known {
			t.Errorf("modelSupportsReasoning(%q) = (%v, %v), want (%v, %v)",
				tc.model, reasons, known, tc.reasons, tc.known)
		}
	}
}

// TestReasoningIsOffByDefault is the whole point of the default. A reasoning
// model left unconfigured reasons at the API's own default, which on a voice
// call is thinking time somebody spends listening to silence.
func TestReasoningIsOffByDefault(t *testing.T) {
	req := mustRequest(t, Config{Model: "gpt-5.6-terra"}, convo())
	if req.Reasoning == nil || req.Reasoning.Effort != "none" {
		t.Errorf("reasoning = %+v, want effort none", req.Reasoning)
	}
}

// A model that does not reason has no field to set, and the o-series is
// reasoning-first and refuses being told not to. Both leave the request
// without one.
func TestReasoningIsOmittedWhereItCannotApply(t *testing.T) {
	for _, model := range []string{"gpt-4.1", "gpt-5-chat-latest", "o3", "some-other-model"} {
		if got := mustRequest(t, Config{Model: model}, convo()).Reasoning; got != nil {
			t.Errorf("%s carries reasoning %+v, want none", model, got)
		}
	}
}

// A configured effort is sent as it stands, on any model.
func TestConfiguredReasoningWins(t *testing.T) {
	cfg := Config{Model: "gpt-5.6-terra", Reasoning: &ReasoningConfig{Effort: "high", Summary: "concise"}}
	got := mustRequest(t, cfg, convo()).Reasoning
	if got == nil || got.Effort != "high" || got.Summary != "concise" {
		t.Fatalf("reasoning = %+v, want the configured one", got)
	}
	// And the caller's own value is not the one the request holds, so a request
	// cannot edit the service's configuration.
	if got == cfg.Reasoning {
		t.Error("the request shares the configured value rather than copying it")
	}
}

// An empty config is the same as none: it must not send an empty object, which
// asks the API to reason at its default rather than not at all.
func TestEmptyReasoningConfigStillDisablesIt(t *testing.T) {
	req := mustRequest(t, Config{Model: "gpt-5.6-terra", Reasoning: &ReasoningConfig{}}, convo())
	if req.Reasoning == nil || req.Reasoning.Effort != "none" {
		t.Errorf("reasoning = %+v, want effort none", req.Reasoning)
	}
}

// What actually reaches the API. An unset field must be absent rather than
// empty: "summary": "" is not a summary the API knows.
func TestReasoningEncoding(t *testing.T) {
	raw, err := json.Marshal(mustRequest(t, Config{Model: "gpt-5.6-terra"}, convo()))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("no reasoning object in %s", raw)
	}
	if reasoning["effort"] != "none" {
		t.Errorf("effort = %v, want none", reasoning["effort"])
	}
	if _, present := reasoning["summary"]; present {
		t.Errorf("an unset summary reached the request: %s", raw)
	}

	plain, err := json.Marshal(mustRequest(t, Config{Model: "gpt-4.1"}, convo()))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var plainBody map[string]any
	if err := json.Unmarshal(plain, &plainBody); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, present := plainBody["reasoning"]; present {
		t.Errorf("a non-reasoning model carries a reasoning field: %s", plain)
	}
}
