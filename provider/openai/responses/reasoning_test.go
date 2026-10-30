package responses

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/gojargo/jargo/adapter"
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

// reasoningSent is the reasoning a request for model carries.
func reasoningSent(t *testing.T, model string) *ReasoningConfig {
	t.Helper()
	return mustRequest(t, Config{Model: model}, convo()).Reasoning
}

// TestOriginalGPT5GetsMinimal covers the original gpt-5 models, which reason
// before most replies and reject "none".
func TestOriginalGPT5GetsMinimal(t *testing.T) {
	for _, model := range []string{"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-2025-08-07"} {
		if got := reasoningSent(t, model); got == nil || *got != (ReasoningConfig{Effort: "minimal"}) {
			t.Errorf("%s carries reasoning %+v, want effort minimal", model, got)
		}
	}
}

// TestLaterModelsGetNone covers the models from gpt-5.1 on, most of which can
// skip reasoning entirely.
func TestLaterModelsGetNone(t *testing.T) {
	for _, model := range []string{"gpt-5.1", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6-sol", "gpt-6-luna", "gpt-6.1-terra"} {
		if got := reasoningSent(t, model); got == nil || *got != (ReasoningConfig{Effort: "none"}) {
			t.Errorf("%s carries reasoning %+v, want effort none", model, got)
		}
	}
}

// TestModelsThatMustReasonGetTheirLowestEffort covers the models that reject
// "none": they get the lowest effort they accept.
func TestModelsThatMustReasonGetTheirLowestEffort(t *testing.T) {
	for model, effort := range map[string]string{
		"gpt-6-astra": "low",
		"gpt-6.1-sol": "low",
		"o3":          "low",
		"gpt-5.5-pro": "medium",
		"gpt-5-pro":   "high",
	} {
		if got := reasoningSent(t, model); got == nil || *got != (ReasoningConfig{Effort: effort}) {
			t.Errorf("%s carries reasoning %+v, want effort %s", model, got, effort)
		}
	}
}

// TestUnlistedModelsRunAtOpenAIsDefault covers every other model: only listed
// models get an effort, so a model is never sent one it rejects.
func TestUnlistedModelsRunAtOpenAIsDefault(t *testing.T) {
	for _, model := range []string{"gpt-6.2", "gpt-5-codex", "gpt-4.1", "gpt-5-chat-latest"} {
		if got := reasoningSent(t, model); got != nil {
			t.Errorf("%s carries reasoning %+v, want none", model, got)
		}
	}
}

// TestReasoningInExtraReplacesTheDefault covers reasoning passed through Extra,
// which wins over the lowest-effort default.
func TestReasoningInExtraReplacesTheDefault(t *testing.T) {
	cfg := Config{Model: "gpt-5-mini", Extra: map[string]any{"reasoning": map[string]any{"effort": "low"}}}
	req := mustRequest(t, cfg, convo())
	body, err := encodeBody(req, cfg.Extra)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatal(err)
	}
	if got, _ := m["reasoning"].(map[string]any); got["effort"] != "low" || len(got) != 1 {
		t.Errorf("reasoning = %v, want the one set through Extra", m["reasoning"])
	}
}

// recorder records the thinking defaults reported to it.
type recorder struct {
	lines           []string
	providerDefault bool
}

func (r *recorder) UseProviderThinkingDefault() bool { return r.providerDefault }

func (r *recorder) ReportThinkingDefault(model, setting string, canThink bool, _, _ string) {
	switch {
	case r.providerDefault:
		if canThink {
			r.lines = append(r.lines, "DEBUG "+model)
		}
	case setting != "":
		r.lines = append(r.lines, "INFO "+model+" "+setting)
	case canThink:
		r.lines = append(r.lines, "WARN "+model)
	default:
		r.lines = append(r.lines, "NONE "+model)
	}
}

// TestTheDefaultIsReportedForEachRequest covers what the base is told about
// each request: the effort sent, or a warning for a reasoning model with no
// default. The base reports each model once; see ReportThinkingDefault.
func TestTheDefaultIsReportedForEachRequest(t *testing.T) {
	var r recorder
	for _, model := range []string{"gpt-5-mini", "gpt-6.2", "gpt-4.1"} {
		var req request
		Config{Model: model}.applyThinkingDefault(&req, &r)
	}
	want := []string{`INFO gpt-5-mini reasoning={"effort":"minimal"}`, "WARN gpt-6.2", "NONE gpt-4.1"}
	if !slices.Equal(r.lines, want) {
		t.Errorf("reported %q, want %q", r.lines, want)
	}
}

// TestNothingIsReportedWhenReasoningIsConfigured covers an explicit setting,
// which is the developer's choice and needs no comment.
func TestNothingIsReportedWhenReasoningIsConfigured(t *testing.T) {
	var r recorder
	cfg := Config{Model: "gpt-6.2", Reasoning: &ReasoningConfig{Effort: "low"}}
	req, err := cfg.newRequest(convo(), adapter.Options{}, false)
	if err != nil {
		t.Fatal(err)
	}
	cfg.applyThinkingDefault(&req, &r)
	if len(r.lines) != 0 {
		t.Errorf("reported %q, want nothing", r.lines)
	}
}

// TestGPT61SolExplicitEffortHonored checks an explicit effort reaches the
// request unchanged.
func TestGPT61SolExplicitEffortHonored(t *testing.T) {
	cfg := Config{Model: "gpt-6.1-sol", Reasoning: &ReasoningConfig{Effort: "medium"}}
	if got := mustRequest(t, cfg, convo()).Reasoning; got == nil || got.Effort != "medium" {
		t.Errorf("reasoning = %+v, want the configured effort medium", got)
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

// TestReasoningModeIsSentAlongsideTheEffort covers the mode a model such as the
// gpt-5.6 series offers beside the effort level: the two travel together, the
// effort choosing how hard the model thinks within the mode.
func TestReasoningModeIsSentAlongsideTheEffort(t *testing.T) {
	cfg := Config{Model: "gpt-5.6-sol", Reasoning: &ReasoningConfig{Effort: "low", Mode: "pro"}}
	got := mustRequest(t, cfg, convo()).Reasoning
	if got == nil || got.Effort != "low" || got.Mode != "pro" {
		t.Fatalf("reasoning = %+v, want effort low in mode pro", got)
	}
}

// TestUnsetReasoningModeIsOmitted checks the mode is absent rather than empty
// when nobody chose one: "mode": "" is not a mode the API knows.
func TestUnsetReasoningModeIsOmitted(t *testing.T) {
	raw, err := json.Marshal(mustRequest(t, Config{Model: "gpt-5.6-terra"}, convo()))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	reasoning, _ := body["reasoning"].(map[string]any)
	if _, present := reasoning["mode"]; present {
		t.Errorf("an unset mode reached the request: %s", raw)
	}
}

// An empty config is the same as none: it must not send an empty object, which
// asks the API to reason at its default rather than at the lowest effort.
func TestEmptyReasoningConfigFallsBackToTheDefault(t *testing.T) {
	req := mustRequest(t, Config{Model: "gpt-5-mini", Reasoning: &ReasoningConfig{}}, convo())
	if req.Reasoning == nil || req.Reasoning.Effort != "minimal" {
		t.Errorf("reasoning = %+v, want effort minimal", req.Reasoning)
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

// TestUseProviderThinkingDefaultRunsAtOpenAIsDefault checks the setting sends no
// default, and that the base is told the model runs at OpenAI's default.
func TestUseProviderThinkingDefaultRunsAtOpenAIsDefault(t *testing.T) {
	r := recorder{providerDefault: true}
	var req request
	Config{Model: "gpt-5.5"}.applyThinkingDefault(&req, &r)

	if req.Reasoning != nil {
		t.Errorf("sent reasoning %+v, want none", req.Reasoning)
	}
	if want := []string{"DEBUG gpt-5.5"}; !slices.Equal(r.lines, want) {
		t.Errorf("reported %q, want %q", r.lines, want)
	}
}

// TestUseProviderThinkingDefaultOnAModelThatDoesNotReasonLogsNothing checks a
// model without reasoning has no default to skip.
func TestUseProviderThinkingDefaultOnAModelThatDoesNotReasonLogsNothing(t *testing.T) {
	r := recorder{providerDefault: true}
	var req request
	Config{Model: "gpt-4.1"}.applyThinkingDefault(&req, &r)

	if len(r.lines) != 0 {
		t.Errorf("reported %q, want nothing", r.lines)
	}
}
