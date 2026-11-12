package anthropic

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// appliedDefault returns the request fields a service built for model sends
// when its config sets nothing about thinking, as JSON-decoded values.
func appliedDefault(t *testing.T, model string) map[string]any {
	t.Helper()
	return paramsOf(t, NewLLM(Config{Model: model}))
}

// paramsOf returns the request the service builds for a one-message
// conversation, decoded from its JSON.
func paramsOf(t *testing.T, s *Service) map[string]any {
	t.Helper()
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	b, err := json.Marshal(s.mustParams(t, convo))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// thinkingFields keeps the two request fields the thinking default sets.
func thinkingFields(params map[string]any) map[string]any {
	out := map[string]any{}
	for _, k := range []string{"thinking", "output_config"} {
		if v, ok := params[k]; ok {
			out[k] = v
		}
	}
	return out
}

// captureLogs routes the default logger to a buffer at INFO and above until
// the test ends.
func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&syncWriter{w: &buf}, &slog.HandlerOptions{Level: slog.LevelInfo})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &buf
}

// syncWriter serializes writes to the buffer it wraps.
type syncWriter struct {
	mu sync.Mutex
	w  *bytes.Buffer
}

func (s *syncWriter) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w.Write(p)
}

// logLines applies the default twice for each model in turn, on one service,
// and returns what it logged.
func logLines(t *testing.T, models ...string) []string {
	t.Helper()
	buf := captureLogs(t)
	s := NewLLM(Config{Model: models[0]})
	for _, model := range models {
		s.live.Model = settings.Set(model)
		paramsOf(t, s)
		paramsOf(t, s)
	}
	out := strings.TrimSpace(buf.String())
	if out == "" {
		return nil
	}
	return strings.Split(out, "\n")
}

// requestBody runs one inference through a stub server and returns the body
// the service sent.
func requestBody(t *testing.T, cfg Config, opts llm.InferenceOptions) map[string]any {
	t.Helper()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"claude",` +
			`"content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)
	cfg.APIKey, cfg.BaseURL = "k", srv.URL
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	if _, err := NewLLM(cfg).RunInference(t.Context(), convo, opts); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	return body
}

// streamedRequestBody runs one streamed generation through a stub server and
// returns the body the service sent.
func streamedRequestBody(t *testing.T, cfg Config) map[string]any {
	t.Helper()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	}))
	t.Cleanup(srv.Close)
	cfg.APIKey, cfg.BaseURL = "k", srv.URL
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	if err := NewLLM(cfg).Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return body
}

// obj builds a JSON object from alternating keys and values.
func obj(kv ...any) map[string]any {
	m := map[string]any{}
	for i := 0; i+1 < len(kv); i += 2 {
		if k, ok := kv[i].(string); ok {
			m[k] = kv[i+1]
		}
	}
	return m
}

// --- default thinking config per model ---

// TestModelsThatAllowItGetThinkingDisabled checks that turning thinking off is
// the lowest setting wherever a model allows it.
func TestModelsThatAllowItGetThinkingDisabled(t *testing.T) {
	for _, model := range []string{
		"claude-opus-5", "claude-sonnet-5",
		"claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5",
	} {
		want := obj("thinking", obj("type", "disabled"))
		if got := thinkingFields(appliedDefault(t, model)); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: %v, want %v", model, got, want)
		}
	}
}

// TestHaiku55GetsLowEffort checks that Haiku 5.5, which speaks its reasoning
// with thinking disabled, gets low effort.
func TestHaiku55GetsLowEffort(t *testing.T) {
	want := obj("output_config", obj("effort", "low"))
	if got := thinkingFields(appliedDefault(t, "claude-haiku-5-5")); !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// TestSonnet55GetsBetweenTools checks that Sonnet 5.5, which rejects
// "disabled", gets "between_tools", its lowest setting.
func TestSonnet55GetsBetweenTools(t *testing.T) {
	want := obj("thinking", obj("type", "between_tools"))
	if got := thinkingFields(appliedDefault(t, "claude-sonnet-5-5")); !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// TestModelsThatAlwaysThinkGetLowEffort checks that models whose thinking
// cannot be turned off get the lowest effort instead.
func TestModelsThatAlwaysThinkGetLowEffort(t *testing.T) {
	for _, model := range []string{"claude-opus-5-5", "claude-fable-5-1", "claude-fable-5"} {
		want := obj("output_config", obj("effort", "low"))
		if got := thinkingFields(appliedDefault(t, model)); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: %v, want %v", model, got, want)
		}
	}
}

// TestLowEffortJoinsAResponseSchemaInOutputConfig checks that the effort
// default does not displace a response schema in output_config.
func TestLowEffortJoinsAResponseSchemaInOutputConfig(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	body := requestBody(t, Config{Model: "claude-opus-5-5"}, llm.InferenceOptions{ResponseSchema: schema})
	want := obj("effort", "low", "format", obj("type", "json_schema", "schema", obj("type", "object")))
	if got := body["output_config"]; !reflect.DeepEqual(got, want) {
		t.Errorf("output_config = %v, want %v", got, want)
	}
}

// TestEveryIDFormOfAModelIsRecognized checks that Bedrock prefixes, Vertex
// versions and dated snapshots name the same model.
func TestEveryIDFormOfAModelIsRecognized(t *testing.T) {
	for _, model := range []string{
		"anthropic.claude-sonnet-5",
		"us.anthropic.claude-sonnet-5",
		"claude-sonnet-5@20260630",
		"claude-sonnet-5-20260630",
		"claude-haiku-4-5-20251001",
		"anthropic.claude-haiku-4-5-20251001-v1:0",
	} {
		if got := appliedDefault(t, model)["thinking"]; !reflect.DeepEqual(got, obj("type", "disabled")) {
			t.Errorf("%s: thinking = %v, want disabled", model, got)
		}
	}
}

// TestUnlistedModelsRunAtAnthropicsDefault checks that only listed models get
// a thinking default, so a new model is never guessed at.
func TestUnlistedModelsRunAtAnthropicsDefault(t *testing.T) {
	for _, model := range []string{"claude-sonnet-6", "claude-mythos-5", "claude-3-5-sonnet-20241022"} {
		if got := thinkingFields(appliedDefault(t, model)); len(got) != 0 {
			t.Errorf("%s: %v, want nothing", model, got)
		}
	}
}

// --- explicit configuration wins ---

// TestAConfiguredThinkingConfigIsLeftAlone checks that an explicit thinking
// config wins over the low-latency default.
func TestAConfiguredThinkingConfigIsLeftAlone(t *testing.T) {
	body := requestBody(t, Config{
		Model:    "claude-sonnet-5",
		Thinking: &ThinkingConfig{Type: "adaptive", Display: "summarized"},
	}, llm.InferenceOptions{})
	if want := obj("type", "adaptive", "display", "summarized"); !reflect.DeepEqual(body["thinking"], want) {
		t.Errorf("thinking = %v, want %v", body["thinking"], want)
	}
}

// TestThinkingPassedThroughExtraIsLeftAlone checks that a thinking config in
// Extra wins too.
func TestThinkingPassedThroughExtraIsLeftAlone(t *testing.T) {
	body := requestBody(t, Config{
		Model: "claude-sonnet-5",
		Extra: map[string]any{"thinking": map[string]any{"type": "adaptive"}},
	}, llm.InferenceOptions{})
	if want := obj("type", "adaptive"); !reflect.DeepEqual(body["thinking"], want) {
		t.Errorf("thinking = %v, want %v", body["thinking"], want)
	}
}

// TestAConfiguredEffortReplacesTheThinkingDefault checks that effort and
// thinking together decide how much the model thinks, so either one wins.
func TestAConfiguredEffortReplacesTheThinkingDefault(t *testing.T) {
	body := requestBody(t, Config{Model: "claude-sonnet-5", Effort: "low"}, llm.InferenceOptions{})
	if _, ok := body["thinking"]; ok {
		t.Errorf("thinking = %v, want none", body["thinking"])
	}
	if want := obj("effort", "low"); !reflect.DeepEqual(body["output_config"], want) {
		t.Errorf("output_config = %v, want %v", body["output_config"], want)
	}
}

// TestEffortPassedThroughExtraReplacesTheThinkingDefault checks that an effort
// in Extra wins too.
func TestEffortPassedThroughExtraReplacesTheThinkingDefault(t *testing.T) {
	body := requestBody(t, Config{
		Model: "claude-sonnet-5",
		Extra: map[string]any{"output_config": map[string]any{"effort": "low"}},
	}, llm.InferenceOptions{})
	if _, ok := body["thinking"]; ok {
		t.Errorf("thinking = %v, want none", body["thinking"])
	}
	if want := obj("effort", "low"); !reflect.DeepEqual(body["output_config"], want) {
		t.Errorf("output_config = %v, want %v", body["output_config"], want)
	}
}

// --- effort setting ---

// TestEffortIsOmittedWhenUnset checks that without an effort setting, only a
// default can add one.
func TestEffortIsOmittedWhenUnset(t *testing.T) {
	body := requestBody(t, Config{Model: "claude-sonnet-5"}, llm.InferenceOptions{})
	if _, ok := body["output_config"]; ok {
		t.Errorf("output_config = %v, want none", body["output_config"])
	}
}

// TestEffortAndResponseSchemaShareOutputConfig checks that a response schema
// does not displace the effort setting, nor the reverse.
func TestEffortAndResponseSchemaShareOutputConfig(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"ok":{"type":"boolean"}}}`)
	body := requestBody(t, Config{Model: "claude-opus-5-5", Effort: "medium"},
		llm.InferenceOptions{ResponseSchema: schema})
	var schemaObj map[string]any
	if err := json.Unmarshal(schema, &schemaObj); err != nil {
		t.Fatal(err)
	}
	want := obj("effort", "medium", "format", obj("type", "json_schema", "schema", schemaObj))
	if !reflect.DeepEqual(body["output_config"], want) {
		t.Errorf("output_config = %v, want %v", body["output_config"], want)
	}
}

// --- logging ---

// TestTheAppliedDefaultIsLoggedOncePerModel checks that the developer learns
// what was sent, and when to change it, once per model.
func TestTheAppliedDefaultIsLoggedOncePerModel(t *testing.T) {
	lines := logLines(t, "claude-sonnet-5", "claude-sonnet-5-20260630", "claude-opus-5-5")

	if len(lines) != 3 {
		t.Fatalf("logged %d lines, want 3:\n%s", len(lines), strings.Join(lines, "\n"))
	}
	for _, line := range lines {
		if !strings.Contains(line, "level=INFO") {
			t.Errorf("not at INFO: %s", line)
		}
	}
	for i, want := range []string{
		`model=claude-sonnet-5 setting="thinking={\"type\":\"disabled\"}"`,
		"model=claude-sonnet-5-20260630 ",
		`model=claude-opus-5-5 setting="output_config={\"effort\":\"low\"}"`,
	} {
		if !strings.Contains(lines[i], want) {
			t.Errorf("line %d = %s, want it to contain %s", i, lines[i], want)
		}
	}
	if !strings.Contains(lines[0], "multistep reasoning") ||
		!strings.Contains(lines[0], `configure="Thinking or Effort in anthropic.Config"`) {
		t.Errorf("line 0 does not say what to set: %s", lines[0])
	}
}

// TestAnUnlistedClaudeModelIsWarnedAboutOnce checks that a Claude model with no
// default, which may think before replying, is reported to the developer.
func TestAnUnlistedClaudeModelIsWarnedAboutOnce(t *testing.T) {
	lines := logLines(t, "claude-sonnet-6")

	if len(lines) != 1 {
		t.Fatalf("logged %d lines, want 1:\n%s", len(lines), strings.Join(lines, "\n"))
	}
	if !strings.Contains(lines[0], "level=WARN") ||
		!strings.Contains(lines[0], "no thinking default") ||
		!strings.Contains(lines[0], "model=claude-sonnet-6") {
		t.Errorf("line = %s, want a warning about claude-sonnet-6", lines[0])
	}
}

// TestNothingIsLoggedForAModelIDThatNamesNoClaudeModel checks that a Bedrock
// inference profile ARN, which does not say which model it runs, is not
// warned about.
func TestNothingIsLoggedForAModelIDThatNamesNoClaudeModel(t *testing.T) {
	profile := "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
	if lines := logLines(t, profile); len(lines) != 0 {
		t.Errorf("logged %v, want nothing", lines)
	}
}

// TestNothingIsLoggedWhenReasoningIsConfigured checks that an explicit setting
// is the developer's choice, and needs no comment.
func TestNothingIsLoggedWhenReasoningIsConfigured(t *testing.T) {
	buf := captureLogs(t)
	paramsOf(t, NewLLM(Config{Model: "claude-sonnet-6", Thinking: &ThinkingConfig{Type: "adaptive"}}))
	paramsOf(t, NewLLM(Config{Model: "claude-sonnet-6", Effort: "low"}))
	paramsOf(t, NewLLM(Config{
		Model: "claude-sonnet-6",
		Extra: map[string]any{"output_config": map[string]any{"effort": "low"}},
	}))
	if buf.Len() != 0 {
		t.Errorf("logged %q, want nothing", buf.String())
	}
}

// --- every inference path ---

// TestRunInferenceAppliesTheThinkingDefault checks that out-of-band inference
// gets the same default as the in-pipeline path.
func TestRunInferenceAppliesTheThinkingDefault(t *testing.T) {
	body := requestBody(t, Config{Model: "claude-sonnet-5"}, llm.InferenceOptions{})
	if want := obj("type", "disabled"); !reflect.DeepEqual(body["thinking"], want) {
		t.Errorf("thinking = %v, want %v", body["thinking"], want)
	}
}

// TestStreamingAppliesTheThinkingDefaultAndEffort checks that the in-pipeline
// request carries the default, and the effort setting when set.
func TestStreamingAppliesTheThinkingDefaultAndEffort(t *testing.T) {
	first := streamedRequestBody(t, Config{Model: "claude-sonnet-5"})
	second := streamedRequestBody(t, Config{Model: "claude-sonnet-5-5", Effort: "low"})

	if want := obj("type", "disabled"); !reflect.DeepEqual(first["thinking"], want) {
		t.Errorf("first thinking = %v, want %v", first["thinking"], want)
	}
	if _, ok := first["output_config"]; ok {
		t.Errorf("first output_config = %v, want none", first["output_config"])
	}
	if _, ok := second["thinking"]; ok {
		t.Errorf("second thinking = %v, want none", second["thinking"])
	}
	if want := obj("effort", "low"); !reflect.DeepEqual(second["output_config"], want) {
		t.Errorf("second output_config = %v, want %v", second["output_config"], want)
	}
}

// TestConfigAcceptsBetweenTools checks that the thinking type Sonnet 5.5 needs
// passes validation and is sent as configured.
func TestConfigAcceptsBetweenTools(t *testing.T) {
	cfg := Config{Model: "claude-sonnet-5-5", Thinking: &ThinkingConfig{Type: "between_tools"}}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if got := paramsOf(t, NewLLM(cfg))["thinking"]; !reflect.DeepEqual(got, obj("type", "between_tools")) {
		t.Errorf("thinking = %v, want between_tools", got)
	}
}

// --- use_provider_thinking_default ---

// TestUseProviderThinkingDefaultRunsAtAnthropicsDefault checks the setting
// sends no default and logs the reason once, at DEBUG.
func TestUseProviderThinkingDefaultRunsAtAnthropicsDefault(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&syncWriter{w: &buf}, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	s := NewLLM(Config{Model: "claude-opus-5-5"})
	s.SetUseProviderThinkingDefault(true)

	params := paramsOf(t, s)
	paramsOf(t, s)

	if got := thinkingFields(params); len(got) != 0 {
		t.Errorf("sent %v, want no thinking default", got)
	}
	var lines []string
	for line := range strings.SplitSeq(strings.TrimSpace(buf.String()), "\n") {
		if strings.Contains(line, "thinking") {
			lines = append(lines, line)
		}
	}
	if len(lines) != 1 || !strings.Contains(lines[0], "level=DEBUG") ||
		!strings.Contains(lines[0], "use_provider_thinking_default is set") ||
		!strings.Contains(lines[0], "model=claude-opus-5-5") {
		t.Errorf("logged %q, want one DEBUG line saying why", lines)
	}
}
