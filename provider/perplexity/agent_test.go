package perplexity

import (
	"bytes"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/openai/responses"
)

// completedEvent ends a response.
const completedEvent = `{"type":"response.completed","response":{"id":"r1"}}`

// agentServer stands in for the Agent API: it records each request body and
// streams events back.
type agentServer struct {
	*httptest.Server
	bodies []map[string]any
}

func newAgentServer(t *testing.T, events []string) *agentServer {
	t.Helper()
	srv := &agentServer{}
	srv.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		srv.bodies = append(srv.bodies, body)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range events {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// request runs one turn of a service built from cfg against a stub and
// returns the body it sent.
func request(t *testing.T, cfg AgentConfig, convo *frames.LLMContext) map[string]any {
	t.Helper()
	srv := newAgentServer(t, []string{completedEvent})
	cfg.APIKey, cfg.BaseURL = "test-key", srv.URL
	if convo == nil {
		convo = hello()
	}
	if err := NewAgentLLM(cfg).GenerateWithTools(t.Context(), convo, &calls{}); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	return srv.bodies[0]
}

func hello() *frames.LLMContext {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hi")
	return convo
}

// calls is a sink that records the function calls a turn reports.
type calls struct{ names []string }

func (c *calls) Text(string) error { return nil }

func (c *calls) Tool(call frames.ToolCall) error {
	c.names = append(c.names, call.Name)
	return nil
}

// toolTypes lists the type of each tool a request offers.
func toolTypes(body map[string]any) []string {
	tools, _ := body["tools"].([]any)
	var types []string
	for _, tool := range tools {
		if t, ok := tool.(map[string]any); ok {
			s, _ := t["type"].(string)
			types = append(types, s)
		}
	}
	return types
}

// TestNewAgentLLM checks the service's label.
func TestNewAgentLLM(t *testing.T) {
	providertest.Service(t, "PerplexityAgentHTTPLLM", NewAgentLLM(AgentConfig{APIKey: "k"}))
}

// TestDefaultRequestUsesTheFastPresetWithAPinnedModel covers the default: the
// preset decides how much to reason, and web search is on.
func TestDefaultRequestUsesTheFastPresetWithAPinnedModel(t *testing.T) {
	body := request(t, AgentConfig{}, nil)

	if body["model"] != "openai/gpt-5.6-luna" {
		t.Errorf("model = %v, want openai/gpt-5.6-luna", body["model"])
	}
	if body["preset"] != "fast" {
		t.Errorf("preset = %v, want fast", body["preset"])
	}
	if _, ok := body["reasoning"]; ok {
		t.Errorf("reasoning = %v, want none: the preset decides", body["reasoning"])
	}
	if got := toolTypes(body); !reflect.DeepEqual(got, []string{"web_search"}) {
		t.Errorf("tools = %v, want web_search", got)
	}
}

// TestPresetAloneChoosesTheModel covers a preset left to choose the model.
func TestPresetAloneChoosesTheModel(t *testing.T) {
	body := request(t, AgentConfig{PresetModel: true, Preset: "fast"}, nil)

	if _, ok := body["model"]; ok {
		t.Errorf("model = %v, want none", body["model"])
	}
	if body["preset"] != "fast" {
		t.Errorf("preset = %v, want fast", body["preset"])
	}
	if _, ok := body["reasoning"]; ok {
		t.Errorf("reasoning = %v, want none", body["reasoning"])
	}
}

// TestNeitherModelNorPresetIsRejected covers a config that leaves the API
// nothing to choose the model with.
func TestNeitherModelNorPresetIsRejected(t *testing.T) {
	err := AgentConfig{APIKey: "k", PresetModel: true}.Validate()
	if !errors.Is(err, errModelOrPresetRequired) {
		t.Errorf("Validate = %v, want the model-or-preset error", err)
	}
	if err := (AgentConfig{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("Validate default = %v, want nil", err)
	}
	if err := (AgentConfig{}).Validate(); err == nil {
		t.Error("Validate without an API key = nil, want an error")
	}
}

// TestEveryModelGetsTheFastPreset covers models from every provider Perplexity
// serves.
func TestEveryModelGetsTheFastPreset(t *testing.T) {
	for _, model := range []string{
		"openai/gpt-5.5", "anthropic/claude-haiku-4-5", "google/gemini-3.8-flash", "xai/grok-4.6",
	} {
		body := request(t, AgentConfig{Model: model}, nil)
		if body["model"] != model || body["preset"] != "fast" {
			t.Errorf("%s: model = %v, preset = %v, want the model with the fast preset",
				model, body["model"], body["preset"])
		}
		if _, ok := body["reasoning"]; ok {
			t.Errorf("%s: reasoning = %v, want none", model, body["reasoning"])
		}
	}
}

// TestWebSearchJoinsContextToolsOnce covers web search offered beside the
// application's own tools.
func TestWebSearchJoinsContextToolsOnce(t *testing.T) {
	convo := hello()
	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "Get the weather.",
		Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},` +
			`"required":["city"]}`),
	}})

	body := request(t, AgentConfig{}, convo)

	if got := toolTypes(body); !reflect.DeepEqual(got, []string{"function", "web_search"}) {
		t.Errorf("tools = %v, want function then web_search", got)
	}
}

// TestWebSearchCanBeTurnedOff covers NoWebSearch.
func TestWebSearchCanBeTurnedOff(t *testing.T) {
	if body := request(t, AgentConfig{NoWebSearch: true}, nil); body["tools"] != nil {
		t.Errorf("tools = %v, want none", body["tools"])
	}
}

// TestPerplexityFieldsJoinExtra covers Perplexity's fields beside ones set
// through Extra.
func TestPerplexityFieldsJoinExtra(t *testing.T) {
	body := request(t, AgentConfig{MaxSteps: 2, Extra: map[string]any{"language_preference": "en"}}, nil)

	if body["language_preference"] != "en" || body["preset"] != "fast" || body["max_steps"] != float64(2) {
		t.Errorf("body = %v, want language_preference, the fast preset and max_steps together", body)
	}
}

// TestUsageReportsCacheCreationTokens covers the field Perplexity reports the
// tokens written to the prompt cache under.
func TestUsageReportsCacheCreationTokens(t *testing.T) {
	raw := json.RawMessage(`{"input_tokens":3046,"input_tokens_details":{"cached_tokens":1780,` +
		`"cache_creation_input_tokens":1263,"cache_read_input_tokens":1780},"output_tokens":29,` +
		`"output_tokens_details":{"reasoning_tokens":0},"total_tokens":3075}`)
	tokens := frames.LLMTokenUsage{PromptTokens: 3046, TotalTokens: 3075}

	cacheCreationTokens(raw, &tokens)

	if tokens.CacheCreationTokens == nil || *tokens.CacheCreationTokens != 1263 {
		t.Errorf("cache creation tokens = %v, want 1263", tokens.CacheCreationTokens)
	}
}

// TestAPresetKeepsTheModelUnlessItIsCleared covers a preset beside a model:
// the model replaces the preset's own unless the preset is to choose it.
func TestAPresetKeepsTheModelUnlessItIsCleared(t *testing.T) {
	if body := request(t, AgentConfig{Preset: "high"}, nil); body["model"] != "openai/gpt-5.6-luna" {
		t.Errorf("model = %v, want openai/gpt-5.6-luna", body["model"])
	}
	if body := request(t, AgentConfig{Preset: "high", PresetModel: true}, nil); body["model"] != nil {
		t.Errorf("model = %v, want none", body["model"])
	}
}

// TestAnExplicitModelWithAPresetKeepsThePresetReasoning covers a model chosen
// beside a preset, which still decides the reasoning.
func TestAnExplicitModelWithAPresetKeepsThePresetReasoning(t *testing.T) {
	body := request(t, AgentConfig{Preset: "high", Model: "openai/gpt-5.5"}, nil)

	if body["model"] != "openai/gpt-5.5" || body["preset"] != "high" {
		t.Errorf("model = %v, preset = %v, want openai/gpt-5.5 with high", body["model"], body["preset"])
	}
	if _, ok := body["reasoning"]; ok {
		t.Errorf("reasoning = %v, want none", body["reasoning"])
	}
}

// TestExplicitReasoningIsSentAsConfigured covers reasoning set by the caller,
// which replaces the fast preset.
func TestExplicitReasoningIsSentAsConfigured(t *testing.T) {
	body := request(t, AgentConfig{Reasoning: &responses.ReasoningConfig{Effort: "low"}}, nil)

	if got, _ := body["reasoning"].(map[string]any); got["effort"] != "low" {
		t.Errorf("reasoning = %v, want effort low", body["reasoning"])
	}
	if _, ok := body["preset"]; ok {
		t.Errorf("preset = %v, want none", body["preset"])
	}
}

// TestAPresetInExtraIsLeftAlone covers a preset set through Extra.
func TestAPresetInExtraIsLeftAlone(t *testing.T) {
	if body := request(t, AgentConfig{Extra: map[string]any{"preset": "high"}}, nil); body["preset"] != "high" {
		t.Errorf("preset = %v, want high", body["preset"])
	}
}

// providerDefaultRequest runs one turn of a service built from cfg, with
// UseProviderThinkingDefault set, and returns the body it sent.
func providerDefaultRequest(t *testing.T, cfg AgentConfig) map[string]any {
	t.Helper()
	srv := newAgentServer(t, []string{completedEvent})
	cfg.APIKey, cfg.BaseURL = "test-key", srv.URL
	svc := NewAgentLLM(cfg)
	svc.SetUseProviderThinkingDefault(true)
	if err := svc.GenerateWithTools(t.Context(), hello(), &calls{}); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	return srv.bodies[0]
}

// TestProviderThinkingDefaultSendsNoPreset covers UseProviderThinkingDefault:
// the "fast" preset is not sent, and neither is reasoning, so the model runs at
// Perplexity's own default.
func TestProviderThinkingDefaultSendsNoPreset(t *testing.T) {
	body := providerDefaultRequest(t, AgentConfig{})

	if body["model"] != "openai/gpt-5.6-luna" {
		t.Errorf("model = %v, want openai/gpt-5.6-luna", body["model"])
	}
	if _, ok := body["preset"]; ok {
		t.Errorf("preset = %v, want none", body["preset"])
	}
	if _, ok := body["reasoning"]; ok {
		t.Errorf("reasoning = %v, want none", body["reasoning"])
	}
}

// TestProviderThinkingDefaultKeepsAChosenPreset covers a preset the developer
// chose, which UseProviderThinkingDefault leaves in place: it only stops the
// default being sent.
func TestProviderThinkingDefaultKeepsAChosenPreset(t *testing.T) {
	if body := providerDefaultRequest(t, AgentConfig{Preset: "fast"}); body["preset"] != "fast" {
		t.Errorf("preset = %v, want the chosen fast", body["preset"])
	}
}

// infoLogs routes the default logger to a buffer at INFO and above until the
// test ends.
func infoLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &buf
}

// TestTheDefaultPresetIsLoggedOnce covers what the developer is told about the
// preset sent for them: once, not on every request.
func TestTheDefaultPresetIsLoggedOnce(t *testing.T) {
	srv := newAgentServer(t, []string{completedEvent})
	svc := NewAgentLLM(AgentConfig{APIKey: "k", BaseURL: srv.URL})
	logs := infoLogs(t)

	for range 2 {
		if err := svc.GenerateWithTools(t.Context(), hello(), &calls{}); err != nil {
			t.Fatal(err)
		}
	}

	var lines []string
	for line := range strings.SplitSeq(strings.TrimSpace(logs.String()), "\n") {
		if strings.Contains(line, "preset") {
			lines = append(lines, line)
		}
	}
	if len(lines) != 1 || !strings.Contains(lines[0], "level=INFO") ||
		!strings.Contains(lines[0], `setting="preset=\"fast\""`) {
		t.Errorf("logged %q, want one INFO line about the fast preset", lines)
	}
}

// TestAChosenPresetIsNotLogged covers a preset the developer chose.
func TestAChosenPresetIsNotLogged(t *testing.T) {
	logs := infoLogs(t)
	request(t, AgentConfig{Preset: "high"}, nil)
	if strings.Contains(logs.String(), "preset") {
		t.Errorf("logged %q, want nothing about the preset", logs.String())
	}
}

// filtered runs events through a fresh filter and returns what passes.
func filtered(events ...string) []string {
	filter := withoutServerToolCalls()
	var out []string
	for _, ev := range events {
		for _, e := range filter([]byte(ev)) {
			out = append(out, string(e))
		}
	}
	return out
}

const (
	searchAdded = `{"type":"response.output_item.added","output_index":0,` +
		`"item":{"type":"function_call","id":"fc_1","call_id":"call_fc_1","name":"search_web"}}`
	searchDone = `{"type":"response.output_item.done","output_index":0,"item":{"type":"search_results"}}`
	textDelta  = `{"type":"response.output_text.delta","delta":"Sunny."}`
	callAdded  = `{"type":"response.output_item.added","output_index":1,` +
		`"item":{"type":"function_call","id":"fc_2","call_id":"call_fc_2","name":"get_weather"}}`
	argsDelta = `{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc_2",` +
		`"delta":"{\"city\": \"Paris\"}"}`
	callDone = `{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call",` +
		`"id":"fc_2","call_id":"call_fc_2","name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}`
)

// TestASearchTheServerRunsIsDropped covers the web search Perplexity runs
// itself, announced as a function call and completed as search results.
func TestASearchTheServerRunsIsDropped(t *testing.T) {
	got := filtered(searchAdded, searchDone, textDelta)
	if want := []string{searchDone, textDelta}; !reflect.DeepEqual(got, want) {
		t.Errorf("passed %q, want %q", got, want)
	}
}

// TestAFunctionCallPassesThroughInOrder covers an application function call:
// its announcement waits for its output to complete, and other events do not.
func TestAFunctionCallPassesThroughInOrder(t *testing.T) {
	got := filtered(callAdded, argsDelta, textDelta, callDone)
	if want := []string{argsDelta, textDelta, callAdded, callDone}; !reflect.DeepEqual(got, want) {
		t.Errorf("passed %q, want %q", got, want)
	}
}

// TestOnlyApplicationFunctionCallsRun covers a whole turn: the search the
// server ran is not reported as a call, and the application's call is, with
// its arguments.
func TestOnlyApplicationFunctionCallsRun(t *testing.T) {
	srv := newAgentServer(t, []string{searchAdded, searchDone, callAdded, callDone, completedEvent})
	svc := NewAgentLLM(AgentConfig{APIKey: "k", BaseURL: srv.URL})
	sink := &argsSink{}

	if err := svc.GenerateWithTools(t.Context(), hello(), sink); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}

	if len(sink.calls) != 1 || sink.calls[0].Name != "get_weather" {
		t.Fatalf("calls = %v, want get_weather alone", sink.calls)
	}
	if string(sink.calls[0].Args) != `{"city": "Paris"}` {
		t.Errorf("arguments = %s, want the completed item's", sink.calls[0].Args)
	}
}

// argsSink records the function calls a turn reports, with their arguments.
type argsSink struct{ calls []frames.ToolCall }

func (s *argsSink) Text(string) error { return nil }

func (s *argsSink) Tool(call frames.ToolCall) error {
	s.calls = append(s.calls, call)
	return nil
}
