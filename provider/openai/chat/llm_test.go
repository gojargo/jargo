package chat

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// This package cannot use internal/providertest: that package builds on this one
// to assert the compatible providers, so importing it here would be a cycle. The
// two assertions it would have contributed are spelled out below.

// wantLabel checks a constructor returned a usable service carrying label.
// processor.New appends a "#<id>" instance counter, so only the label the
// provider chose is stable across runs.
func wantLabel(t *testing.T, label string, svc interface{ Name() string }) {
	t.Helper()
	if svc == nil {
		t.Fatalf("constructor returned nil, want a %s service", label)
	}
	if got := svc.Name(); !strings.HasPrefix(got, label+"#") {
		t.Errorf("Name() = %q, want the %q label", got, label)
	}
}

// sse renders data lines as the chat-completions stream delivers them: one
// "data:" line per chunk, terminated by the [DONE] sentinel.
func sse(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

// contentChunk is one streamed text delta.
func contentChunk(text string) string {
	return `{"choices":[{"delta":{"content":` + quote(text) + `}}]}`
}

// quote renders s as a JSON string. The test inputs are plain text, for which Go
// and JSON quoting agree.
func quote(s string) string { return strconv.Quote(s) }

// errDownstream stands in for a downstream consumer that has gone away.
//
//nolint:gochecknoglobals // sentinel error for the tests below
var errDownstream = errors.New("downstream closed")

// llmServer stands in for the chat-completions endpoint, recording the one
// request it receives and replying with body.
type llmServer struct {
	*httptest.Server
	path   string
	header http.Header
	body   map[string]any
}

func newLLMServer(t *testing.T, reply string) *llmServer {
	t.Helper()
	s := &llmServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.path = r.URL.RequestURI()
		s.header = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&s.body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(s.Close)
	return s
}

// generate runs one plain generation against srv and returns the streamed text.
func generate(t *testing.T, srv *llmServer, cfg LLMConfig) string {
	t.Helper()
	cfg.BaseURL = srv.URL
	svc := NewLLM(cfg)

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")

	var out strings.Builder
	if err := svc.Generate(t.Context(), convo, func(text string) error {
		out.WriteString(text)
		return nil
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return out.String()
}

// TestLLMConfigValidate pins the one thing the config insists on: nothing. An
// OpenAI-compatible endpoint may need no credential at all (a local Ollama, for
// one), so the key is the caller's business and an empty config still builds.
func TestLLMConfigValidate(t *testing.T) {
	if err := (LLMConfig{}).Validate(); err != nil {
		t.Errorf("Validate() on an empty config = %v, want nil", err)
	}
	if err := (LLMConfig{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	wantLabel(t, "OpenAILLM", NewLLM(LLMConfig{APIKey: "k"}))
	wantLabel(t, "OpenAISTT", NewSTT(STTConfig{APIKey: "k"}))
	wantLabel(t, "OpenAITTS", NewTTS(TTSConfig{APIKey: "k"}))
}

// TestNewCompatLLMDefaults checks the three fields a compatible provider hands
// in are used only to fill gaps: a caller who sets them keeps their own.
func TestNewCompatLLMDefaults(t *testing.T) {
	compat := Compat{Name: "GroqLLM", BaseURL: "https://base.example", DefaultModel: "default-model"}
	svc := NewCompatLLM(compat, LLMConfig{APIKey: "k"})
	if svc.cfg.BaseURL != "https://base.example" {
		t.Errorf("BaseURL = %q, want the provider default", svc.cfg.BaseURL)
	}
	if svc.cfg.Model != "default-model" {
		t.Errorf("Model = %q, want the provider default", svc.cfg.Model)
	}
	if svc.cfg.MaxTokens != 0 {
		t.Errorf("MaxTokens = %d, want it left unset so the API's own bound stands", svc.cfg.MaxTokens)
	}

	override := NewCompatLLM(compat, LLMConfig{
		BaseURL: "https://mine.example", Model: "mine", MaxTokens: 7,
	})
	if override.cfg.BaseURL != "https://mine.example" || override.cfg.Model != "mine" || override.cfg.MaxTokens != 7 {
		t.Errorf("configured values were overwritten by the defaults: %+v", override.cfg)
	}
}

// TestNewLLMDefaults checks the plain OpenAI constructor picks OpenAI's own base
// URL and model.
func TestNewLLMDefaults(t *testing.T) {
	svc := NewLLM(LLMConfig{APIKey: "k"})
	if svc.cfg.BaseURL != defaultLLMBaseURL {
		t.Errorf("BaseURL = %q, want %q", svc.cfg.BaseURL, defaultLLMBaseURL)
	}
	if svc.cfg.Model != defaultLLMModel {
		t.Errorf("Model = %q, want %q", svc.cfg.Model, defaultLLMModel)
	}
}

// TestGenerateRequestShape checks a plain generation is addressed, authorized
// and streamed the way the chat-completions API expects.
func TestGenerateRequestShape(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("hi")))
	if got := generate(t, srv, LLMConfig{APIKey: "test-key"}); got != "hi" {
		t.Errorf("streamed text = %q, want %q", got, "hi")
	}

	if srv.path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", srv.path)
	}
	if got := srv.header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", got)
	}
	if got := srv.header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}
	if srv.body["stream"] != true {
		t.Errorf("stream = %v, want true", srv.body["stream"])
	}
	if srv.body["model"] != defaultLLMModel {
		t.Errorf("model = %v, want %q", srv.body["model"], defaultLLMModel)
	}
	if _, ok := srv.body["max_tokens"]; ok {
		t.Errorf("body carries a max_tokens nobody asked for: %v", srv.body)
	}
	if got := srv.body["stream_options"]; !reflect.DeepEqual(got, map[string]any{"include_usage": true}) {
		t.Errorf("stream_options = %v, want the usage counts requested", got)
	}
}

// TestGenerateSendsSystemThenMessages checks the system prompt leads the message
// list rather than traveling as an ordinary turn.
func TestGenerateSendsSystemThenMessages(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("ok")))
	generate(t, srv, LLMConfig{APIKey: "k"})

	msgs, ok := srv.body["messages"].([]any)
	if !ok || len(msgs) != 2 {
		t.Fatalf("messages = %v, want the system prompt and the user turn", srv.body["messages"])
	}
	first, _ := msgs[0].(map[string]any)
	second, _ := msgs[1].(map[string]any)
	if first["role"] != "system" || first["content"] != "be brief" {
		t.Errorf("message 0 = %v, want the system prompt", first)
	}
	if second["role"] != "user" || second["content"] != "hello" {
		t.Errorf("message 1 = %v, want the user turn", second)
	}
}

// TestGenerateOmitsUnsetSamplingControls checks the pointer-valued controls are
// left off the request when nil, so the API applies its own defaults rather than
// receiving a Go zero value that means something else.
func TestGenerateOmitsUnsetSamplingControls(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("ok")))
	generate(t, srv, LLMConfig{APIKey: "k"})

	for _, field := range []string{
		"temperature", "top_p", "frequency_penalty", "presence_penalty", "seed",
		"max_completion_tokens", "tools",
	} {
		if _, present := srv.body[field]; present {
			t.Errorf("%s was sent for an unset config: %v", field, srv.body[field])
		}
	}
}

// TestGenerateSendsSamplingControls checks a deliberate zero still crosses the
// wire, which is the reason the fields are pointers.
func TestGenerateSendsSamplingControls(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("ok")))
	zero, one := 0.0, 1.0
	seed, completion := 42, 256
	generate(t, srv, LLMConfig{
		APIKey:              "k",
		Temperature:         &zero,
		TopP:                &one,
		FrequencyPenalty:    &zero,
		PresencePenalty:     &one,
		Seed:                &seed,
		MaxCompletionTokens: &completion,
	})

	want := map[string]any{
		"temperature":           0.0,
		"top_p":                 1.0,
		"frequency_penalty":     0.0,
		"presence_penalty":      1.0,
		"seed":                  float64(42),
		"max_completion_tokens": float64(256),
	}
	for field, wantVal := range want {
		if got, present := srv.body[field]; !present || got != wantVal {
			t.Errorf("%s = %v (present %v), want %v", field, got, present, wantVal)
		}
	}
}

// TestGenerateExtraOverridesModeledFields checks Extra is merged over the
// modeled body, so a provider-specific parameter can be added and a modeled one
// replaced without a new config field.
func TestGenerateExtraOverridesModeledFields(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("ok")))
	generate(t, srv, LLMConfig{
		APIKey: "k",
		Extra:  map[string]any{"reasoning_effort": "low", "max_tokens": 99},
	})

	if srv.body["reasoning_effort"] != "low" {
		t.Errorf("reasoning_effort = %v, want the extra field", srv.body["reasoning_effort"])
	}
	if srv.body["max_tokens"] != float64(99) {
		t.Errorf("max_tokens = %v, want the extra field to win over the modeled one", srv.body["max_tokens"])
	}
}

// TestGenerateJoinsDeltas checks the streamed content deltas arrive at the
// caller in order and nothing else does.
func TestGenerateJoinsDeltas(t *testing.T) {
	srv := newLLMServer(t, sse(
		contentChunk("Hello"),
		contentChunk(", "),
		`{"choices":[]}`, // A chunk with no choices contributes nothing.
		`not json`,       // Neither does a malformed one.
		contentChunk("world"),
	))
	if got := generate(t, srv, LLMConfig{APIKey: "k"}); got != "Hello, world" {
		t.Errorf("streamed text = %q, want %q", got, "Hello, world")
	}
}

// TestGenerateStatusError checks a non-200 reply is reported as an error rather
// than read as an empty stream, and that it carries the body for diagnosis.
func TestGenerateStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("slow down"))
	}))
	defer srv.Close()

	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(string) error { return nil })
	if !errors.Is(err, errStatus) {
		t.Fatalf("Generate error = %v, want errStatus", err)
	}
	if !strings.Contains(err.Error(), "slow down") {
		t.Errorf("error = %v, want it to carry the response body", err)
	}
}

// TestGenerateEmitError checks an error from the caller's sink stops the stream
// and is reported back rather than swallowed.
func TestGenerateEmitError(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("a"), contentChunk("b")))
	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})

	calls := 0
	err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(string) error {
		calls++
		return errDownstream
	})
	if !errors.Is(err, errDownstream) {
		t.Fatalf("Generate error = %v, want the sink's error", err)
	}
	if calls != 1 {
		t.Errorf("sink called %d times, want the stream to stop at the first error", calls)
	}
}

// TestGenerateWithToolsAdvertisesTools checks the context's tools are sent on
// the request in OpenAI's function-tool shape.
func TestGenerateWithToolsAdvertisesTools(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("ok")))
	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("weather?")
	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "Look up the weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})

	sink := &fakeSink{}
	if err := svc.GenerateWithTools(t.Context(), convo, sink); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}

	tools, ok := srv.body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want the one advertised tool", srv.body["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool type = %v, want %q", tool["type"], "function")
	}
	fn, _ := tool["function"].(map[string]any)
	if fn["name"] != "get_weather" || fn["description"] != "Look up the weather" {
		t.Errorf("function = %v, want the advertised name and description", fn)
	}
}

// TestGenerateWithToolsOmitsToolsWhenNoneDeclared checks a tool-capable
// generation over a context with no tools sends no tools field at all, rather
// than an empty list the API would have to interpret.
func TestGenerateWithToolsOmitsToolsWhenNoneDeclared(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("ok")))
	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})

	sink := &fakeSink{}
	if err := svc.GenerateWithTools(t.Context(), frames.NewLLMContext(""), sink); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	if _, present := srv.body["tools"]; present {
		t.Errorf("tools = %v, want the field omitted", srv.body["tools"])
	}
	if sink.text.String() != "ok" {
		t.Errorf("text = %q, want the streamed delta", sink.text.String())
	}
}

// TestGenerateWithToolsCoalescesStreamedCalls checks the whole streamed path:
// text arrives live, and the tool-call fragments OpenAI spreads across deltas
// are assembled into whole calls reported once the stream ends.
func TestGenerateWithToolsCoalescesStreamedCalls(t *testing.T) {
	srv := newLLMServer(t, sse(
		contentChunk("Let me check. "),
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a",`+
			`"function":{"name":"get_weather","arguments":"{\"loc"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"Paris\"}"}}]}}]}`,
	))
	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})

	sink := &fakeSink{}
	if err := svc.GenerateWithTools(t.Context(), frames.NewLLMContext(""), sink); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	if sink.text.String() != "Let me check. " {
		t.Errorf("text = %q, want the preamble", sink.text.String())
	}
	if len(sink.calls) != 1 {
		t.Fatalf("calls = %+v, want the one assembled call", sink.calls)
	}
	got := sink.calls[0]
	if got.ID != "call_a" || got.Name != "get_weather" || string(got.Args) != `{"location":"Paris"}` {
		t.Errorf("call = %+v (args %s), want the fragments joined", got, got.Args)
	}
}

// TestGenerateWithToolsStatusError checks the tool-capable path reports a
// non-200 the same way the plain one does.
func TestGenerateWithToolsStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("bad key"))
	}))
	defer srv.Close()

	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	err := svc.GenerateWithTools(t.Context(), frames.NewLLMContext(""), &fakeSink{})
	if !errors.Is(err, errStatus) {
		t.Fatalf("GenerateWithTools error = %v, want errStatus", err)
	}
}

// shaper addresses and authorizes requests the way a deployment with its own URL
// layout does, which is what Azure OpenAI needs from this base.
type shaper struct{ key string }

func (shaper) Endpoint(baseURL string) string {
	return baseURL + "/openai/deployments/mine/chat/completions?api-version=2024-06-01"
}

func (s *shaper) Authorize(req *http.Request, apiKey string) {
	s.key = apiKey
	req.Header.Set("api-key", apiKey)
}

// TestCompatShaperUsedForURLAndAuth checks a custom shaper decides both the URL
// and the authorization, so a non-OpenAI URL layout reuses this implementation
// whole.
func TestCompatShaperUsedForURLAndAuth(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("ok")))
	sh := &shaper{}
	svc := NewCompatLLM(Compat{
		Name: "AzureLLM", BaseURL: srv.URL, DefaultModel: "gpt-4o", Shaper: sh,
	}, LLMConfig{APIKey: "azure-key"})

	if err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if srv.path != "/openai/deployments/mine/chat/completions?api-version=2024-06-01" {
		t.Errorf("path = %q, want the shaper's URL including its query", srv.path)
	}
	if got := srv.header.Get("api-key"); got != "azure-key" {
		t.Errorf("api-key header = %q, want the shaper's scheme", got)
	}
	if srv.header.Get("Authorization") != "" {
		t.Error("the default Bearer header was sent alongside the shaper's scheme")
	}
	if sh.key != "azure-key" {
		t.Errorf("shaper saw key %q, want the configured one", sh.key)
	}
}

// TestEncodeBodyWithoutExtra checks the fast path: with no extra fields the
// modeled request is marshaled directly.
func TestEncodeBodyWithoutExtra(t *testing.T) {
	raw, err := encodeBody(chatRequest{Model: "m", Stream: true}, nil)
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["model"] != "m" || got["stream"] != true {
		t.Errorf("body = %v, want the modeled fields", got)
	}
}
