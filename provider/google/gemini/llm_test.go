package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/service/llm"
)

// sse renders data lines as streamGenerateContent delivers them with alt=sse.
func sse(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	return b.String()
}

// textChunk is one streamed text part.
func textChunk(text string) string {
	// The test inputs are plain text, for which Go and JSON quoting agree.
	return `{"candidates":[{"content":{"parts":[{"text":` + strconv.Quote(text) + `}]}}]}`
}

// errNoToken stands in for an authorization scheme that cannot mint a token.
//
//nolint:gochecknoglobals // sentinel error for the tests below
var errNoToken = errors.New("no token")

// genServer stands in for the generateContent endpoint, recording the one
// request it receives and replying with body.
type genServer struct {
	*httptest.Server
	path   string
	header http.Header
	body   map[string]any
}

func newGenServer(t *testing.T, reply string) *genServer {
	t.Helper()
	s := &genServer{}
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

// testShaper points the service at srv while keeping the Gemini URL layout, so
// the request the real shaper would build is the one under test.
type testShaper struct {
	base   string
	model  string
	authed bool
}

func (s *testShaper) Endpoint(model string, stream bool) string {
	s.model = model
	if !stream {
		return s.base + "/" + model + ":generateContent"
	}
	return s.base + "/" + model + ":streamGenerateContent?alt=sse"
}

func (s *testShaper) Authorize(_ context.Context, req *http.Request) error {
	s.authed = true
	req.Header.Set("x-goog-api-key", "test-key")
	return nil
}

// serviceFor builds a service that talks to srv through sh.
func serviceFor(srv *genServer, sh RequestShaper, cfg Config) *Service {
	return NewShapedLLM("GoogleLLM", sh, cfg)
}

// generate runs one plain generation against srv and returns the streamed text.
func generate(t *testing.T, srv *genServer, cfg Config) string {
	t.Helper()
	svc := serviceFor(srv, &testShaper{base: srv.URL}, cfg)

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

// TestConfigValidate pins the credential the Gemini API requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: Config{}, Valid: false},
		{Name: "API key only", Cfg: Config{APIKey: "k"}, Valid: true},
		{Name: "missing STT key", Cfg: STTConfig{}, Valid: false},
		{Name: "STT key only", Cfg: STTConfig{APIKey: "k"}, Valid: true},
		{Name: "missing TTS key", Cfg: TTSConfig{}, Valid: false},
		{Name: "TTS key only", Cfg: TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "GoogleLLM", NewLLM(Config{APIKey: "k"}))
	providertest.Service(t, "GoogleSTT", NewSTT(STTConfig{APIKey: "k"}))
	providertest.Service(t, "GoogleTTS", NewTTS(TTSConfig{APIKey: "k"}))
}

// TestNewLLMDefaults checks the model and token cap fall back to the low-latency
// defaults, and that a configured value is kept.
func TestNewLLMDefaults(t *testing.T) {
	svc := NewLLM(Config{APIKey: "k"})
	if svc.cfg.Model != defaultModel {
		t.Errorf("Model = %q, want %q", svc.cfg.Model, defaultModel)
	}
	if svc.cfg.MaxTokens != defaultMaxTokens {
		t.Errorf("MaxTokens = %d, want %d", svc.cfg.MaxTokens, defaultMaxTokens)
	}

	override := NewLLM(Config{APIKey: "k", Model: "gemini-2.5-pro", MaxTokens: 42})
	if override.cfg.Model != "gemini-2.5-pro" || override.cfg.MaxTokens != 42 {
		t.Errorf("configured values were overwritten: %+v", override.cfg)
	}
}

// TestAPIKeyShaper checks the default addressing and authorization: the model is
// named in the path, the stream is requested as SSE, and the key travels in the
// header Google expects rather than the query string.
func TestAPIKeyShaper(t *testing.T) {
	sh := apiKeyShaper{apiKey: "secret"}
	oneShot := apiBase + "/gemini-2.5-flash:generateContent"
	if got := sh.Endpoint("gemini-2.5-flash", false); got != oneShot {
		t.Errorf("Endpoint(one-shot) = %q, want %q", got, oneShot)
	}
	want := apiBase + "/gemini-2.5-flash:streamGenerateContent?alt=sse"
	if got := sh.Endpoint("gemini-2.5-flash", true); got != want {
		t.Errorf("Endpoint() = %q, want %q", got, want)
	}

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.invalid", nil)
	if err != nil {
		t.Fatalf("building request: %v", err)
	}
	if err := sh.Authorize(t.Context(), req); err != nil {
		t.Fatalf("Authorize: %v", err)
	}
	if got := req.Header.Get("x-goog-api-key"); got != "secret" {
		t.Errorf("x-goog-api-key = %q, want the configured key", got)
	}
}

// TestGenerateRequestShape checks a plain generation is addressed at the
// configured model, authorized by the shaper, and carries the generation config.
func TestGenerateRequestShape(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("hi")))
	sh := &testShaper{base: srv.URL}
	svc := serviceFor(srv, sh, Config{APIKey: "k"})

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	var out strings.Builder
	if err := svc.Generate(t.Context(), convo, func(s string) error {
		out.WriteString(s)
		return nil
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if out.String() != "hi" {
		t.Errorf("streamed text = %q, want %q", out.String(), "hi")
	}
	if sh.model != defaultModel {
		t.Errorf("shaper was asked for model %q, want %q", sh.model, defaultModel)
	}
	if !sh.authed {
		t.Error("the shaper was not asked to authorize the request")
	}
	if got := srv.header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}
	if !strings.HasSuffix(srv.path, ":streamGenerateContent?alt=sse") {
		t.Errorf("path = %q, want the streaming SSE endpoint", srv.path)
	}

	cfg, ok := srv.body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig = %v, want the generation block", srv.body["generationConfig"])
	}
	if cfg["maxOutputTokens"] != float64(defaultMaxTokens) {
		t.Errorf("maxOutputTokens = %v, want %d", cfg["maxOutputTokens"], defaultMaxTokens)
	}
}

// TestGenerateSendsSystemInstruction checks the system prompt travels in its own
// block rather than as a turn, which is how Gemini takes it.
func TestGenerateSendsSystemInstruction(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	generate(t, srv, Config{APIKey: "k"})

	sys, ok := srv.body["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("systemInstruction = %v, want the system block", srv.body["systemInstruction"])
	}
	parts, ok := sys["parts"].([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("systemInstruction parts = %v, want one text part", sys["parts"])
	}
	part, _ := parts[0].(map[string]any)
	if part["text"] != "be brief" {
		t.Errorf("systemInstruction text = %v, want the system prompt", part["text"])
	}

	contents, ok := srv.body["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("contents = %v, want only the user turn", srv.body["contents"])
	}
	turn, _ := contents[0].(map[string]any)
	if turn["role"] != "user" {
		t.Errorf("contents[0] role = %v, want user", turn["role"])
	}
}

// TestGenerateOmitsSystemInstructionWhenEmpty checks a conversation with no
// system prompt sends no system block at all.
func TestGenerateOmitsSystemInstructionWhenEmpty(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	svc := serviceFor(srv, &testShaper{base: srv.URL}, Config{APIKey: "k"})
	if err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if _, present := srv.body["systemInstruction"]; present {
		t.Errorf("systemInstruction = %v, want the block omitted", srv.body["systemInstruction"])
	}
}

// TestGenerationConfigControls checks the sampling controls are omitted when nil
// and sent when set, including a deliberate zero.
func TestGenerationConfigControls(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	generate(t, srv, Config{APIKey: "k"})
	cfg, _ := srv.body["generationConfig"].(map[string]any)
	for _, field := range []string{"temperature", "topP", "topK"} {
		if _, present := cfg[field]; present {
			t.Errorf("%s was sent for an unset config: %v", field, cfg[field])
		}
	}

	zero, p := 0.0, 0.9
	k := 40
	generate(t, srv, Config{APIKey: "k", Temperature: &zero, TopP: &p, TopK: &k})
	cfg, _ = srv.body["generationConfig"].(map[string]any)
	if cfg["temperature"] != 0.0 {
		t.Errorf("temperature = %v, want a deliberate zero to be sent", cfg["temperature"])
	}
	if cfg["topP"] != 0.9 {
		t.Errorf("topP = %v, want 0.9", cfg["topP"])
	}
	if cfg["topK"] != float64(40) {
		t.Errorf("topK = %v, want 40", cfg["topK"])
	}
}

// TestGenerationConfigExtra checks Extra is merged into the generation config,
// so an unmodeled parameter can be set without a new field.
func TestGenerationConfigExtra(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	generate(t, srv, Config{
		APIKey: "k",
		Extra:  map[string]any{"responseMimeType": "text/plain", "maxOutputTokens": 7},
	})

	cfg, _ := srv.body["generationConfig"].(map[string]any)
	if cfg["responseMimeType"] != "text/plain" {
		t.Errorf("responseMimeType = %v, want the extra field", cfg["responseMimeType"])
	}
	if cfg["maxOutputTokens"] != float64(7) {
		t.Errorf("maxOutputTokens = %v, want the extra field to win over the modeled one", cfg["maxOutputTokens"])
	}
}

// TestGenerateJoinsParts checks every text part of every candidate reaches the
// caller in order, and that a malformed chunk is skipped rather than fatal.
func TestGenerateJoinsParts(t *testing.T) {
	srv := newGenServer(t, sse(
		textChunk("Hello"),
		`{"candidates":[{"content":{"parts":[{"text":", "},{"text":"world"}]}}]}`,
		`not json`,
	))
	if got := generate(t, srv, Config{APIKey: "k"}); got != "Hello, world" {
		t.Errorf("streamed text = %q, want %q", got, "Hello, world")
	}
}

// TestGenerateStatusError checks a non-200 is reported with the body attached
// rather than read as an empty stream.
func TestGenerateStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("quota exceeded"))
	}))
	defer srv.Close()

	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{APIKey: "k"})
	err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(string) error { return nil })
	if !errors.Is(err, errStatus) {
		t.Fatalf("Generate error = %v, want errStatus", err)
	}
	if !strings.Contains(err.Error(), "quota exceeded") {
		t.Errorf("error = %v, want it to carry the response body", err)
	}
}

// failingShaper refuses to authorize, standing in for a token mint that fails.
type failingShaper struct{ base string }

func (s failingShaper) Endpoint(model string, _ bool) string { return s.base + "/" + model }

func (failingShaper) Authorize(context.Context, *http.Request) error { return errNoToken }

// TestGenerateAuthorizeError checks a shaper that cannot authorize stops the
// request rather than sending an unauthenticated one.
func TestGenerateAuthorizeError(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	svc := NewShapedLLM("VertexLLM", failingShaper{base: srv.URL}, Config{APIKey: "k"})

	err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(string) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "no token") {
		t.Fatalf("Generate error = %v, want the shaper's error", err)
	}
	if srv.body != nil {
		t.Error("an unauthorized request was sent anyway")
	}
}

// recordingSink collects what a tool-capable turn produced.
type recordingSink struct {
	text  strings.Builder
	calls []frames.ToolCall
}

func (s *recordingSink) Text(text string) error {
	s.text.WriteString(text)
	return nil
}

func (s *recordingSink) Tool(c frames.ToolCall) error {
	s.calls = append(s.calls, c)
	return nil
}

// TestGenerateWithToolsAdvertisesTools checks the context's tools reach the
// request as functionDeclarations.
func TestGenerateWithToolsAdvertisesTools(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{APIKey: "k"})

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("weather?")
	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "Look up the weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})

	sink := &recordingSink{}
	if err := svc.GenerateWithTools(t.Context(), convo, sink); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}

	tools, ok := srv.body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want the one tools block", srv.body["tools"])
	}
	block, _ := tools[0].(map[string]any)
	decls, ok := block["functionDeclarations"].([]any)
	if !ok || len(decls) != 1 {
		t.Fatalf("functionDeclarations = %v, want the one declared tool", block["functionDeclarations"])
	}
	decl, _ := decls[0].(map[string]any)
	if decl["name"] != "get_weather" || decl["description"] != "Look up the weather" {
		t.Errorf("declaration = %v, want the advertised name and description", decl)
	}
}

// TestGenerateWithToolsOmitsToolsWhenNoneDeclared checks a context with no tools
// sends no tools block.
func TestGenerateWithToolsOmitsToolsWhenNoneDeclared(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{APIKey: "k"})

	sink := &recordingSink{}
	if err := svc.GenerateWithTools(t.Context(), frames.NewLLMContext(""), sink); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	if _, present := srv.body["tools"]; present {
		t.Errorf("tools = %v, want the block omitted", srv.body["tools"])
	}
	if sink.text.String() != "ok" {
		t.Errorf("text = %q, want the streamed part", sink.text.String())
	}
}

// TestGenerateWithToolsReportsCalls checks the streamed path end to end: text
// reaches the sink and each functionCall is reported with a synthetic id, Gemini
// supplying none of its own.
func TestGenerateWithToolsReportsCalls(t *testing.T) {
	srv := newGenServer(t, sse(
		textChunk("Let me check. "),
		`{"candidates":[{"content":{"parts":[`+
			`{"functionCall":{"name":"get_weather","args":{"location":"Paris"}}},`+
			`{"functionCall":{"name":"get_time"}}]}}]}`,
	))
	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{APIKey: "k"})

	sink := &recordingSink{}
	if err := svc.GenerateWithTools(t.Context(), frames.NewLLMContext(""), sink); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	if sink.text.String() != "Let me check. " {
		t.Errorf("text = %q, want the preamble", sink.text.String())
	}
	if len(sink.calls) != 2 {
		t.Fatalf("calls = %+v, want both function calls", sink.calls)
	}
	if sink.calls[0].ID != "call_0" || sink.calls[0].Name != "get_weather" ||
		string(sink.calls[0].Args) != `{"location":"Paris"}` {
		t.Errorf("first call = %+v (args %s)", sink.calls[0], sink.calls[0].Args)
	}
	// A call with no arguments still has to carry an object.
	if sink.calls[1].ID != "call_1" || sink.calls[1].Name != "get_time" ||
		string(sink.calls[1].Args) != "{}" {
		t.Errorf("second call = %+v (args %s)", sink.calls[1], sink.calls[1].Args)
	}
}

// TestGenerateWithToolsStatusError checks the tool-capable path reports a
// non-200 the way the plain one does.
func TestGenerateWithToolsStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte("permission denied"))
	}))
	defer srv.Close()

	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{APIKey: "k"})
	err := svc.GenerateWithTools(t.Context(), frames.NewLLMContext(""), &recordingSink{})
	if !errors.Is(err, errStatus) {
		t.Fatalf("GenerateWithTools error = %v, want errStatus", err)
	}
}

// TestRunInferenceAnswersOnce checks the one-shot path: the request addresses
// the non-streaming method, carries the instruction and bound this inference was
// given, and the answer is handed straight back.
func TestRunInferenceAnswersOnce(t *testing.T) {
	srv := newGenServer(t, `{"candidates":[{"content":{"parts":[{"text":"a short summary"}]}}]}`)
	svc := serviceFor(srv, &testShaper{base: srv.URL}, Config{APIKey: "k"})

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("what did we agree?")

	got, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{
		MaxTokens:         64,
		SystemInstruction: "Summarize the conversation.",
	})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got != "a short summary" {
		t.Errorf("answer = %q, want the candidate's text", got)
	}
	if strings.Contains(srv.path, "streamGenerateContent") {
		t.Errorf("path = %q, want the one-shot method", srv.path)
	}
	cfg, ok := srv.body["generationConfig"].(map[string]any)
	if !ok || cfg["maxOutputTokens"] != float64(64) {
		t.Errorf("generationConfig = %v, want the bound this inference was given", srv.body["generationConfig"])
	}
	sys, ok := srv.body["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("systemInstruction = %v, want the instruction this inference was given", srv.body["systemInstruction"])
	}
	parts, _ := sys["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("systemInstruction parts = %v, want one", parts)
	}
	part, _ := parts[0].(map[string]any)
	if part["text"] != "Summarize the conversation." {
		t.Errorf("systemInstruction = %v, want it to stand in for the conversation's own", part)
	}
}

// TestServiceRunsOneShotInferences checks the service satisfies the interface a
// summarizer or a judge asks for.
func TestServiceRunsOneShotInferences(t *testing.T) {
	var _ llm.Inferencer = (*Service)(nil)
}
