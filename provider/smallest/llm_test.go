package smallest_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/smallest"
	"github.com/gojargo/jargo/service/settings"
)

// smallestLLM builds the service from the shared OpenAI-compatible config.
func smallestLLM(cfg chat.LLMConfig) *chat.LLMService {
	return smallest.NewLLM(smallest.LLMConfig{LLMConfig: cfg})
}

// Ported from upstream (test_default_model_is_electron and
// test_model_can_be_overridden): Electron is the default model, a caller can
// pick another, and the service is an ordinary OpenAI-compatible one.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "SmallestLLM", "electron", smallestLLM)
}

// capturedRequest is one request the stand-in endpoint received.
type capturedRequest struct {
	url    string
	header http.Header
	body   map[string]any
}

// llmServer stands in for the chat completions endpoint, recording each
// request and answering it with an empty stream.
type llmServer struct {
	*httptest.Server
	mu       sync.Mutex
	requests []capturedRequest
}

func newLLMServer(t *testing.T) *llmServer {
	t.Helper()
	s := &llmServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.record(t, r)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *llmServer) record(t *testing.T, r *http.Request) {
	t.Helper()
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Errorf("decoding request body: %v", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, capturedRequest{url: r.URL.String(), header: r.Header.Clone(), body: body})
}

// last returns the one request a test expects to have been sent.
func (s *llmServer) last(t *testing.T) capturedRequest {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.requests) != 1 {
		t.Fatalf("requests = %d, want one", len(s.requests))
	}
	return s.requests[0]
}

// nopSink takes a turn's output and drops it.
type nopSink struct{}

func (nopSink) Text(string) error          { return nil }
func (nopSink) Tool(frames.ToolCall) error { return nil }

// generate runs one turn of a service built from cfg against srv.
func generate(t *testing.T, srv *llmServer, cfg smallest.LLMConfig) capturedRequest {
	t.Helper()
	cfg.APIKey = "test-key"
	cfg.BaseURL = srv.URL
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	if err := smallest.NewLLM(cfg).Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return srv.last(t)
}

// roundTripFunc is an http.RoundTripper made of a function.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// Ported from upstream (test_base_url_defaults_to_waves_api). The request is
// caught before it leaves the process, so no real host is contacted.
func TestBaseURLDefaultsToWavesAPI(t *testing.T) {
	var got string
	prev := http.DefaultTransport
	http.DefaultTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		got = r.URL.String()
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader("data: [DONE]\n\n")),
			Request:    r,
		}, nil
	})
	t.Cleanup(func() { http.DefaultTransport = prev })

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	svc := smallest.NewLLM(smallest.LLMConfig{APIKey: "test-key"})
	if err := svc.Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if want := "https://api.smallest.ai/waves/v1/chat/completions"; got != want {
		t.Errorf("request URL = %q, want %q", got, want)
	}
}

// Ported from upstream (test_temperature_reaches_the_request): the standard
// OpenAI-compatible parameters pass through untouched.
func TestTemperatureReachesTheRequest(t *testing.T) {
	temperature := 0.3
	req := generate(t, newLLMServer(t), smallest.LLMConfig{Temperature: &temperature})
	if req.body["temperature"] != 0.3 {
		t.Errorf("temperature = %v, want 0.3", req.body["temperature"])
	}
}

// Ported from upstream (test_unset_params_are_omitted): a parameter nobody set
// stays off the wire, as does the prompt cache key.
func TestUnsetParamsAreOmitted(t *testing.T) {
	req := generate(t, newLLMServer(t), smallest.LLMConfig{})
	for _, key := range []string{"max_tokens", "frequency_penalty", "prompt_cache_key"} {
		if v, present := req.body[key]; present {
			t.Errorf("%s = %v, want it omitted", key, v)
		}
	}
}

// Ported from upstream (test_developer_messages_are_converted_to_user): Electron
// does not support the developer role, so a developer message is sent as a user
// message.
func TestDeveloperMessagesAreConvertedToUser(t *testing.T) {
	srv := newLLMServer(t)
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Be terse."})
	convo.AddUserMessage("Hello")
	svc := smallest.NewLLM(smallest.LLMConfig{APIKey: "k", BaseURL: srv.URL})
	if err := svc.Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	messages, _ := srv.last(t).body["messages"].([]any)
	if len(messages) == 0 {
		t.Fatal("the request carried no messages")
	}
	first, _ := messages[0].(map[string]any)
	if first["role"] != "user" {
		t.Errorf("first message role = %v, want user", first["role"])
	}
}

// The prompt cache key travels in the request body, and the caller's own extra
// fields win over it, as they do over every other request parameter.
func TestPromptCacheKeyReachesTheRequest(t *testing.T) {
	req := generate(t, newLLMServer(t), smallest.LLMConfig{PromptCacheKey: "conversation-1"})
	if req.body["prompt_cache_key"] != "conversation-1" {
		t.Errorf("prompt_cache_key = %v, want conversation-1", req.body["prompt_cache_key"])
	}

	req = generate(t, newLLMServer(t), smallest.LLMConfig{
		Extra:          map[string]any{"prompt_cache_key": "from-extra"},
		PromptCacheKey: "conversation-1",
	})
	if req.body["prompt_cache_key"] != "from-extra" {
		t.Errorf("prompt_cache_key = %v, want the caller's extra", req.body["prompt_cache_key"])
	}
}

// The prompt cache key can be set while the pipeline runs.
func TestPromptCacheKeyCanBeUpdated(t *testing.T) {
	srv := newLLMServer(t)
	svc := smallest.NewLLM(smallest.LLMConfig{APIKey: "k", BaseURL: srv.URL})
	delta := &smallest.LLMSettings{PromptCacheKey: settings.Set("conversation-2")}
	update := frames.NewLLMUpdateSettingsFrame(delta)
	if err := svc.Base.ProcessFrame(t.Context(), update, processor.Downstream); err != nil {
		t.Fatalf("update: %v", err)
	}
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	if err := svc.GenerateWithTools(t.Context(), convo, nopSink{}); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	if got := srv.last(t).body["prompt_cache_key"]; got != "conversation-2" {
		t.Errorf("prompt_cache_key = %v, want the updated key", got)
	}
}

// Smallest's source identification header is always sent, over the caller's
// own headers, which are kept.
func TestSourceHeaderIsSent(t *testing.T) {
	req := generate(t, newLLMServer(t), smallest.LLMConfig{
		DefaultHeaders: map[string]string{"X-Source": "caller", "X-Custom": "kept"},
	})
	if got := req.header.Get("X-Source"); got != "jargo" {
		t.Errorf("X-Source = %q, want jargo", got)
	}
	if got := req.header.Get("X-Custom"); got != "kept" {
		t.Errorf("X-Custom = %q, want the caller's header kept", got)
	}
}

// Ported from upstream (the Smallest cases of test_missing_api_key_raises and
// test_explicit_api_key_is_used): the service needs an API key, and sends the
// one it is given.
func TestAPIKeyIsRequired(t *testing.T) {
	if err := (smallest.LLMConfig{}).Validate(); err == nil || !strings.Contains(err.Error(), "requires an api_key") {
		t.Errorf("Validate() = %v, want the missing key refused", err)
	}
	if err := (smallest.LLMConfig{APIKey: "provider-key"}).Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
	req := generate(t, newLLMServer(t), smallest.LLMConfig{})
	if got := req.header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the configured key", got)
	}
}
