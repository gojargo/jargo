package sarvam_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/sarvam"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/utils/events"
)

// sarvamRequest is one request the stand-in endpoint received.
type sarvamRequest struct {
	header http.Header
	body   map[string]any
}

// sarvamServer stands in for Sarvam's chat-completions endpoint: it records
// every request, and answers a streamed one with an empty stream and a one-shot
// one with answer, unless it was told to fail with status.
type sarvamServer struct {
	*httptest.Server
	mu       sync.Mutex
	requests []sarvamRequest
	status   int
	errBody  string
	answer   string
}

func newSarvamServer(t *testing.T) *sarvamServer {
	t.Helper()
	s := &sarvamServer{answer: "ok"}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		s.mu.Lock()
		s.requests = append(s.requests, sarvamRequest{header: r.Header.Clone(), body: body})
		status, errBody, answer := s.status, s.errBody, s.answer
		s.mu.Unlock()
		if status != 0 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(errBody))
			return
		}
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":` + quote(answer) + `}}]}`))
	}))
	t.Cleanup(s.Close)
	return s
}

// failWith makes every later request fail with status and body.
func (s *sarvamServer) failWith(status int, body string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status, s.errBody = status, body
}

// sent returns the requests received so far.
func (s *sarvamServer) sent() []sarvamRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]sarvamRequest(nil), s.requests...)
}

// last returns the one request a test expects to have been sent.
func (s *sarvamServer) last(t *testing.T) sarvamRequest {
	t.Helper()
	got := s.sent()
	if len(got) != 1 {
		t.Fatalf("requests = %d, want one", len(got))
	}
	return got[0]
}

func quote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// newService builds a service for cfg that talks to srv.
func newService(srv *sarvamServer, cfg sarvam.LLMConfig) *chat.LLMService {
	cfg.APIKey = "test-key"
	cfg.BaseURL = srv.URL
	return sarvam.NewLLM(cfg)
}

// withModel is a config for model and nothing else.
func withModel(model string) sarvam.LLMConfig {
	return sarvam.LLMConfig{Model: model}
}

// hello is a conversation of one user message.
func hello() *frames.LLMContext {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	return convo
}

// withImage is a conversation whose one user message carries an image.
func withImage() *frames.LLMContext {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.NewLLMSpecificMessage("openai", openai.Message{
		Role: openai.RoleUser,
		ContentParts: []openai.ContentPart{
			openai.TextPart("Describe this image."),
			{Type: "image_url"},
		},
	}))
	return convo
}

// weatherTool is a tool a conversation can advertise.
func weatherTool() frames.Tool {
	return frames.Tool{
		Name:        "get_weather",
		Description: "Get weather for a city",
		Parameters: json.RawMessage(
			`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
}

// nopSink takes a turn's output and drops it.
type nopSink struct{}

func (nopSink) Text(string) error               { return nil }
func (nopSink) Tool(call frames.ToolCall) error { return nil }

// requestFor runs one streamed turn of convo against a service built from cfg
// and returns the request it sent.
func requestFor(t *testing.T, cfg sarvam.LLMConfig, convo *frames.LLMContext) sarvamRequest {
	t.Helper()
	srv := newSarvamServer(t)
	if err := newService(srv, cfg).GenerateWithTools(t.Context(), convo, nopSink{}); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}
	return srv.last(t)
}

// reportedErrors collects the error frames svc reports.
func reportedErrors(svc *chat.LLMService) func() []*frames.ErrorFrame {
	var mu sync.Mutex
	var got []*frames.ErrorFrame
	svc.Events().Add(processor.EventError, func(_ context.Context, _ any, args ...any) {
		for _, a := range args {
			if ef, ok := a.(*frames.ErrorFrame); ok {
				mu.Lock()
				got = append(got, ef)
				mu.Unlock()
			}
		}
	})
	return func() []*frames.ErrorFrame {
		mu.Lock()
		defer mu.Unlock()
		return append([]*frames.ErrorFrame(nil), got...)
	}
}

// runTurn runs one turn of convo through svc the way the pipeline drives it,
// and returns the errors it reported.
func runTurn(t *testing.T, svc *chat.LLMService, convo *frames.LLMContext) []*frames.ErrorFrame {
	t.Helper()
	errs := reportedErrors(svc)
	done := make(chan struct{})
	var once sync.Once
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
			once.Do(func() { close(done) })
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- worker.Run(context.Background()) }()
	t.Cleanup(func() {
		worker.StopWhenDone()
		<-runDone
	})

	worker.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the response never ended")
	}
	return errs()
}

// refusedTurn runs one turn that is expected to be refused before it is sent,
// and checks it was: one error naming want, and nothing sent.
func refusedTurn(t *testing.T, cfg sarvam.LLMConfig, convo *frames.LLMContext, want string) {
	t.Helper()
	srv := newSarvamServer(t)
	reported := runTurn(t, newService(srv, cfg), convo)

	if len(reported) != 1 || !strings.Contains(reported[0].Error, want) {
		t.Errorf("errors = %v, want one saying %q", reported, want)
	}
	if n := len(srv.sent()); n != 0 {
		t.Errorf("requests = %d, want the refused turn not sent", n)
	}
}

// TestLLMConfigRejectsAnUnsupportedModel checks a model Sarvam does not serve
// is refused before anything is sent to it.
func TestLLMConfigRejectsAnUnsupportedModel(t *testing.T) {
	cfg := sarvam.LLMConfig{APIKey: "k", Model: "sarvam-100b"}
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "unsupported LLM model") {
		t.Errorf("Validate() = %v, want the model refused", err)
	}
}

// TestLLMConfigAcceptsEverySupportedModel checks every model Sarvam serves is
// accepted, and is the one the service runs.
func TestLLMConfigAcceptsEverySupportedModel(t *testing.T) {
	for _, model := range []string{
		"deepseekv4.1-flash", "gemma4", "glm5.3", "sarvam-105b", "sarvam-105b-conversations",
	} {
		t.Run(model, func(t *testing.T) {
			cfg := sarvam.LLMConfig{APIKey: "k", Model: model}
			if err := cfg.Validate(); err != nil {
				t.Errorf("Validate() = %v, want the model accepted", err)
			}
			if got := sarvam.NewLLM(cfg).Model(); got != model {
				t.Errorf("model = %q, want %q", got, model)
			}
		})
	}
}

// TestAnExplicitBaseURLOverridesTheModelDefault checks a configured base URL is
// used whatever version the model is served under by default.
func TestAnExplicitBaseURLOverridesTheModelDefault(t *testing.T) {
	req := requestFor(t, withModel("sarvam-105b-conversations"), hello())
	if req.body["model"] != "sarvam-105b-conversations" {
		t.Errorf("model = %v, want the request sent to the configured base", req.body["model"])
	}
}

// TestTheSubscriptionKeyHeaderIsSent checks the API key is sent as Sarvam's
// api-subscription-key header as well as a Bearer token.
func TestTheSubscriptionKeyHeaderIsSent(t *testing.T) {
	req := requestFor(t, withModel("sarvam-105b"), hello())
	if got := req.header.Get("api-subscription-key"); got != "test-key" {
		t.Errorf("api-subscription-key = %q, want the API key", got)
	}
	if got := req.header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", got)
	}
}

// TestTheCallersHeadersAreMergedWithTheSubscriptionKey checks a caller's own
// headers are sent too, and that the subscription key Sarvam authorizes by
// wins over one of the same name.
func TestTheCallersHeadersAreMergedWithTheSubscriptionKey(t *testing.T) {
	cfg := withModel("sarvam-105b")
	cfg.DefaultHeaders = map[string]string{"X-Test-Header": "enabled", "api-subscription-key": "stale"}
	req := requestFor(t, cfg, hello())
	if got := req.header.Get("X-Test-Header"); got != "enabled" {
		t.Errorf("X-Test-Header = %q, want the caller's header", got)
	}
	if got := req.header.Get("api-subscription-key"); got != "test-key" {
		t.Errorf("api-subscription-key = %q, want the API key", got)
	}
	if got := cfg.DefaultHeaders["api-subscription-key"]; got != "stale" {
		t.Errorf("the caller's headers were rewritten: %q", got)
	}
}

// TestTheReasoningEffortIsSent checks each reasoning effort reaches a model
// that takes one.
func TestTheReasoningEffortIsSent(t *testing.T) {
	for _, effort := range []string{"low", "medium", "high"} {
		t.Run(effort, func(t *testing.T) {
			cfg := withModel("sarvam-105b")
			cfg.ReasoningEffort = effort
			if got := requestFor(t, cfg, hello()).body["reasoning_effort"]; got != effort {
				t.Errorf("reasoning_effort = %v, want %q", got, effort)
			}
		})
	}
}

// TestSarvamsOwnFieldsAreLeftOffWhenUnset checks no wiki grounding or reasoning
// effort is sent unless one is configured.
func TestSarvamsOwnFieldsAreLeftOffWhenUnset(t *testing.T) {
	body := requestFor(t, withModel("sarvam-105b"), hello()).body
	for _, field := range []string{"wiki_grounding", "reasoning_effort"} {
		if _, ok := body[field]; ok {
			t.Errorf("body carries %s, which nothing configured: %v", field, body)
		}
	}
}

// TestWikiGroundingIsSent checks the wiki grounding toggle reaches a model that
// takes it.
func TestWikiGroundingIsSent(t *testing.T) {
	cfg := withModel("sarvam-105b")
	cfg.WikiGrounding = new(true)
	if got := requestFor(t, cfg, hello()).body["wiki_grounding"]; got != true {
		t.Errorf("wiki_grounding = %v, want true", got)
	}
}

// TestFieldsSarvamDoesNotTakeAreLeftOff checks the request carries none of
// OpenAI's fields Sarvam does not take, even when the config sets them, next
// to Sarvam's own.
func TestFieldsSarvamDoesNotTakeAreLeftOff(t *testing.T) {
	limit := 300
	cfg := withModel("sarvam-105b")
	cfg.Temperature = new(0.7)
	cfg.MaxTokens = 128
	cfg.MaxCompletionTokens = &limit
	cfg.ServiceTier = "flex"
	cfg.WikiGrounding = new(false)
	cfg.ReasoningEffort = "medium"
	body := requestFor(t, cfg, hello()).body

	for _, field := range []string{"stream_options", "max_completion_tokens", "service_tier"} {
		if _, ok := body[field]; ok {
			t.Errorf("body carries %s: %v", field, body)
		}
	}
	if body["wiki_grounding"] != false {
		t.Errorf("wiki_grounding = %v, want false", body["wiki_grounding"])
	}
	if body["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort = %v, want medium", body["reasoning_effort"])
	}
}

// TestTheCoreAndExtraFieldsAreForwarded checks the sampling controls and a
// caller's extra fields still reach Sarvam.
func TestTheCoreAndExtraFieldsAreForwarded(t *testing.T) {
	cfg := withModel("sarvam-105b")
	cfg.Temperature = new(0.9)
	cfg.TopP = new(0.8)
	cfg.FrequencyPenalty = new(0.1)
	cfg.PresencePenalty = new(0.2)
	cfg.Seed = new(11)
	cfg.MaxTokens = 222
	cfg.Extra = map[string]any{"n": 2}
	body := requestFor(t, cfg, hello()).body

	want := map[string]any{
		"temperature": 0.9, "top_p": 0.8, "frequency_penalty": 0.1, "presence_penalty": 0.2,
		"seed": float64(11), "max_tokens": float64(222), "n": float64(2),
	}
	for field, value := range want {
		if body[field] != value {
			t.Errorf("%s = %v, want %v", field, body[field], value)
		}
	}
}

// TestSarvamsFieldsAreMergedWithTheCallersExtra checks the wiki grounding toggle
// and a caller's own extra field are both sent.
func TestSarvamsFieldsAreMergedWithTheCallersExtra(t *testing.T) {
	cfg := withModel("sarvam-105b")
	cfg.WikiGrounding = new(true)
	cfg.Extra = map[string]any{"user_field": 1}
	body := requestFor(t, cfg, hello()).body

	if body["user_field"] != float64(1) || body["wiki_grounding"] != true {
		t.Errorf("body = %v, want both the caller's field and wiki_grounding", body)
	}
}

// TestAToolChoiceWithToolsIsAllowed checks a tool_choice sent with tools goes
// through as asked.
func TestAToolChoiceWithToolsIsAllowed(t *testing.T) {
	for _, model := range []string{"sarvam-105b", "sarvam-105b-conversations"} {
		t.Run(model, func(t *testing.T) {
			convo := hello()
			convo.SetTools([]frames.Tool{weatherTool()})
			convo.SetToolChoice(frames.ToolChoiceRequired)
			body := requestFor(t, withModel(model), convo).body

			if body["tool_choice"] != "required" {
				t.Errorf("tool_choice = %v, want required", body["tool_choice"])
			}
			tools, _ := body["tools"].([]any)
			if len(tools) != 1 {
				t.Fatalf("tools = %v, want the one advertised", body["tools"])
			}
			fn, _ := tools[0].(map[string]any)["function"].(map[string]any)
			if fn["name"] != "get_weather" {
				t.Errorf("tool = %v, want get_weather", tools[0])
			}
		})
	}
}

// TestAnImageIsRefusedOnAModelThatCannotSee checks an image sent to a model
// without vision is refused before it is sent, with an error saying so.
func TestAnImageIsRefusedOnAModelThatCannotSee(t *testing.T) {
	for _, model := range []string{"sarvam-105b", "sarvam-105b-conversations", "glm5.3"} {
		t.Run(model, func(t *testing.T) {
			refusedTurn(t, withModel(model), withImage(), "does not support image input")
		})
	}
}

// TestAnImageIsAcceptedOnGemma4 checks the vision model is sent the image.
func TestAnImageIsAcceptedOnGemma4(t *testing.T) {
	if got := requestFor(t, withModel("gemma4"), withImage()).body["model"]; got != "gemma4" {
		t.Errorf("model = %v, want the image sent to gemma4", got)
	}
}

// TestAReasoningEffortIsRefusedOnTheConversationsModel checks a reasoning
// effort set on the model that does not take one is refused rather than
// dropped without a word.
func TestAReasoningEffortIsRefusedOnTheConversationsModel(t *testing.T) {
	cfg := withModel("sarvam-105b-conversations")
	cfg.ReasoningEffort = "high"
	refusedTurn(t, cfg, hello(), "does not support reasoning_effort")
}

// TestWikiGroundingIsRefusedOnAModelThatDoesNotTakeIt checks wiki grounding set
// on a model without it is refused rather than dropped without a word.
func TestWikiGroundingIsRefusedOnAModelThatDoesNotTakeIt(t *testing.T) {
	for _, model := range []string{"sarvam-105b-conversations", "glm5.3"} {
		t.Run(model, func(t *testing.T) {
			cfg := withModel(model)
			cfg.WikiGrounding = new(true)
			refusedTurn(t, cfg, hello(), "does not support wiki_grounding")
		})
	}
}

// TestGLM53TakesAReasoningEffort checks a reasoning model other than Sarvam's
// own is sent its reasoning effort.
func TestGLM53TakesAReasoningEffort(t *testing.T) {
	cfg := withModel("glm5.3")
	cfg.ReasoningEffort = "high"
	if got := requestFor(t, cfg, hello()).body["reasoning_effort"]; got != "high" {
		t.Errorf("reasoning_effort = %v, want high", got)
	}
}

// TestAnUpdateAppliesSarvamsOwnSettings checks a settings update can change
// Sarvam's own fields while the pipeline runs, and that the next request
// carries them.
func TestAnUpdateAppliesSarvamsOwnSettings(t *testing.T) {
	var logs bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelInfo})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	srv := newSarvamServer(t)
	svc := newService(srv, withModel("sarvam-105b"))
	delta := &sarvam.LLMSettings{}
	delta.WikiGrounding = settings.Set(true)
	delta.ReasoningEffort = settings.Set("low")
	update := frames.NewLLMUpdateSettingsFrame(delta)
	if err := svc.Base.ProcessFrame(t.Context(), update, processor.Downstream); err != nil {
		t.Fatalf("update: %v", err)
	}
	if err := svc.GenerateWithTools(t.Context(), hello(), nopSink{}); err != nil {
		t.Fatalf("GenerateWithTools: %v", err)
	}

	if !strings.Contains(logs.String(), "wiki_grounding") || !strings.Contains(logs.String(), "reasoning_effort") {
		t.Errorf("logs = %q, want both fields reported as changed", logs.String())
	}
	body := srv.last(t).body
	if body["wiki_grounding"] != true || body["reasoning_effort"] != "low" {
		t.Errorf("body = %v, want the updated wiki_grounding and reasoning_effort", body)
	}
}

// TestRunInferenceIsShapedForSarvam checks the one-shot path is shaped as a
// streamed turn is: not streamed, without the fields Sarvam does not take, and
// with Sarvam's own.
func TestRunInferenceIsShapedForSarvam(t *testing.T) {
	srv := newSarvamServer(t)
	srv.answer = "Hello! How can I help you today?"
	cfg := withModel("sarvam-105b")
	cfg.Temperature = new(0.7)
	cfg.MaxTokens = 100
	cfg.FrequencyPenalty = new(0.5)
	cfg.Seed = new(42)
	cfg.WikiGrounding = new(true)
	cfg.ReasoningEffort = "high"

	got, err := newService(srv, cfg).RunInference(t.Context(), hello(), llm.InferenceOptions{})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got != "Hello! How can I help you today?" {
		t.Errorf("answer = %q, want the completion's content", got)
	}
	body := srv.last(t).body
	if body["stream"] != false {
		t.Errorf("stream = %v, want false", body["stream"])
	}
	for _, field := range []string{"stream_options", "max_completion_tokens", "service_tier"} {
		if _, ok := body[field]; ok {
			t.Errorf("body carries %s: %v", field, body)
		}
	}
	if body["wiki_grounding"] != true || body["reasoning_effort"] != "high" {
		t.Errorf("body = %v, want wiki_grounding and reasoning_effort", body)
	}
}

// TestRunInferenceOverridesTheBoundInMaxTokens checks an inference's own bound
// goes in max_tokens, the field Sarvam reads.
func TestRunInferenceOverridesTheBoundInMaxTokens(t *testing.T) {
	srv := newSarvamServer(t)
	cfg := withModel("sarvam-105b")
	cfg.MaxTokens = 100
	if _, err := newService(srv, cfg).RunInference(t.Context(), hello(), llm.InferenceOptions{MaxTokens: 23}); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	body := srv.last(t).body
	if body["max_tokens"] != float64(23) {
		t.Errorf("max_tokens = %v, want the inference's 23", body["max_tokens"])
	}
	if _, ok := body["max_completion_tokens"]; ok {
		t.Errorf("body carries max_completion_tokens: %v", body)
	}
}

// TestRunInferenceForwardsTheSystemInstruction checks an inference's own
// instruction is sent.
func TestRunInferenceForwardsTheSystemInstruction(t *testing.T) {
	srv := newSarvamServer(t)
	opts := llm.InferenceOptions{SystemInstruction: "You are a concise assistant."}
	if _, err := newService(srv, withModel("sarvam-105b")).RunInference(t.Context(), hello(), opts); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	msgs, _ := srv.last(t).body["messages"].([]any)
	first, _ := msgs[0].(map[string]any)
	if first["content"] != "You are a concise assistant." {
		t.Errorf("messages = %v, want the inference's instruction first", msgs)
	}
}

// TestRunInferenceReportsARefusedRequest checks a one-shot inference the model
// cannot serve is reported as an error and answers nothing, without being
// sent.
func TestRunInferenceReportsARefusedRequest(t *testing.T) {
	srv := newSarvamServer(t)
	cfg := withModel("sarvam-105b-conversations")
	cfg.ReasoningEffort = "high"
	svc := newService(srv, cfg)
	reported := reportedErrors(svc)

	got, err := svc.RunInference(t.Context(), hello(), llm.InferenceOptions{})
	if err != nil || got != "" {
		t.Errorf("RunInference = (%q, %v), want no answer and no error", got, err)
	}
	if errs := reported(); len(errs) != 1 || !strings.Contains(errs[0].Error, "does not support reasoning_effort") {
		t.Errorf("errors = %v, want the refusal reported", errs)
	}
	if n := len(srv.sent()); n != 0 {
		t.Errorf("requests = %d, want the refused inference not sent", n)
	}
}

// TestRunInferenceTimeoutsAreNotWrapped checks an inference that runs out of
// time fails as having run out of time.
func TestRunInferenceTimeoutsAreNotWrapped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	svc := sarvam.NewLLM(sarvam.LLMConfig{APIKey: "k", BaseURL: srv.URL})
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	if _, err := svc.RunInference(ctx, hello(), llm.InferenceOptions{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("RunInference error = %v, want the deadline", err)
	}
}

// TestTheServersErrorIsSurfaced checks what Sarvam said when it refused a
// request reaches the caller, on the one-shot path, on a streamed turn, and
// through the pipeline.
func TestTheServersErrorIsSurfaced(t *testing.T) {
	refused := func(t *testing.T, message string) *chat.LLMService {
		t.Helper()
		srv := newSarvamServer(t)
		srv.failWith(http.StatusBadRequest, `{"error":{"message":`+quote(message)+`}}`)
		return newService(srv, withModel("sarvam-105b"))
	}

	t.Run("one-shot", func(t *testing.T) {
		_, err := refused(t, "model is not available for this account").
			RunInference(t.Context(), hello(), llm.InferenceOptions{})
		if err == nil || !strings.Contains(err.Error(), "model is not available for this account") {
			t.Errorf("RunInference error = %v, want Sarvam's message", err)
		}
	})

	t.Run("streamed", func(t *testing.T) {
		err := refused(t, "invalid request format").GenerateWithTools(t.Context(), hello(), nopSink{})
		if err == nil || !strings.Contains(err.Error(), "invalid request format") {
			t.Errorf("GenerateWithTools error = %v, want Sarvam's message", err)
		}
	})

	t.Run("through the pipeline", func(t *testing.T) {
		reported := runTurn(t, refused(t, "tool schema is invalid"), hello())
		if len(reported) != 1 || reported[0].Err == nil ||
			!strings.Contains(reported[0].Err.Error(), "tool schema is invalid") {
			t.Errorf("errors = %v, want one carrying Sarvam's message", reported)
		}
	})
}

// TestTheStreamIsClosedWhenTheTurnIsCanceled checks a turn canceled mid-stream,
// as an interruption cancels it, closes the response rather than leaving the
// connection open.
func TestTheStreamIsClosedWhenTheTurnIsCanceled(t *testing.T) {
	closed := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"Hello"}}]}` + "\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
		close(closed)
	}))
	t.Cleanup(srv.Close)

	svc := sarvam.NewLLM(sarvam.LLMConfig{APIKey: "k", BaseURL: srv.URL})
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	if err := svc.Generate(ctx, hello(), func(string) error { cancel(); return nil }); err == nil {
		t.Error("a canceled turn ended without an error")
	}
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Error("the stream of a canceled turn was left open")
	}
}
