package gemini

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// thinkingOf returns the thinking configuration the request carried, and
// whether it carried one at all.
func thinkingOf(t *testing.T, srv *genServer) (map[string]any, bool) {
	t.Helper()
	cfg, _ := srv.body["generationConfig"].(map[string]any)
	thinking, ok := cfg[keyThinkingConfig].(map[string]any)
	return thinking, ok
}

// TestThinkingIsOffByDefaultOnFlash covers the low-latency default. A flash
// model is the one a voice pipeline reaches for, and a model that thinks emits
// nothing at all while it does, so the answer starts later than it needs to.
// The control differs by generation, which is why the default does too.
func TestThinkingIsOffByDefaultOnFlash(t *testing.T) {
	cases := []struct {
		model string
		want  map[string]any
	}{
		{"gemini-2.5-flash", map[string]any{"thinkingBudget": float64(0)}},
		{"gemini-2.5-flash-lite", map[string]any{"thinkingBudget": float64(0)}},
		{"gemini-3-flash", map[string]any{"thinkingLevel": "minimal"}},
		{"gemini-3.6-flash", map[string]any{"thinkingLevel": "minimal"}},
		// 3.7 Flash refuses "minimal", so the lowest it takes is used instead.
		{"gemini-3.7-flash", map[string]any{"thinkingLevel": "low"}},
		// A flash model this package has never heard of takes the fastest
		// setting, which is what an unlisted one is assumed to accept.
		{"gemini-3.9-flash", map[string]any{"thinkingLevel": "minimal"}},
	}
	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			srv := newGenServer(t, sse(textChunk("ok")))
			generate(t, srv, Config{APIKey: "k", Model: c.model})

			thinking, ok := thinkingOf(t, srv)
			if !ok {
				t.Fatalf("no thinking configuration was sent for %s", c.model)
			}
			for k, want := range c.want {
				if thinking[k] != want {
					t.Errorf("%s = %v, want %v", k, thinking[k], want)
				}
			}
		})
	}
}

// TestThinkingIsLeftAloneOnOtherModels covers the models the default does not
// apply to: a pro model is chosen for its reasoning, and an image model has no
// such control at all.
func TestThinkingIsLeftAloneOnOtherModels(t *testing.T) {
	// The image models include a flash one, which would otherwise be given a
	// thinking level by the rule above it.
	for _, model := range []string{"gemini-3-pro", "gemini-2.5-pro", "gemini-3.1-flash-image"} {
		t.Run(model, func(t *testing.T) {
			srv := newGenServer(t, sse(textChunk("ok")))
			generate(t, srv, Config{APIKey: "k", Model: model})

			if thinking, ok := thinkingOf(t, srv); ok {
				t.Errorf("thinking configuration = %v, want none for %s", thinking, model)
			}
		})
	}
}

// TestConfiguredThinkingWins covers a caller who has decided for themselves:
// what they set is sent, and the low-latency default does not overwrite it.
func TestConfiguredThinkingWins(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	budget := 512
	generate(t, srv, Config{
		APIKey:   "k",
		Model:    "gemini-2.5-flash",
		Thinking: &ThinkingConfig{Budget: &budget, IncludeThoughts: true},
	})

	thinking, ok := thinkingOf(t, srv)
	if !ok {
		t.Fatal("no thinking configuration was sent")
	}
	if thinking["thinkingBudget"] != float64(512) {
		t.Errorf("thinkingBudget = %v, want the configured 512", thinking["thinkingBudget"])
	}
	if thinking["includeThoughts"] != true {
		t.Errorf("includeThoughts = %v, want it asked for", thinking["includeThoughts"])
	}
}

// TestThinkingLevelReachesTheRequest covers the Gemini 3 control.
func TestThinkingLevelReachesTheRequest(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	generate(t, srv, Config{
		APIKey:   "k",
		Model:    "gemini-3-pro",
		Thinking: &ThinkingConfig{Level: "high"},
	})

	thinking, ok := thinkingOf(t, srv)
	if !ok {
		t.Fatal("no thinking configuration was sent")
	}
	if thinking["thinkingLevel"] != "high" {
		t.Errorf("thinkingLevel = %v, want the configured level", thinking["thinkingLevel"])
	}
}

// TestExtraThinkingWins covers a thinking configuration set through Extra, which
// is the escape hatch for a control this package does not model. The default
// must not overwrite it either.
func TestExtraThinkingWins(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	generate(t, srv, Config{
		APIKey: "k",
		Model:  "gemini-2.5-flash",
		Extra:  map[string]any{keyThinkingConfig: map[string]any{"thinkingBudget": 128}},
	})

	thinking, _ := thinkingOf(t, srv)
	if thinking["thinkingBudget"] != float64(128) {
		t.Errorf("thinkingBudget = %v, want the one set through Extra", thinking["thinkingBudget"])
	}
}

// TestSeedReachesTheRequest covers the sampling seed, which Gemini honors as
// best effort and which was previously dropped.
func TestSeedReachesTheRequest(t *testing.T) {
	srv := newGenServer(t, sse(textChunk("ok")))
	generate(t, srv, Config{APIKey: "k"})
	if cfg, _ := srv.body["generationConfig"].(map[string]any); cfg["seed"] != nil {
		t.Errorf("seed = %v, want none sent when unset", cfg["seed"])
	}

	seed := 42
	generate(t, srv, Config{APIKey: "k", Seed: &seed})
	cfg, _ := srv.body["generationConfig"].(map[string]any)
	if cfg["seed"] != float64(42) {
		t.Errorf("seed = %v, want the configured 42", cfg["seed"])
	}
}

// stallingServer answers with the given chunks and then holds the response open
// without closing it, which is how a stalled generation presents: nothing
// arrives and nothing ends.
type stallingServer struct {
	*httptest.Server
	requests atomic.Int64
	release  chan struct{}
}

// attempt is what the server does for one request: the chunks it writes, and
// whether it then holds the response open rather than ending it.
type attempt struct {
	chunks []string
	stall  bool
}

// newStallingServer serves each request the attempt of the same number, and
// anything beyond the list stalls with nothing sent at all.
func newStallingServer(t *testing.T, attempts ...attempt) *stallingServer {
	t.Helper()
	s := &stallingServer{release: make(chan struct{})}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := int(s.requests.Add(1)) - 1
		w.Header().Set("Content-Type", "text/event-stream")
		a := attempt{stall: true}
		if n < len(attempts) {
			a = attempts[n]
		}
		for _, c := range a.chunks {
			_, _ = w.Write([]byte("data: " + c + "\n\n"))
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		if !a.stall {
			return
		}
		select {
		case <-s.release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(func() {
		close(s.release)
		s.Close()
	})
	return s
}

// generateAgainst runs one generation against a bare URL, returning the text and
// the error.
func generateAgainst(t *testing.T, url string, cfg Config) (string, error) {
	t.Helper()
	svc := NewShapedLLM("GoogleLLM", &testShaper{base: url}, cfg)
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")

	var out strings.Builder
	err := svc.Generate(t.Context(), convo, func(text string) error {
		out.WriteString(text)
		return nil
	})
	return out.String(), err
}

// TestStalledStreamTimesOut covers a request the API accepts and then never
// answers. Nothing closes such a stream, so without a bound the turn stays open
// and the caller waits on a bot that will never speak.
func TestStalledStreamTimesOut(t *testing.T) {
	srv := newStallingServer(t, attempt{chunks: []string{textChunk("hello")}, stall: true})
	idle := 150 * time.Millisecond

	start := time.Now()
	text, err := generateAgainst(t, srv.URL, Config{APIKey: "k", StreamIdleTimeout: &idle})

	if err == nil {
		t.Fatal("a stalled stream ended without an error")
	}
	if !errorIsCompletionTimeout(err) {
		t.Errorf("error = %v, want a completion timeout", err)
	}
	// Whatever arrived before the stall is still the answer so far.
	if text != "hello" {
		t.Errorf("text = %q, want what arrived before the stall", text)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("the stall was waited out for %s, want the idle timeout to end it", elapsed)
	}
}

// TestStreamIdleTimeoutBoundsTheGapNotTheResponse covers a slow but healthy
// stream: each chunk re-arms the bound, so a long answer is never cut short.
func TestStreamIdleTimeoutBoundsTheGapNotTheResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for range 6 {
			_, _ = w.Write([]byte("data: " + textChunk("x") + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			time.Sleep(30 * time.Millisecond)
		}
	}))
	t.Cleanup(srv.Close)

	idle := 150 * time.Millisecond
	text, err := generateAgainst(t, srv.URL, Config{APIKey: "k", StreamIdleTimeout: &idle})
	if err != nil {
		t.Fatalf("a healthy stream failed: %v", err)
	}
	if text != "xxxxxx" {
		t.Errorf("text = %q, want every chunk of a stream that took longer than the bound", text)
	}
}

// TestNoIdleTimeoutWaitsIndefinitely covers the caller who asks for no bound at
// all, which is what a zero value says.
func TestNoIdleTimeoutWaitsIndefinitely(t *testing.T) {
	srv := newStallingServer(t, attempt{chunks: []string{textChunk("hello")}, stall: true})
	none := time.Duration(0)

	ctx, cancel := context.WithTimeout(t.Context(), 300*time.Millisecond)
	defer cancel()
	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{
		APIKey: "k", StreamIdleTimeout: &none,
	})
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	err := svc.Generate(ctx, convo, func(string) error { return nil })

	// The generation ended because the caller gave up, not because the service
	// bounded the wait.
	if err != nil && errorIsCompletionTimeout(err) {
		t.Errorf("error = %v, want no completion timeout when the bound is off", err)
	}
}

// TestRetryOnTimeoutReissuesTheRequest covers a request the API accepts and then
// never begins answering. Re-issuing costs a few seconds where waiting out the
// idle timeout costs the whole turn.
func TestRetryOnTimeoutReissuesTheRequest(t *testing.T) {
	srv := newStallingServer(t,
		attempt{stall: true},
		attempt{chunks: []string{textChunk("hello"), doneChunk()}},
	)

	text, err := generateAgainst(t, srv.URL, Config{
		APIKey:         "k",
		RetryOnTimeout: true,
		RetryTimeout:   150 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if text != "hello" {
		t.Errorf("text = %q, want the answer the second attempt gave", text)
	}
	if got := srv.requests.Load(); got != 2 {
		t.Errorf("requests = %d, want the stalled one and its retry", got)
	}
}

// TestRetryOnlyCoversTheFirstChunk covers the limit of the retry: a stream that
// has said something is already downstream, so re-issuing it would say
// everything twice.
func TestRetryOnlyCoversTheFirstChunk(t *testing.T) {
	srv := newStallingServer(t, attempt{chunks: []string{textChunk("hel")}, stall: true})
	idle := 150 * time.Millisecond

	text, err := generateAgainst(t, srv.URL, Config{
		APIKey:            "k",
		RetryOnTimeout:    true,
		RetryTimeout:      100 * time.Millisecond,
		StreamIdleTimeout: &idle,
	})
	if err == nil {
		t.Fatal("a stream that stalled mid-answer ended without an error")
	}
	if text != "hel" {
		t.Errorf("text = %q, want the part that arrived, said once", text)
	}
	if got := srv.requests.Load(); got != 1 {
		t.Errorf("requests = %d, want no retry once the answer had begun", got)
	}
}

// doneChunk is a final chunk carrying the reason the model stopped.
func doneChunk() string {
	return `{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}`
}

// errorIsCompletionTimeout reports whether err is the timeout the base reports
// to anything watching for one.
func errorIsCompletionTimeout(err error) bool {
	return err != nil && strings.Contains(err.Error(), llm.ErrCompletionTimeout.Error())
}

// TestFinishReasonIsReported covers the model stopping for a notable reason: the
// turn ends short with whatever arrived, and without this nothing says why.
func TestFinishReasonIsReported(t *testing.T) {
	logged := capturedWarnings(t)

	chunk := `{"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}]}`
	srv := newGenServer(t, sse(chunk))
	if text := generate(t, srv, Config{APIKey: "k"}); text != "partial" {
		t.Errorf("text = %q, want what did arrive to be passed on", text)
	}

	if !strings.Contains(logged(), "MAX_TOKENS") {
		t.Errorf("logs = %q, want the reason the model stopped", logged())
	}
}

// TestNormalFinishIsQuiet covers the ordinary end of a response, which is not
// worth a warning.
func TestNormalFinishIsQuiet(t *testing.T) {
	logged := capturedWarnings(t)

	srv := newGenServer(t, sse(textChunk("ok"), doneChunk()))
	generate(t, srv, Config{APIKey: "k"})

	if logged() != "" {
		t.Errorf("logs = %q, want nothing said about a normal stop", logged())
	}
}

// TestThinkingBudgetOnAThinkingLevelModelWarns covers the configuration that
// cannot be relied on: whether a Gemini 3 model honors a budget, ignores it, or
// refuses the request varies, and a refusal names no field.
func TestThinkingBudgetOnAThinkingLevelModelWarns(t *testing.T) {
	logged := capturedWarnings(t)

	budget := 256
	NewLLM(Config{APIKey: "k", Model: "gemini-3-pro", Thinking: &ThinkingConfig{Budget: &budget}})

	if !strings.Contains(logged(), "thinking budget") {
		t.Errorf("logs = %q, want the budget flagged on a model that reads a level", logged())
	}
}

// capturedWarnings swaps the default logger for one writing into a buffer and
// returns a reader for what was logged, restoring the logger when the test ends.
func capturedWarnings(t *testing.T) func() string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return buf.String
}

// TestRunInferenceAppliesTheThinkingDefault covers the inference a service runs
// off to the side of the pipeline, for a summary or a classification. It builds
// its request the same way, so it must get the same default: without it the
// summary of a call thinks where the call itself does not.
func TestRunInferenceAppliesTheThinkingDefault(t *testing.T) {
	srv := newGenServer(t, `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`)
	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{
		APIKey: "k", Model: "gemini-3.6-flash",
	})

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	if _, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{}); err != nil {
		t.Fatalf("RunInference: %v", err)
	}

	thinking, ok := thinkingOf(t, srv)
	if !ok || thinking["thinkingLevel"] != "minimal" {
		t.Errorf("thinking configuration = %v, want the model's low-latency default", thinking)
	}
}

// TestThinkingBudgetOnAGemini25ModelIsQuiet covers the model the budget is the
// right control for: nothing to warn about.
func TestThinkingBudgetOnAGemini25ModelIsQuiet(t *testing.T) {
	logged := capturedWarnings(t)

	budget := 256
	NewLLM(Config{APIKey: "k", Model: "gemini-2.5-flash", Thinking: &ThinkingConfig{Budget: &budget}})

	if strings.Contains(logged(), "thinking budget") {
		t.Errorf("logs = %q, want nothing said about a budget on the model that reads one", logged())
	}
}

// TestThinkingLevelOnAGemini3ModelIsQuiet covers the right control on a Gemini 3
// model, which is likewise not worth a warning.
func TestThinkingLevelOnAGemini3ModelIsQuiet(t *testing.T) {
	logged := capturedWarnings(t)

	NewLLM(Config{APIKey: "k", Model: "gemini-3-pro", Thinking: &ThinkingConfig{Level: "high"}})

	if strings.Contains(logged(), "thinking budget") {
		t.Errorf("logs = %q, want nothing said about a level", logged())
	}
}

// TestFirstChunkIsNotRetriedByDefault covers the opt-in: the wait for the first
// chunk covers whatever thinking the model does before it says anything, so
// re-issuing is a decision the caller makes rather than a default.
func TestFirstChunkIsNotRetriedByDefault(t *testing.T) {
	srv := newStallingServer(t, attempt{stall: true})
	idle := 150 * time.Millisecond

	if _, err := generateAgainst(t, srv.URL, Config{APIKey: "k", StreamIdleTimeout: &idle}); err == nil {
		t.Fatal("a stalled request ended without an error")
	}
	if got := srv.requests.Load(); got != 1 {
		t.Errorf("requests = %d, want the one attempt", got)
	}
}

// TestRequestIsReIssuedOnlyOnce covers a second stall: the turn ends rather than
// the service trying a third time.
func TestRequestIsReIssuedOnlyOnce(t *testing.T) {
	srv := newStallingServer(t, attempt{stall: true}, attempt{stall: true})
	idle := 150 * time.Millisecond

	_, err := generateAgainst(t, srv.URL, Config{
		APIKey:            "k",
		RetryOnTimeout:    true,
		RetryTimeout:      100 * time.Millisecond,
		StreamIdleTimeout: &idle,
	})
	if err == nil {
		t.Fatal("two stalled attempts ended without an error")
	}
	if !errorIsCompletionTimeout(err) {
		t.Errorf("error = %v, want a completion timeout", err)
	}
	if got := srv.requests.Load(); got != 2 {
		t.Errorf("requests = %d, want the stalled one and its one retry", got)
	}
}

// TestTheStalledRequestIsReleased covers what happens to the response the
// service walks away from: it is closed, rather than held open for the life of
// the call.
func TestTheStalledRequestIsReleased(t *testing.T) {
	gone := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: " + textChunk("hello") + "\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
		select {
		case gone <- struct{}{}:
		default:
		}
	}))
	t.Cleanup(srv.Close)

	idle := 150 * time.Millisecond
	if _, err := generateAgainst(t, srv.URL, Config{APIKey: "k", StreamIdleTimeout: &idle}); err == nil {
		t.Fatal("a stalled stream ended without an error")
	}
	select {
	case <-gone:
	case <-time.After(3 * time.Second):
		t.Error("the abandoned response was left open")
	}
}

// TestStreamBoundsAreOnByDefault pins the defaults: a stream is bounded unless a
// caller opts out, and re-issuing is opt-in.
func TestStreamBoundsAreOnByDefault(t *testing.T) {
	cfg := Config{APIKey: "k"}
	if got := cfg.streamIdleTimeout(); got != 20*time.Second {
		t.Errorf("stream idle timeout = %v, want 20s", got)
	}
	if got := cfg.retryTimeout(); got != 5*time.Second {
		t.Errorf("retry timeout = %v, want 5s", got)
	}
	if cfg.RetryOnTimeout {
		t.Error("re-issuing a stalled request is on by default, want it opt-in")
	}
}

// TestEveryNotableFinishReasonIsReported covers the reasons a response can be
// curtailed for. Each leaves the turn short of what the model would have said,
// and the log is the only place that says so.
func TestEveryNotableFinishReasonIsReported(t *testing.T) {
	for _, reason := range []string{
		"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "MALFORMED_FUNCTION_CALL", "OTHER", "MAX_TOKENS",
	} {
		t.Run(reason, func(t *testing.T) {
			logged := capturedWarnings(t)
			chunk := `{"candidates":[{"content":{"parts":[]},"finishReason":"` + reason + `"}]}`
			srv := newGenServer(t, sse(chunk))
			generate(t, srv, Config{APIKey: "k"})

			if !strings.Contains(logged(), reason) {
				t.Errorf("logs = %q, want the reason named", logged())
			}
		})
	}
}

// TestChunksWithoutAFinishReasonAreQuiet covers the ordinary streamed chunk,
// which carries no reason at all.
func TestChunksWithoutAFinishReasonAreQuiet(t *testing.T) {
	logged := capturedWarnings(t)

	srv := newGenServer(t, sse(textChunk("Partial"), textChunk(" text.")))
	if text := generate(t, srv, Config{APIKey: "k"}); text != "Partial text." {
		t.Errorf("text = %q, want both chunks", text)
	}
	if logged() != "" {
		t.Errorf("logs = %q, want nothing said for chunks that carry no reason", logged())
	}
}
