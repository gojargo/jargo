package responses

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

const (
	replyCreated   = `{"type":"response.created","response":{"id":"resp_1"}}`
	replyHi        = `{"type":"response.output_text.delta","delta":"hi"}`
	replyCompleted = `{"type":"response.completed","response":{"id":"resp_1","model":"gpt-4.1"}}`
)

// generateText runs one text turn and returns what the model said.
func generateText(t *testing.T, svc interface {
	Generate(context.Context, *frames.LLMContext, llm.Emit) error
},
) (string, error) {
	t.Helper()
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	var said strings.Builder
	err := svc.Generate(context.Background(), convo, func(s string) error { said.WriteString(s); return nil })
	return said.String(), err
}

// Ported from upstream (TestRetryOnTimeout::test_disabled_by_default).
func TestRetryOnTimeoutIsOffByDefault(t *testing.T) {
	var c Config
	if c.RetryOnTimeout || c.retryTimeout() != 0 {
		t.Error("a zero config retries on timeout")
	}
	c.RetryOnTimeout = true
	if got := c.retryTimeout(); got != 5*time.Second {
		t.Errorf("retry timeout = %s, want 5s by default", got)
	}
}

// Ported from upstream (test_silent_response_times_out and
// test_timeout_reconnects_and_retries_with_full_context). A response that says
// nothing in time is abandoned with its connection, and the turn is retried on
// a new one with the full context and no bound.
func TestWSSilentResponseIsRetriedOnANewConnection(t *testing.T) {
	srv := newFakeResponsesWS(t, [][]string{
		{"__pause__10s"},
		{replyCreated, replyHi, replyCompleted},
	})
	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL(), RetryOnTimeout: true, RetryTimeout: 50 * time.Millisecond})
	t.Cleanup(svc.disconnect)

	said, err := generateText(t, svc)
	if err != nil || said != "hi" {
		t.Fatalf("Generate = %q, %v, want hi", said, err)
	}
	if n := srv.dialed(); n != 2 {
		t.Errorf("dialed %d times, want a fresh connection for the retry", n)
	}
	if _, ok := srv.turn(1)["previous_response_id"]; ok {
		t.Error("the retry continued from a previous response, want the full context")
	}
}

// Ported from upstream (test_response_created_alone_does_not_close_the_window).
// An acknowledged but silent response is just what retrying is for.
func TestWSResponseCreatedAloneDoesNotCloseTheWindow(t *testing.T) {
	srv := newFakeResponsesWS(t, [][]string{
		{replyCreated, "__pause__10s"},
		{replyCreated, replyHi, replyCompleted},
	})
	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL(), RetryOnTimeout: true, RetryTimeout: 50 * time.Millisecond})
	t.Cleanup(svc.disconnect)

	if said, err := generateText(t, svc); err != nil || said != "hi" {
		t.Fatalf("Generate = %q, %v, want hi from the retry", said, err)
	}
}

// Ported from upstream (test_first_output_makes_the_wait_unbounded). Once
// output is on its way, a stall does not abandon the response.
func TestWSFirstOutputMakesTheWaitUnbounded(t *testing.T) {
	srv := newFakeResponsesWS(t, [][]string{{replyHi, "__pause__150ms", replyCompleted}})
	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL(), RetryOnTimeout: true, RetryTimeout: 50 * time.Millisecond})
	t.Cleanup(svc.disconnect)

	if said, err := generateText(t, svc); err != nil || said != "hi" {
		t.Fatalf("Generate = %q, %v, want hi", said, err)
	}
	if n := srv.dialed(); n != 1 {
		t.Errorf("dialed %d times, want the one response kept", n)
	}
}

// Ported from upstream (test_no_timeout_when_not_requested).
func TestWSNoTimeoutWhenNotRequested(t *testing.T) {
	srv := newFakeResponsesWS(t, [][]string{{"__pause__100ms", replyHi, replyCompleted}})
	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL(), RetryTimeout: 20 * time.Millisecond})
	t.Cleanup(svc.disconnect)

	if said, err := generateText(t, svc); err != nil || said != "hi" {
		t.Fatalf("Generate = %q, %v, want hi", said, err)
	}
}

// stallingHTTP answers each request after the stall its turn asks for.
func stallingHTTP(t *testing.T, stalls ...time.Duration) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Read whole, so the server notices a client that gives up on it.
		_, _ = io.Copy(io.Discard, r.Body)
		n := int(attempts.Add(1)) - 1
		if n < len(stalls) {
			select {
			case <-time.After(stalls[n]):
			case <-r.Context().Done():
				return
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: " + replyHi + "\n\ndata: " + replyCompleted + "\n\n"))
	}))
	t.Cleanup(srv.Close)
	return srv, &attempts
}

// Ported from upstream (TestHttpRetryOnTimeout::test_reissues_request_on_timeout).
func TestHTTPReissuesTheRequestOnTimeout(t *testing.T) {
	srv, attempts := stallingHTTP(t, 10*time.Second)
	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL, RetryOnTimeout: true, RetryTimeout: 50 * time.Millisecond})
	if said, err := generateText(t, svc); err != nil || said != "hi" {
		t.Fatalf("Generate = %q, %v, want hi", said, err)
	}
	if n := attempts.Load(); n != 2 {
		t.Errorf("attempts = %d, want 2", n)
	}
}

// Ported from upstream (test_retry_is_unbounded). The retry outlives the bound,
// or it would be pointless.
func TestHTTPRetryIsUnbounded(t *testing.T) {
	srv, attempts := stallingHTTP(t, 10*time.Second, 150*time.Millisecond)
	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL, RetryOnTimeout: true, RetryTimeout: 50 * time.Millisecond})
	if said, err := generateText(t, svc); err != nil || said != "hi" {
		t.Fatalf("Generate = %q, %v, want hi", said, err)
	}
	if n := attempts.Load(); n != 2 {
		t.Errorf("attempts = %d, want 2", n)
	}
}

// Ported from upstream (test_no_timeout_applied_when_disabled).
func TestHTTPNoTimeoutAppliedWhenDisabled(t *testing.T) {
	srv, attempts := stallingHTTP(t, 100*time.Millisecond)
	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL, RetryTimeout: 20 * time.Millisecond})
	if said, err := generateText(t, svc); err != nil || said != "hi" {
		t.Fatalf("Generate = %q, %v, want hi", said, err)
	}
	if n := attempts.Load(); n != 1 {
		t.Errorf("attempts = %d, want 1", n)
	}
}
