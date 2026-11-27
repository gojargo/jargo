package anthropic

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// betaRecorder stands in for the Messages API, recording the beta header of
// every request it receives.
type betaRecorder struct {
	mu    sync.Mutex
	betas [][]string
}

func (b *betaRecorder) server(t *testing.T, stream bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.mu.Lock()
		b.betas = append(b.betas, r.Header.Values("anthropic-beta"))
		b.mu.Unlock()
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(thoughtStream))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","model":"claude",` +
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
			`"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// only returns the beta header of the one request received.
func (b *betaRecorder) only(t *testing.T) string {
	t.Helper()
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.betas) != 1 {
		t.Fatalf("requests = %d, want one", len(b.betas))
	}
	return strings.Join(b.betas[0], ",")
}

// discardSink drops what a tool-capable generation reports.
type discardSink struct{}

func (discardSink) Text(string) error          { return nil }
func (discardSink) Tool(frames.ToolCall) error { return nil }

// TestEveryRequestOptsIntoInterleavedThinking checks the streamed generation,
// the tool-capable one and the one-shot inference each ask for interleaved
// thinking, whether thinking is configured or not.
func TestEveryRequestOptsIntoInterleavedThinking(t *testing.T) {
	thinking := map[string]*ThinkingConfig{
		"thinking unset":    nil,
		"thinking disabled": {Type: "disabled"},
		"thinking enabled":  {Type: "enabled", BudgetTokens: 2048},
	}
	calls := map[string]struct {
		stream bool
		run    func(*LLMService, *frames.LLMContext) error
	}{
		"Generate": {true, func(s *LLMService, c *frames.LLMContext) error {
			return s.Generate(t.Context(), c, func(string) error { return nil })
		}},
		"GenerateWithTools": {true, func(s *LLMService, c *frames.LLMContext) error {
			return s.GenerateWithTools(t.Context(), c, discardSink{})
		}},
		"RunInference": {false, func(s *LLMService, c *frames.LLMContext) error {
			_, err := s.RunInference(t.Context(), c, llm.InferenceOptions{})
			return err
		}},
	}
	for tname, cfg := range thinking {
		for cname, call := range calls {
			t.Run(tname+"/"+cname, func(t *testing.T) {
				rec := &betaRecorder{}
				srv := rec.server(t, call.stream)
				svc := NewLLM(Config{APIKey: "k", BaseURL: srv.URL, Thinking: cfg})
				convo := frames.NewLLMContext("sys")
				convo.AddUserMessage("hi")
				if err := call.run(svc, convo); err != nil {
					t.Fatalf("%s: %v", cname, err)
				}
				if got := rec.only(t); got != interleavedThinkingBeta {
					t.Errorf("anthropic-beta = %q, want %q", got, interleavedThinkingBeta)
				}
			})
		}
	}
}

// TestOmitInterleavedThinkingBeta checks a service built for a backend that
// does not take the beta sends none.
func TestOmitInterleavedThinkingBeta(t *testing.T) {
	rec := &betaRecorder{}
	srv := rec.server(t, false)
	svc := NewLLM(Config{APIKey: "k", BaseURL: srv.URL, OmitInterleavedThinkingBeta: true})
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	if _, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{}); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got := rec.only(t); got != "" {
		t.Errorf("anthropic-beta = %q, want none", got)
	}
}
