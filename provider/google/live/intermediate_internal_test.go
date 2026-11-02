package live

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// Gemini takes a tool's intermediate results on the tool-response channel
// itself, as responses that keep the call open, and appended messages as
// client content the aggregator decides whether to answer.

const weather = `{"temperature":75}`

// recordingSession is a WebSocket server that records every client message.
type recordingSession struct {
	*httptest.Server
	mu       sync.Mutex
	received []map[string]any
}

func newRecordingSession(t *testing.T) *recordingSession {
	t.Helper()
	r := &recordingSession{}
	r.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		c, err := websocket.Accept(w, req, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		for {
			_, data, err := c.Read(req.Context())
			if err != nil {
				return
			}
			var msg map[string]any
			if json.Unmarshal(data, &msg) != nil {
				continue
			}
			r.mu.Lock()
			r.received = append(r.received, msg)
			r.mu.Unlock()
		}
	}))
	t.Cleanup(r.Close)
	return r
}

// withKey returns the messages carrying key, in order.
func (r *recordingSession) withKey(key string) []map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []map[string]any
	for _, m := range r.received {
		if v, ok := m[key].(map[string]any); ok {
			out = append(out, v)
		}
	}
	return out
}

// functionResponses are the function responses the client sent, in order.
func (r *recordingSession) functionResponses() []map[string]any {
	var out []map[string]any
	for _, tr := range r.withKey("toolResponse") {
		list, _ := tr["functionResponses"].([]any)
		for _, fr := range list {
			if m, ok := fr.(map[string]any); ok {
				out = append(out, m)
			}
		}
	}
	return out
}

func settle() { time.Sleep(100 * time.Millisecond) }

// startedService runs a service for model against srv in a pipeline, with the
// async tool delegate registered and the call call_1 to it.
func startedService(t *testing.T, srv *recordingSession, model string) *Service {
	t.Helper()
	s := New(Config{APIKey: "k", Model: model, BaseURL: "ws" + strings.TrimPrefix(srv.URL, "http")})
	s.RegisterFunction("delegate", func(context.Context, llm.FunctionCallParams) error { return nil },
		llm.WithCancelOnInterruption(false))
	w := pipeline.NewWorker(pipeline.New(s), pipeline.WorkerConfig{})
	started := make(chan struct{})
	events.On(&w.Registry, pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) { close(started) })
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	t.Cleanup(func() {
		w.Cancel(context.Background(), "")
		<-done
	})
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline never started")
	}
	for deadline := time.Now().Add(3 * time.Second); ; {
		s.mu.Lock()
		connected := s.conn != nil
		s.mu.Unlock()
		if connected {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the service never connected")
		}
		time.Sleep(time.Millisecond)
	}
	s.mu.Lock()
	s.toolNames["call_1"] = "delegate"
	s.mu.Unlock()
	return s
}

func resultFrame(final, run bool) *frames.FunctionCallResultFrame {
	f := frames.NewFunctionCallResultFrame("call_1", "delegate", json.RawMessage(`{}`), weather)
	f.Properties = &frames.FunctionCallResultProperties{IsFinal: &final, RunLLM: &run}
	return f
}

func appendFrame() *frames.LLMMessagesAppendFrame {
	f := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleDeveloper, Text: "Backend: On it."}})
	f.RunLLM = true
	return f
}

func (s *Service) withConvo(convo *frames.LLMContext) {
	s.mu.Lock()
	s.convo = convo
	s.mu.Unlock()
}

func TestAnIntermediateResultKeepsTheCallOpen(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv, defaultModel)

	s.handleFunctionCallResult(t.Context(), resultFrame(false, true))
	settle()

	got := srv.functionResponses()
	if len(got) != 1 || got[0]["willContinue"] != true || got[0]["scheduling"] != "WHEN_IDLE" {
		t.Fatalf("responses = %v, want one keeping the call open, WHEN_IDLE", got)
	}
}

func TestASilentResultAsksTheModelNotToAnswer(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv, defaultModel)

	s.handleFunctionCallResult(t.Context(), resultFrame(false, false))
	settle()

	got := srv.functionResponses()
	if len(got) != 1 || got[0]["scheduling"] != "SILENT" {
		t.Fatalf("responses = %v, want one SILENT", got)
	}
	// The guides put scheduling inside the response, the reference beside it.
	inner, _ := got[0]["response"].(map[string]any)
	if inner["scheduling"] != "SILENT" {
		t.Errorf("response = %v, want scheduling inside it too", inner)
	}
}

func TestAFinalResultSettlesTheCall(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv, defaultModel)

	s.handleFunctionCallResult(t.Context(), resultFrame(true, true))
	settle()

	got := srv.functionResponses()
	// The call stays open for good unless the last response closes it.
	if len(got) != 1 || got[0]["willContinue"] != false || got[0]["scheduling"] != "WHEN_IDLE" {
		t.Fatalf("responses = %v, want one closing the call, WHEN_IDLE", got)
	}
	if !s.resultSeen("call_1") {
		t.Error("the call is not settled")
	}
}

func TestAnAppendedMessageGoesInWithoutCompletingTheTurn(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv, defaultModel)
	s.withConvo(frames.NewLLMContext(""))

	s.handleMessagesAppend(t.Context(), appendFrame())
	settle()

	got := srv.withKey("clientContent")
	if len(got) != 1 || got[0]["turnComplete"] != false {
		t.Fatalf("client content = %v, want one not completing the turn", got)
	}
	turns, _ := got[0]["turns"].([]any)
	turn, _ := turns[0].(map[string]any)
	parts, _ := turn["parts"].([]any)
	part, _ := parts[0].(map[string]any)
	if turn["role"] != "user" || part["text"] != "Backend: On it." {
		t.Errorf("turn = %v, want the message as the user", turn)
	}
}

func TestTheAggregatorAskingToRunCompletesTheTurn(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv, defaultModel)
	convo := frames.NewLLMContext("")
	s.withConvo(convo)
	s.handleMessagesAppend(t.Context(), appendFrame())

	s.handleContext(t.Context(), convo, processor.Upstream)
	settle()

	got := srv.withKey("clientContent")
	if len(got) != 2 || got[1]["turnComplete"] != true {
		t.Fatalf("client content = %v, want the turn completed", got)
	}
	// Answered once: a later push with nothing new completes no turn.
	s.handleContext(t.Context(), convo, processor.Upstream)
	settle()
	if n := len(srv.withKey("clientContent")); n != 2 {
		t.Errorf("client content sent %d times, want 2", n)
	}
}

func TestAnInterruptionDropsAReplyOwedForAppendedContent(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv, defaultModel)
	convo := frames.NewLLMContext("")
	s.withConvo(convo)
	s.handleMessagesAppend(t.Context(), appendFrame())

	if err := s.ProcessFrame(t.Context(), frames.NewInterruptionFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	s.handleContext(t.Context(), convo, processor.Upstream)
	settle()

	if n := len(srv.withKey("clientContent")); n != 1 {
		t.Errorf("client content sent %d times, want only the append", n)
	}
}

func TestWithoutAConversationAnAppendStartsOne(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv, defaultModel)

	s.handleMessagesAppend(t.Context(), appendFrame())
	settle()

	got := srv.withKey("clientContent")
	if len(got) != 1 || got[0]["turnComplete"] != true {
		t.Fatalf("client content = %v, want one completing the turn", got)
	}
}

func TestAModelWithoutNonBlockingToolsTakesNoIntermediateResults(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv, "models/gemini-3.1-flash-live-preview")
	if s.AcceptsIntermediateFunctionCallResults() {
		t.Fatal("AcceptsIntermediateFunctionCallResults() = true, want false")
	}

	s.handleFunctionCallResult(t.Context(), resultFrame(false, true))
	settle()

	if got := srv.functionResponses(); len(got) != 0 {
		t.Errorf("responses = %v, want none", got)
	}
}
