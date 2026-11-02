package realtime

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
	"github.com/gojargo/jargo/utils/events"
)

// A speech-to-speech service takes a tool's intermediate results, its final
// result and messages appended to the conversation as they are produced, and
// runs the model only when the assistant aggregator asks.

const weather = `{"conditions":"sunny"}`

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

func (r *recordingSession) ofType(kind string) []map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []map[string]any
	for _, m := range r.received {
		if m["type"] == kind {
			out = append(out, m)
		}
	}
	return out
}

// items are the conversation items the client created, in order.
func (r *recordingSession) items() []map[string]any {
	var out []map[string]any
	for _, m := range r.ofType("conversation.item.create") {
		item, _ := m["item"].(map[string]any)
		out = append(out, item)
	}
	return out
}

func (r *recordingSession) responses() int { return len(r.ofType("response.create")) }

// settle waits for what the client sent to reach the server.
func settle() { time.Sleep(100 * time.Millisecond) }

// startedService runs the service against srv in a pipeline, with a
// conversation and the call call_1 the session is waiting on.
func startedService(t *testing.T, srv *recordingSession) *Service {
	t.Helper()
	s := New(Config{APIKey: "k", BaseURL: "ws" + strings.TrimPrefix(srv.URL, "http")})
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
	// The service dials once it has passed the StartFrame on.
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
	s.convo = frames.NewLLMContext("")
	s.sessionReady = true
	s.openCalls["call_1"] = true
	s.mu.Unlock()
	return s
}

func resultFrame(final bool) *frames.FunctionCallResultFrame {
	f := frames.NewFunctionCallResultFrame("call_1", "get_weather", json.RawMessage(`{}`), weather)
	f.Properties = &frames.FunctionCallResultProperties{IsFinal: &final}
	return f
}

// itemText is the text of a message item.
func itemText(item map[string]any) string {
	content, _ := item["content"].([]any)
	if len(content) == 0 {
		return ""
	}
	part, _ := content[0].(map[string]any)
	text, _ := part["text"].(string)
	return text
}

// asyncPayload parses an item's text as the async tool envelope.
func asyncPayload(t *testing.T, item map[string]any) frames.AsyncToolMessage {
	t.Helper()
	m, ok := frames.ParseAsyncToolMessage(frames.Message{Role: frames.RoleDeveloper, Text: itemText(item)})
	if !ok {
		t.Fatalf("item %v carries no async tool envelope", item)
	}
	return m
}

func (s *Service) isOpen(callID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.openCalls[callID]
}

func (s *Service) pushContextUpstream(ctx context.Context) {
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	s.handleContext(ctx, convo, processor.Upstream)
}

func (s *Service) responseCreated(ctx context.Context) {
	s.handleResponseEvent(ctx, serverEvent{Type: "response.created"})
}

func (s *Service) responseDone(ctx context.Context) {
	s.handleResponseEvent(ctx, serverEvent{Type: "response.done"})
}

func TestAnIntermediateResultGoesInBesideTheCall(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.handleFunctionCallResult(t.Context(), resultFrame(false))
	settle()

	items := srv.items()
	if len(items) != 1 || items[0]["type"] != "message" {
		t.Fatalf("items = %v, want one message", items)
	}
	p := asyncPayload(t, items[0])
	if p.Kind != frames.AsyncToolIntermediate || p.Result != weather {
		t.Errorf("payload = %+v, want the intermediate result", p)
	}
	// The call stays open for the result that settles it.
	if !s.isOpen("call_1") {
		t.Error("the call was closed by an intermediate result")
	}
}

func TestAFinalResultAnswersTheCall(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.handleFunctionCallResult(t.Context(), resultFrame(true))
	settle()

	items := srv.items()
	if len(items) != 1 || items[0]["type"] != "function_call_output" ||
		items[0]["call_id"] != "call_1" || items[0]["output"] != weather {
		t.Fatalf("items = %v, want the call answered", items)
	}
	if s.isOpen("call_1") {
		t.Error("the call is still open after its final result")
	}
}

// A new session no longer knows the id, and would reject an output for it.
func TestAResultForACallTheSessionLostGoesInAsAMessage(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)
	s.mu.Lock()
	s.openCalls = map[string]bool{}
	s.mu.Unlock()

	s.handleFunctionCallResult(t.Context(), resultFrame(true))
	settle()

	items := srv.items()
	if len(items) != 1 || items[0]["type"] != "message" {
		t.Fatalf("items = %v, want one message", items)
	}
	if p := asyncPayload(t, items[0]); p.Kind != frames.AsyncToolFinal {
		t.Errorf("payload kind = %v, want final", p.Kind)
	}
}

func TestNothingRunsTheModelUntilTheAggregatorAsks(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.handleFunctionCallResult(t.Context(), resultFrame(false))
	settle()
	if n := srv.responses(); n != 0 {
		t.Fatalf("responses = %d before the aggregator asked, want none", n)
	}

	s.pushContextUpstream(t.Context())
	settle()
	if n := srv.responses(); n != 1 {
		t.Fatalf("responses = %d, want one", n)
	}
}

func TestAContextFrameFromDownstreamRecordsRatherThanRuns(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.handleFunctionCallResult(t.Context(), resultFrame(false))
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	s.handleContext(t.Context(), convo, processor.Downstream)
	settle()

	if n := srv.responses(); n != 0 {
		t.Errorf("responses = %d, want none", n)
	}
}

func TestARunAskedForDuringAResponseWaitsForItToFinish(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.responseCreated(t.Context())
	s.handleFunctionCallResult(t.Context(), resultFrame(true))
	s.pushContextUpstream(t.Context())
	settle()
	if n := srv.responses(); n != 0 {
		t.Fatalf("responses = %d while one was running, want none", n)
	}

	s.responseDone(t.Context())
	settle()
	if n := srv.responses(); n != 1 {
		t.Fatalf("responses = %d once it was done, want one", n)
	}
}

func TestAResponseTakesInWhatWasDeliveredBeforeIt(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.handleFunctionCallResult(t.Context(), resultFrame(true))
	s.responseCreated(t.Context())
	s.responseDone(t.Context())
	s.pushContextUpstream(t.Context())
	settle()

	if n := srv.responses(); n != 0 {
		t.Errorf("responses = %d, want none", n)
	}
}

func TestAnInterruptionDropsARunThatWasWaiting(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.responseCreated(t.Context())
	s.handleFunctionCallResult(t.Context(), resultFrame(true))
	s.pushContextUpstream(t.Context())

	if err := s.ProcessFrame(t.Context(), frames.NewInterruptionFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	s.responseDone(t.Context())
	settle()

	if n := srv.responses(); n != 0 {
		t.Errorf("responses = %d, want none", n)
	}
}

func TestACanceledCallIsSettled(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.handleFunctionCallCancel(t.Context(), frames.NewFunctionCallCancelFrame("call_1", "delegate"))
	settle()

	items := srv.items()
	if len(items) != 1 || items[0]["type"] != "function_call_output" ||
		items[0]["output"] != frames.ToolResultCanceled {
		t.Fatalf("items = %v, want the call settled as canceled", items)
	}
}

func appendFrame(run bool) *frames.LLMMessagesAppendFrame {
	f := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: "Backend: On it."}})
	f.RunLLM = run
	return f
}

func TestAnAppendedMessageGoesInAsAnItemAndRunsWhenTheAggregatorAsks(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.handleMessagesAppend(t.Context(), appendFrame(true))
	settle()

	items := srv.items()
	if len(items) != 1 || items[0]["type"] != "message" || items[0]["role"] != "user" ||
		itemText(items[0]) != "Backend: On it." {
		t.Fatalf("items = %v, want the message as a user item", items)
	}
	if n := srv.responses(); n != 0 {
		t.Fatalf("responses = %d before the aggregator asked, want none", n)
	}

	s.pushContextUpstream(t.Context())
	settle()
	if n := srv.responses(); n != 1 {
		t.Fatalf("responses = %d, want one", n)
	}
}

func TestAnAppendedMessageNobodyAsksAboutRunsNothing(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)

	s.handleMessagesAppend(t.Context(), appendFrame(false))
	settle()

	if n := len(srv.items()); n != 1 {
		t.Errorf("items = %d, want one", n)
	}
	if n := srv.responses(); n != 0 {
		t.Errorf("responses = %d, want none", n)
	}
}

func TestNothingIsAppendedBeforeTheConversationIsSetUp(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)
	s.mu.Lock()
	s.convo = nil
	s.mu.Unlock()

	s.handleMessagesAppend(t.Context(), appendFrame(true))
	settle()

	if items := srv.items(); len(items) != 0 {
		t.Errorf("items = %v, want none", items)
	}
}

func TestTheContextScanNeitherErrorsNorResends(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedService(t, srv)
	var errs []*frames.ErrorFrame
	var mu sync.Mutex
	events.On(s.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		mu.Lock()
		errs = append(errs, ef)
		mu.Unlock()
	})

	s.handleFunctionCallResult(t.Context(), resultFrame(false))
	s.handleFunctionCallResult(t.Context(), resultFrame(true))
	settle()
	sent := len(srv.items())

	// The same results, as the aggregator records them in the conversation.
	s.mu.Lock()
	s.convo.AddMessage(frames.NewAsyncToolIntermediateMessage("call_1", weather))
	s.convo.AddMessage(frames.NewAsyncToolFinalMessage("call_1", weather))
	s.mu.Unlock()
	s.pushContextUpstream(t.Context())
	settle()

	if n := len(srv.items()); n != sent {
		t.Errorf("items = %d after the scan, want the %d already sent", n, sent)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(errs) != 0 {
		t.Errorf("errors = %v, want none", errs)
	}
}
