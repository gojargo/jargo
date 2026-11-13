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
	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// droppingSession is a Live server that acknowledges each session's setup,
// hands out a resumption handle, and then drops the first drops sessions.
type droppingSession struct {
	*httptest.Server
	mu     sync.Mutex
	setups []map[string]any
}

func newDroppingSession(t *testing.T, drops int) *droppingSession {
	t.Helper()
	d := &droppingSession{}
	d.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		c, err := websocket.Accept(w, req, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		_, data, err := c.Read(req.Context())
		if err != nil {
			return
		}
		var msg map[string]any
		_ = json.Unmarshal(data, &msg)
		setup, _ := msg["setup"].(map[string]any)
		d.mu.Lock()
		d.setups = append(d.setups, setup)
		n := len(d.setups)
		d.mu.Unlock()
		for _, m := range []map[string]any{
			{"setupComplete": map[string]any{}},
			{"sessionResumptionUpdate": map[string]any{"newHandle": "handle-1", "resumable": true}},
		} {
			raw, _ := json.Marshal(m)
			if c.Write(req.Context(), websocket.MessageText, raw) != nil {
				return
			}
		}
		if n <= drops {
			return // dropped
		}
		for {
			if _, _, err := c.Read(req.Context()); err != nil {
				return
			}
		}
	}))
	t.Cleanup(d.Close)
	return d
}

func (d *droppingSession) setupCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.setups)
}

func (d *droppingSession) setup(i int) map[string]any {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.setups[i]
}

// startedOn sets up a service against url, which opens its session.
func startedOn(t *testing.T, url string) (*Service, *capture, *capture) {
	t.Helper()
	s := New(Config{APIKey: "k", BaseURL: url})
	sink, source := newCapture(), newCapture()
	s.Link(sink)
	source.Link(s)
	ctx := context.Background()
	setup := processor.Setup{Clock: clock.NewSystem()}
	for _, p := range []processor.Processor{source, sink, s} {
		if err := p.Setup(ctx, setup); err != nil {
			t.Fatalf("setup %s: %v", p.Name(), err)
		}
		t.Cleanup(func() { _ = p.Cleanup(ctx) })
	}
	if err := s.Base.ProcessFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(s.stop)
	return s, sink, source
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(3 * time.Second); !cond(); {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// The setup always asks for resumption, so the server hands out handles.
func TestTheSetupAsksForResumption(t *testing.T) {
	setup := setupOf(t, New(Config{APIKey: "k"}))
	if r, ok := setup["sessionResumption"].(map[string]any); !ok || len(r) != 0 {
		t.Errorf("sessionResumption = %v, want an empty request", setup["sessionResumption"])
	}
}

// A lost session is opened again, resuming from the last handle the server
// gave, and the resumed session takes input at once.
func TestALostSessionIsResumed(t *testing.T) {
	srv := newDroppingSession(t, 1)
	s, _, _ := startedOn(t, wsURL(&recordingSession{Server: srv.Server}))

	waitFor(t, "the session to be opened again", func() bool { return srv.setupCount() >= 2 })
	resumption, _ := srv.setup(1)["sessionResumption"].(map[string]any)
	if resumption["handle"] != "handle-1" {
		t.Errorf("sessionResumption = %v, want the handle the server gave", resumption)
	}
	waitFor(t, "the resumed session to take input", s.readyForInput.Load)
}

// After maxConsecutiveFailures lost sessions in a row, the service stops
// reopening and reports an error.
func TestRepeatedLossesAreReportedAndNotRetried(t *testing.T) {
	srv := newDroppingSession(t, 10)
	_, _, source := startedOn(t, wsURL(&recordingSession{Server: srv.Server}))

	waitFor(t, "the error to be reported", func() bool { return hasName(source.names(), "ErrorFrame") })
	time.Sleep(200 * time.Millisecond)
	if n := srv.setupCount(); n != maxConsecutiveFailures {
		t.Errorf("opened %d sessions, want %d before giving up", n, maxConsecutiveFailures)
	}
}

// A session closed by the service itself is not reopened.
func TestAStoppedServiceDoesNotReopen(t *testing.T) {
	srv := newDroppingSession(t, 0)
	s, _, _ := startedOn(t, wsURL(&recordingSession{Server: srv.Server}))
	waitFor(t, "the session to be ready", s.ready.Load)

	s.stop()
	time.Sleep(200 * time.Millisecond)
	if n := srv.setupCount(); n != 1 {
		t.Errorf("opened %d sessions, want the one", n)
	}
}

// replyDroppingSession is a Live server whose first session starts a spoken
// reply and drops before finishing it, once drop is closed. The sessions after
// it are acknowledged and kept open.
type replyDroppingSession struct {
	*droppingSession
}

func newReplyDroppingSession(t *testing.T, drop <-chan struct{}) *replyDroppingSession {
	t.Helper()
	d := &replyDroppingSession{droppingSession: &droppingSession{}}
	d.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		c, err := websocket.Accept(w, req, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusInternalError, "Internal error encountered.") }()
		_, data, err := c.Read(req.Context())
		if err != nil {
			return
		}
		var msg map[string]any
		_ = json.Unmarshal(data, &msg)
		setup, _ := msg["setup"].(map[string]any)
		d.mu.Lock()
		d.setups = append(d.setups, setup)
		first := len(d.setups) == 1
		d.mu.Unlock()
		if c.Write(req.Context(), websocket.MessageText, []byte(`{"setupComplete":{}}`)) != nil {
			return
		}
		if first {
			if c.Write(req.Context(), websocket.MessageText, []byte(audioMessage([]byte{1, 2, 3, 4}))) != nil {
				return
			}
			select {
			case <-drop:
			case <-req.Context().Done():
			}
			return // dropped mid-reply
		}
		for {
			if _, _, err := c.Read(req.Context()); err != nil {
				return
			}
		}
	}))
	t.Cleanup(d.Close)
	return d
}

// replyRun runs a service against srv in a pipeline, counting the frames that
// reach its end.
type replyRun struct {
	s    *Service
	w    *pipeline.Worker
	done chan struct{}

	mu     sync.Mutex
	counts map[string]int
}

func runReply(t *testing.T, srv *replyDroppingSession) *replyRun {
	t.Helper()
	r := &replyRun{done: make(chan struct{}), counts: map[string]int{}}
	r.s = New(Config{APIKey: "k", BaseURL: wsURL(&recordingSession{Server: srv.Server})})
	r.w = pipeline.NewWorker(pipeline.New(r.s), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&r.w.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		// Names carry a per-instance suffix ("#7"); the type is what matters.
		name, _, _ := strings.Cut(f.Name(), "#")
		r.mu.Lock()
		r.counts[name]++
		r.mu.Unlock()
	})
	go func() {
		_ = r.w.Run(context.Background())
		close(r.done)
	}()
	t.Cleanup(func() {
		r.w.Cancel(context.Background(), "")
		<-r.done
	})
	return r
}

func (r *replyRun) count(name string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.counts[name]
}

// awaitEnd waits for the pipeline to end, well before a held EndFrame's
// deferral would release it.
func (r *replyRun) awaitEnd(t *testing.T) {
	t.Helper()
	select {
	case <-r.done:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline did not end: the EndFrame was held for a reply that will never finish")
	}
}

// assertReplyEnded checks the cut-off reply was closed once, as a completed
// turn would have closed it.
func (r *replyRun) assertReplyEnded(t *testing.T) {
	t.Helper()
	for _, name := range []string{"TTSStartedFrame", "TTSStoppedFrame", "LLMFullResponseEndFrame"} {
		if got := r.count(name); got != 1 {
			t.Errorf("%s reached the end %d times, want 1", name, got)
		}
	}
	r.s.mu.Lock()
	defer r.s.mu.Unlock()
	if r.s.botResponding {
		t.Error("the service still believes the model is answering")
	}
}

// A session that drops while the bot is speaking ends the reply, so an EndFrame
// after the reconnect is not held for a reply that will never finish.
func TestADropMidReplyEndsTheReply(t *testing.T) {
	drop := make(chan struct{})
	close(drop)
	srv := newReplyDroppingSession(t, drop)
	r := runReply(t, srv)

	waitFor(t, "the reconnect", func() bool { return srv.setupCount() >= 2 })
	waitFor(t, "the new session to be ready", r.s.ready.Load)
	r.w.StopWhenDone()
	r.awaitEnd(t)

	r.assertReplyEnded(t)
}

// An EndFrame held for the bot's reply is released when the session drops
// before the reply finishes, and the session is not opened again for a
// pipeline that is ending.
func TestAnEndFrameHeldForTheReplyIsReleasedWhenItDrops(t *testing.T) {
	drop := make(chan struct{})
	srv := newReplyDroppingSession(t, drop)
	r := runReply(t, srv)

	waitFor(t, "the bot to start replying", func() bool {
		r.s.mu.Lock()
		defer r.s.mu.Unlock()
		return r.s.botResponding
	})
	r.w.StopWhenDone()
	waitFor(t, "the EndFrame to be held", func() bool {
		r.s.mu.Lock()
		defer r.s.mu.Unlock()
		return r.s.deferredEnd != nil
	})
	close(drop)
	r.awaitEnd(t)

	r.assertReplyEnded(t)
	if n := srv.setupCount(); n != 1 {
		t.Errorf("opened %d sessions, want no reconnect while the pipeline is ending", n)
	}
}

// A failed send while the service closes its own session, to stop or to open it
// again, is expected and ignored.
func TestASendErrorWhileDisconnectingIsIgnored(t *testing.T) {
	for name, disconnect := range map[string]func(s *Service){
		"stopping":  func(s *Service) { s.stopped.Store(true) },
		"reopening": func(s *Service) { s.disconnect() },
	} {
		t.Run(name, func(t *testing.T) {
			s, _, source := newTestServiceBothWays(t)
			s.ready.Store(true)
			disconnect(s)

			s.handleSendError(context.Background(), errNotGenerator)
			if hasName(source.names(), "ErrorFrame") {
				t.Error("an error was reported for a send failing on the service's own disconnect")
			}
		})
	}
}

// A failed send on a ready session is reported as an error; one on a session
// not yet ready is expected and ignored.
func TestASendErrorIsReported(t *testing.T) {
	s, _, source := newTestServiceBothWays(t)

	s.handleSendError(context.Background(), errNotGenerator)
	if hasName(source.names(), "ErrorFrame") {
		t.Fatal("an error was reported for a session not yet ready")
	}

	s.ready.Store(true)
	s.handleSendError(context.Background(), errNotGenerator)
	var found bool
	for _, f := range source.frames() {
		if e, ok := f.(*frames.ErrorFrame); ok && !e.Fatal {
			found = true
		}
	}
	if !found {
		t.Error("a send error on a ready session was not reported")
	}
}
