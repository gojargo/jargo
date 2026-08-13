package stt_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
)

// scriptedStream replays canned results, then ends the session the way the test
// asked for: with endErr, or by blocking until the session is canceled.
type scriptedStream struct {
	results [][]stt.Result
	endErr  error

	idx int
	ctx context.Context //nolint:containedctx // the session context, set by the connector on dial
}

func (s *scriptedStream) Send([]byte) error { return nil }

func (s *scriptedStream) Recv() ([]stt.Result, error) {
	if s.idx < len(s.results) {
		r := s.results[s.idx]
		s.idx++
		return r, nil
	}
	if s.endErr != nil {
		return nil, s.endErr
	}
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *scriptedStream) Close() error { return nil }

// scriptedConnector hands out one scripted session per dial. Once the script
// runs out it hands out sessions that stay open, so a test that expects no
// further reconnection blocks rather than spins.
type scriptedConnector struct {
	mu       sync.Mutex
	sessions []*scriptedStream
	dials    int
}

func (c *scriptedConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := c.dials
	c.dials++
	if n < len(c.sessions) {
		s := c.sessions[n]
		s.ctx = ctx
		return s, nil
	}
	return &scriptedStream{ctx: ctx}, nil
}

func (c *scriptedConnector) Dials() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.dials
}

// collector records what reached each end of the pipeline.
type collector struct {
	mu    sync.Mutex
	texts []string
	errs  []*frames.ErrorFrame
	got   chan struct{}
}

func newCollector() *collector {
	return &collector{got: make(chan struct{}, 64)}
}

func (c *collector) downstream(f frames.Frame) {
	fr, ok := f.(*frames.TranscriptionFrame)
	if !ok {
		return
	}
	c.mu.Lock()
	c.texts = append(c.texts, fr.Text)
	c.mu.Unlock()
	c.signal()
}

func (c *collector) upstream(f frames.Frame) {
	fr, ok := f.(*frames.ErrorFrame)
	if !ok {
		return
	}
	c.mu.Lock()
	c.errs = append(c.errs, fr)
	c.mu.Unlock()
	c.signal()
}

func (c *collector) signal() {
	select {
	case c.got <- struct{}{}:
	default:
	}
}

func (c *collector) transcriptions() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.texts...)
}

func (c *collector) errorFrames() []*frames.ErrorFrame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*frames.ErrorFrame(nil), c.errs...)
}

// wait blocks until cond holds or the test runs out of patience. It wakes on
// every frame collected, and polls besides, for a condition that no frame
// announces.
func (c *collector) wait(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for {
		if cond() {
			return
		}
		select {
		case <-c.got:
		case <-tick.C:
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

// runService runs svc in a pipeline until stop is called.
func runService(t *testing.T, svc *stt.StreamService, c *collector) (stop func()) {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream,
		func(_ context.Context, f frames.Frame) { c.downstream(f) })
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream,
		func(_ context.Context, f frames.Frame) { c.upstream(f) })
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("pipeline did not shut down")
		}
	}
}

func closed(code websocket.StatusCode, reason string) error {
	return websocket.CloseError{Code: code, Reason: reason}
}

// A session that drops mid-call is reopened, and transcription carries on.
// Without the reconnection the read loop ends with the first session and the
// rest of the call is transcribed by nobody.
func TestStreamServiceReopensADroppedSession(t *testing.T) {
	conn := &scriptedConnector{sessions: []*scriptedStream{
		{
			results: [][]stt.Result{{{Text: "before the drop", Final: true}}},
			endErr:  closed(websocket.StatusAbnormalClosure, "abnormal closure"),
		},
		{results: [][]stt.Result{{{Text: "after the drop", Final: true}}}},
	}}
	c := newCollector()
	svc := stt.NewStream("FakeSTT", conn, 16000)
	stop := runService(t, svc, c)
	defer stop()

	c.wait(t, "the transcription that follows the reconnect", func() bool {
		return len(c.transcriptions()) == 2
	})

	if got, want := c.transcriptions(), []string{"before the drop", "after the drop"}; !equal(got, want) {
		t.Errorf("transcriptions = %v, want %v", got, want)
	}
	if got := conn.Dials(); got != 2 {
		t.Errorf("dialed %d times, want 2: the original session and the one replacing it", got)
	}
}

// A peer closing the session the way a peer that is done closes it ends the
// call's transcription. There is nothing to restore.
func TestStreamServiceDoesNotReopenAfterANormalClosure(t *testing.T) {
	conn := &scriptedConnector{sessions: []*scriptedStream{{
		results: [][]stt.Result{{{Text: "all of it", Final: true}}},
		endErr:  closed(websocket.StatusNormalClosure, "normal closure"),
	}}}
	c := newCollector()
	svc := stt.NewStream("FakeSTT", conn, 16000)
	stop := runService(t, svc, c)
	defer stop()

	c.wait(t, "the transcription", func() bool { return len(c.transcriptions()) == 1 })
	// Give a reconnect that should not happen the chance to happen.
	time.Sleep(100 * time.Millisecond)

	if got := conn.Dials(); got != 1 {
		t.Errorf("dialed %d times, want 1: a normal closure is not reconnected", got)
	}
	if got := c.errorFrames(); len(got) != 0 {
		t.Errorf("reported %d errors, want none for a normal closure", len(got))
	}
}

// A session that keeps dying the instant it opens is not waiting on the network,
// so the service stops redialing and says so rather than reconnecting forever.
func TestStreamServiceStopsRedialingAnInstantlyFailingSession(t *testing.T) {
	conn := &scriptedConnector{}
	// Every session the connector hands out dies at once.
	for range 8 {
		conn.sessions = append(conn.sessions, &scriptedStream{
			endErr: closed(websocket.StatusPolicyViolation, "invalid api key"),
		})
	}
	c := newCollector()
	svc := stt.NewStream("FakeSTT", conn, 16000)
	stop := runService(t, svc, c)
	defer stop()

	c.wait(t, "the report that redialing stopped", func() bool {
		return len(c.errorFrames()) > 0
	})

	errs := c.errorFrames()
	if len(errs) != 1 {
		t.Fatalf("reported %d errors, want exactly one saying the redialing stopped", len(errs))
	}
	if errs[0].Fatal {
		t.Error("losing the provider connection was reported as fatal, which ends the call")
	}
	if !strings.Contains(errs[0].Error, "immediately after connecting") {
		t.Errorf("report %q does not say why the redialing stopped", errs[0].Error)
	}
	// Three sessions is what the quick-failure ceiling allows: the original and
	// two replacements.
	if got := conn.Dials(); got != 3 {
		t.Errorf("dialed %d times, want 3", got)
	}
}

// Shutting the pipeline down is not a dropped session, so nothing is reopened.
func TestStreamServiceDoesNotReopenOnShutdown(t *testing.T) {
	conn := &scriptedConnector{}
	c := newCollector()
	svc := stt.NewStream("FakeSTT", conn, 16000)
	stop := runService(t, svc, c)

	c.wait(t, "the session to open", func() bool { return conn.Dials() == 1 })
	stop()
	time.Sleep(100 * time.Millisecond)

	if got := conn.Dials(); got != 1 {
		t.Errorf("dialed %d times, want 1: shutting down is not a drop to recover from", got)
	}
	if got := c.errorFrames(); len(got) != 0 {
		t.Errorf("reported %d errors, want none for a shutdown we asked for", len(got))
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
