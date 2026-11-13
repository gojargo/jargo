package stt

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
)

// How a session is torn down: whether the provider is told the audio is over,
// which depends on how the pipeline ended, and the order the session and its
// read are ended in, which decides whether what the session sends as it closes
// reaches the server.

// errSessionClosed is what a closing stream's read returns once it is closed.
//
//nolint:gochecknoglobals // sentinel error
var errSessionClosed = errors.New("session closed")

// closingStream records what the teardown did to it, in order. Its read ends
// when the stream is closed, as a socket's does, or when its context ends.
type closingStream struct {
	ctx    context.Context //nolint:containedctx // the session context, set on dial
	closed chan struct{}

	mu      sync.Mutex
	steps   []string
	reading bool
	// atClose is what held when Close was called: whether the session's
	// context was still live, and whether its read was still pending.
	ctxLiveAtClose, readingAtClose bool
}

func newClosingStream() *closingStream { return &closingStream{closed: make(chan struct{})} }

func (s *closingStream) Send([]byte) error { return nil }

func (s *closingStream) Recv() ([]Result, error) {
	s.mu.Lock()
	s.reading = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.reading = false
		s.mu.Unlock()
	}()
	select {
	case <-s.closed:
		return nil, errSessionClosed
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
}

func (s *closingStream) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.steps = append(s.steps, "stop")
	return nil
}

func (s *closingStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.steps = append(s.steps, "close")
	s.ctxLiveAtClose = s.ctx.Err() == nil
	s.readingAtClose = s.reading
	close(s.closed)
	return nil
}

func (s *closingStream) record() (steps []string, ctxLive, reading bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.steps...), s.ctxLiveAtClose, s.readingAtClose
}

func (s *closingStream) isReading() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reading
}

type closingConnector struct{ stream *closingStream }

func (c *closingConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// runClosing runs a service on st until the session is being read, and returns
// the worker with a channel that is closed once the run is over.
func runClosing(t *testing.T, st *closingStream) (*pipeline.Worker, <-chan struct{}) {
	t.Helper()
	svc := NewStream("FakeSTT", &closingConnector{stream: st}, 16000)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	done := make(chan struct{})
	go func() {
		_ = task.Run(context.Background())
		close(done)
	}()
	waitFor(t, "the session to be read", st.isReading)
	return task, done
}

func waitForRun(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline did not shut down")
	}
}

// A pipeline ending gracefully tells a provider that asks for it that the audio
// is over, before the session is closed.
func TestAGracefulEndTellsTheProviderTheAudioIsOver(t *testing.T) {
	t.Parallel()
	st := newClosingStream()
	task, done := runClosing(t, st)

	task.StopWhenDone()
	waitForRun(t, done)

	if steps, _, _ := st.record(); !reflect.DeepEqual(steps, []string{"stop", "close"}) {
		t.Errorf("teardown = %v, want the provider told the audio is over, then the session closed", steps)
	}
}

// A cancel closes the session without telling the provider the audio is over:
// nothing waits on what it would answer.
func TestACancelClosesWithoutTellingTheProviderTheAudioIsOver(t *testing.T) {
	t.Parallel()
	st := newClosingStream()
	task, done := runClosing(t, st)

	task.Cancel(context.Background(), "the caller hung up")
	waitForRun(t, done)

	if steps, _, _ := st.record(); !reflect.DeepEqual(steps, []string{"close"}) {
		t.Errorf("teardown = %v, want the session closed and nothing else", steps)
	}
}

// The session is closed while its context is live and its read still pending,
// so what it sends as it closes goes out on a connection that is still up. Only
// then is the read ended.
func TestASessionIsClosedBeforeItsReadIsEnded(t *testing.T) {
	t.Parallel()
	st := newClosingStream()
	task, done := runClosing(t, st)

	task.StopWhenDone()
	waitForRun(t, done)

	_, ctxLive, reading := st.record()
	if !ctxLive {
		t.Error("the session's context had ended before it was closed")
	}
	if !reading {
		t.Error("the session's read had ended before it was closed")
	}
}

// drainingStream is a closingStream whose provider answers the end of a session
// with what it was still holding, so it is drained before it is closed.
type drainingStream struct{ *closingStream }

func (s drainingStream) Drain() (time.Duration, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.steps = append(s.steps, "drain")
	return 0, nil
}

// reopeningConnector holds settings, asks for a new session on any change, and
// hands out a fresh draining stream per session.
type reopeningConnector struct {
	store STTSettings

	mu      sync.Mutex
	streams []*closingStream
}

func (c *reopeningConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	st := newClosingStream()
	st.ctx = ctx
	c.mu.Lock()
	c.streams = append(c.streams, st)
	c.mu.Unlock()
	return drainingStream{st}, nil
}

func (c *reopeningConnector) Settings() any { return &c.store }

func (c *reopeningConnector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	return true, nil
}

func (c *reopeningConnector) opened() []*closingStream {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*closingStream(nil), c.streams...)
}

// A session replaced for a settings change is ended the way any other is: the
// provider is asked to finish it, and answers, before it is closed. It is not a
// stop, so the provider is not told the audio is over.
func TestASettingsReopenDrainsTheSessionItReplaces(t *testing.T) {
	t.Parallel()
	conn := &reopeningConnector{}
	svc := NewStream("FakeSTT", conn, 16000)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	done := make(chan struct{})
	go func() {
		_ = task.Run(context.Background())
		close(done)
	}()
	waitFor(t, "the session to be read", func() bool {
		s := conn.opened()
		return len(s) == 1 && s[0].isReading()
	})

	task.QueueFrame(updateFrame(t, "model", "another"))
	waitFor(t, "the session to be replaced", func() bool { return len(conn.opened()) == 2 })
	task.StopWhenDone()
	waitForRun(t, done)

	if steps, _, _ := conn.opened()[0].record(); !reflect.DeepEqual(steps, []string{"drain", "close"}) {
		t.Errorf("the replaced session's teardown = %v, want it drained, then closed", steps)
	}
}
