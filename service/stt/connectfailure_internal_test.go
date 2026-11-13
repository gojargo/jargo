package stt

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/utils/events"
)

// errUnavailable is a transient refusal, the kind a provider recycling its
// servers answers a handshake with.
//
//nolint:gochecknoglobals // sentinel error
var errUnavailable = errors.New("HTTP 503")

// idleStream is an open session that never produces a result.
type idleStream struct {
	ctx context.Context //nolint:containedctx // the session context, set on dial
}

func (s *idleStream) Send([]byte) error { return nil }

func (s *idleStream) Recv() ([]Result, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *idleStream) Close() error { return nil }

// failingConnector refuses its first failures dials, then opens sessions that
// stay up.
type failingConnector struct {
	failures int

	mu       sync.Mutex
	attempts int
}

func (c *failingConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.attempts++
	if c.attempts <= c.failures {
		return nil, errUnavailable
	}
	return &idleStream{ctx: ctx}, nil
}

func (c *failingConnector) connectAttempts() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.attempts
}

// optionedConnector is a connector whose configuration carries websocket
// options, the way a provider's does.
type optionedConnector struct {
	Connector
	opts wsservice.Options
}

func (c optionedConnector) WebsocketOptions() wsservice.Options { return c.opts }

// newFailingService builds a streaming service over conn whose reconnection,
// configured through the connector's options as a provider's is, waits backoff
// between attempts.
func newFailingService(conn Connector, backoff time.Duration, reconnectOnError bool) *StreamService {
	return NewStream("FailingSTT", optionedConnector{Connector: conn, opts: wsservice.Options{
		ReconnectOnError:        &reconnectOnError,
		ReconnectBackoffMinWait: backoff,
		ReconnectBackoffMaxWait: backoff,
	}}, 16000)
}

// runFailing streams audio through s for d and returns the errors that reached
// the start of the pipeline.
func runFailing(t *testing.T, s *StreamService, d time.Duration) []*frames.ErrorFrame {
	t.Helper()
	var mu sync.Mutex
	var errs []*frames.ErrorFrame
	task := pipeline.NewWorker(pipeline.New(s), pipeline.WorkerConfig{
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			mu.Lock()
			errs = append(errs, ef)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for range int(d / (20 * time.Millisecond)) {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
		time.Sleep(20 * time.Millisecond)
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline did not shut down")
	}
	mu.Lock()
	defer mu.Unlock()
	return append([]*frames.ErrorFrame(nil), errs...)
}

// A session that fails to open is retried the way a dropped one is, and the
// service stays usable once it is up.
func TestFailedConnectIsRetriedLikeADrop(t *testing.T) {
	t.Parallel()

	conn := &failingConnector{failures: 1}
	s := newFailingService(conn, 50*time.Millisecond, true)
	runFailing(t, s, 500*time.Millisecond)

	if got := conn.connectAttempts(); got != 2 {
		t.Errorf("connect attempts = %d, want 2", got)
	}
	if !s.Usable() {
		t.Error("the service is unusable after connecting on the retry")
	}
}

// A connect that keeps failing gives up once the reconnect attempts run out, and
// says so as a permanent failure, so a switcher can move off the service.
func TestConnectThatKeepsFailingGivesUpAsPermanent(t *testing.T) {
	t.Parallel()

	conn := &failingConnector{failures: 100}
	s := newFailingService(conn, 50*time.Millisecond, true)
	errs := runFailing(t, s, 500*time.Millisecond)

	if got := conn.connectAttempts(); got != 1+wsservice.DefaultMaxRetries {
		t.Errorf("connect attempts = %d, want the setup connect and %d retries",
			got, wsservice.DefaultMaxRetries)
	}
	if s.Usable() {
		t.Error("the service is still usable after every reconnect attempt failed")
	}
	if len(errs) == 0 {
		t.Fatal("no error was reported")
	}
	if last := errs[len(errs)-1].Error; !strings.Contains(last, "failed to reconnect after 3 attempts") {
		t.Errorf("last error %q does not say the reconnecting gave up", last)
	}
}

// permanentConnector refuses every dial with a failure it says will recur.
type permanentConnector struct{ failingConnector }

func (c *permanentConnector) Connect(ctx context.Context, rate int) (Stream, error) {
	_, err := c.failingConnector.Connect(ctx, rate)
	return nil, Permanent(err)
}

// A connect the provider says will keep failing, for a reason its category does
// not convey, is reported as permanent and not retried: the service is left
// unusable at once.
func TestAPermanentlyRefusedConnectIsReportedAndNotRetried(t *testing.T) {
	t.Parallel()

	conn := &permanentConnector{failingConnector{failures: 100}}
	s := newFailingService(conn, 50*time.Millisecond, true)
	errs := runFailing(t, s, 300*time.Millisecond)

	if got := conn.connectAttempts(); got != 1 {
		t.Errorf("connect attempts = %d, want 1", got)
	}
	if s.Usable() {
		t.Error("the service is still usable after a permanent refusal")
	}
	if len(errs) == 0 || !errors.Is(errs[0].Err, ErrPermanent) {
		t.Errorf("errors = %v, want the permanent refusal reported", errs)
	}
}

// A session that opened is not reopened.
func TestConnectedServiceIsNotReconnected(t *testing.T) {
	t.Parallel()

	conn := &failingConnector{}
	s := newFailingService(conn, 50*time.Millisecond, true)
	runFailing(t, s, 500*time.Millisecond)

	if got := conn.connectAttempts(); got != 1 {
		t.Errorf("connect attempts = %d, want 1", got)
	}
	if !s.Usable() {
		t.Error("the service is unusable with its session open")
	}
}

// With reconnection turned off, a failed connect is reported and left there.
func TestFailedConnectIsNotRetriedWithoutReconnectOnError(t *testing.T) {
	t.Parallel()

	conn := &failingConnector{failures: 100}
	s := newFailingService(conn, 50*time.Millisecond, false)
	runFailing(t, s, 500*time.Millisecond)

	if got := conn.connectAttempts(); got != 1 {
		t.Errorf("connect attempts = %d, want 1", got)
	}
}

// Tearing the service down stops a reconnect that is waiting out its backoff,
// rather than holding the shutdown for the length of the wait.
func TestTeardownStopsAReconnectWaitingOnBackoff(t *testing.T) {
	t.Parallel()

	conn := &failingConnector{failures: 100}
	s := newFailingService(conn, 30*time.Second, true)

	start := time.Now()
	runFailing(t, s, 200*time.Millisecond)

	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("shutting down took %s, want it not to wait out the backoff", elapsed)
	}
	if got := conn.connectAttempts(); got != 2 {
		t.Errorf("connect attempts = %d, want 2", got)
	}
}

// A session reopened for a settings change that fails to open is retried by the
// read loop, the same as a first connect.
func TestFailedReopenIsRetried(t *testing.T) {
	t.Parallel()

	// The setup connect succeeds, the reopen's dial fails, and the retry after
	// it succeeds.
	conn := &reopenConnector{fail: map[int]bool{2: true}}
	s := newFailingService(conn, 50*time.Millisecond, true)
	ctx := t.Context()
	s.sampleRate, s.sendRate = 16000, 16000
	s.connect(ctx)
	defer s.disconnect(ctx, false)

	s.reopen(ctx)

	deadline := time.Now().Add(3 * time.Second)
	for !s.Connected() || conn.connectAttempts() < 3 {
		if time.Now().After(deadline) {
			t.Fatalf("no session after the failed reopen: connect attempts = %d", conn.connectAttempts())
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := conn.connectAttempts(); got != 3 {
		t.Errorf("connect attempts = %d, want 3", got)
	}
}

// reopenConnector refuses the dials whose number is in fail, counting from one.
type reopenConnector struct {
	fail map[int]bool

	mu       sync.Mutex
	attempts int
}

func (c *reopenConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.attempts++
	if c.fail[c.attempts] {
		return nil, errUnavailable
	}
	return &idleStream{ctx: ctx}, nil
}

func (c *reopenConnector) connectAttempts() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.attempts
}
