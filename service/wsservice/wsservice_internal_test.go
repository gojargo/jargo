package wsservice

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/wsutil"
)

// Static failures for the tests to hand back, so each one names what broke.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errBroke    = errors.New("something broke")
	errDialFail = errors.New("fail")
	errRefused  = errors.New("connection refused")
	errPipe     = errors.New("broken pipe")
)

// clock is a stand-in for the wall clock, so a test can say how long a
// connection lasted rather than wait for it to last that long.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func newClock() *clock {
	return &clock{now: time.Date(2026, time.August, 3, 12, 0, 0, 0, time.UTC)}
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// handler is a Handler whose every step is scripted by the test.
type handler struct {
	mu sync.Mutex

	// receive is what ReceiveMessages does on its call'th call, counting from
	// one. Returning nil stands for the peer closing the connection gracefully.
	receive func(call int) error
	// connect is what ConnectWebsocket returns on its call'th call.
	connect func(call int) error
	// live is what Connected reports. A redial that leaves nothing behind is
	// how a service reports a dial it could not complete.
	live func() bool

	receiveCalls    int
	connectCalls    int
	disconnectCalls int
}

func (h *handler) ConnectWebsocket(context.Context) error {
	h.mu.Lock()
	h.connectCalls++
	call := h.connectCalls
	fn := h.connect
	h.mu.Unlock()
	if fn == nil {
		return nil
	}
	return fn(call)
}

func (h *handler) DisconnectWebsocket(context.Context) error {
	h.mu.Lock()
	h.disconnectCalls++
	h.mu.Unlock()
	return nil
}

func (h *handler) ReceiveMessages(context.Context) error {
	h.mu.Lock()
	h.receiveCalls++
	call := h.receiveCalls
	fn := h.receive
	h.mu.Unlock()
	if fn == nil {
		return nil
	}
	return fn(call)
}

func (h *handler) Connected() bool {
	h.mu.Lock()
	fn := h.live
	h.mu.Unlock()
	if fn == nil {
		return true
	}
	return fn()
}

func (h *handler) counts() (receive, connect, disconnect int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.receiveCalls, h.connectCalls, h.disconnectCalls
}

// recorder collects what the base reported to the pipeline.
type recorder struct {
	mu       sync.Mutex
	messages []string
}

func (r *recorder) report(_ context.Context, message string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.messages = append(r.messages, message)
}

func (r *recorder) all() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.messages...)
}

func (r *recorder) last() string {
	all := r.all()
	if len(all) == 0 {
		return ""
	}
	return all[len(all)-1]
}

// newTestBase builds a Base whose backoff costs no wall-clock time and whose
// clock the test drives.
func newTestBase(t *testing.T, h *handler, clk *clock) *Base {
	t.Helper()
	b := New(h, Config{})
	b.now = clk.Now
	b.sleep = func(context.Context, time.Duration) {}
	return b
}

func closeErr(code websocket.StatusCode, reason string) error {
	return websocket.CloseError{Code: code, Reason: reason}
}

// The receive loop, and what it makes of each way a connection can end.

func TestNormalClosureExitsCleanly(t *testing.T) {
	t.Parallel()

	h := &handler{receive: func(int) error {
		return closeErr(websocket.StatusNormalClosure, "Normal closure")
	}}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())

	b.ReceiveTaskHandler(t.Context(), rec.report)

	if got := rec.all(); len(got) != 0 {
		t.Errorf("reported %v, want nothing reported for a normal closure", got)
	}
	if _, connects, _ := h.counts(); connects != 0 {
		t.Errorf("redialed %d times, want no reconnection after a normal closure", connects)
	}
}

func TestCloseWithErrorTriggersReconnect(t *testing.T) {
	t.Parallel()

	h := &handler{}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())
	h.receive = func(call int) error {
		if call == 1 {
			return closeErr(websocket.StatusAbnormalClosure, "Abnormal closure")
		}
		b.Disconnect()
		return nil
	}

	b.ReceiveTaskHandler(t.Context(), rec.report)

	receives, connects, _ := h.counts()
	if receives != 2 {
		t.Errorf("read %d times, want 2", receives)
	}
	if connects != 1 {
		t.Errorf("redialed %d times, want 1", connects)
	}
}

func TestGracefulServerCloseTriggersReconnect(t *testing.T) {
	t.Parallel()

	h := &handler{}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())
	h.receive = func(call int) error {
		if call > 1 {
			b.Disconnect()
		}
		return nil
	}

	b.ReceiveTaskHandler(t.Context(), rec.report)

	receives, connects, _ := h.counts()
	if receives != 2 {
		t.Errorf("read %d times, want 2", receives)
	}
	if connects != 1 {
		t.Errorf("redialed %d times, want 1", connects)
	}
}

func TestGeneralErrorTriggersReconnect(t *testing.T) {
	t.Parallel()

	h := &handler{}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())
	h.receive = func(call int) error {
		if call == 1 {
			return errBroke
		}
		b.Disconnect()
		return nil
	}

	b.ReceiveTaskHandler(t.Context(), rec.report)

	receives, connects, _ := h.counts()
	if receives != 2 {
		t.Errorf("read %d times, want 2", receives)
	}
	if connects != 1 {
		t.Errorf("redialed %d times, want 1", connects)
	}
}

// Retrying a server that will not answer.

func TestReconnectSucceedsOnLaterAttempt(t *testing.T) {
	t.Parallel()

	h := &handler{connect: func(call int) error {
		if call < 3 {
			return errDialFail
		}
		return nil
	}}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())

	if !b.TryReconnect(t.Context(), rec.report) {
		t.Fatal("reconnect gave up despite the third attempt succeeding")
	}
	if _, connects, _ := h.counts(); connects != 3 {
		t.Errorf("redialed %d times, want 3", connects)
	}
}

func TestReconnectExhaustedReportsTheFailure(t *testing.T) {
	t.Parallel()

	h := &handler{connect: func(int) error { return errRefused }}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())

	if b.TryReconnect(t.Context(), rec.report) {
		t.Fatal("reconnect claimed success with every attempt failing")
	}
	if _, connects, _ := h.counts(); connects != 3 {
		t.Errorf("redialed %d times, want 3", connects)
	}
	if last := rec.last(); !strings.Contains(last, "connection refused") {
		t.Errorf("final report %q does not name the failure that caused it", last)
	}
}

// A redial that reports its failure by leaving no connection behind, rather than
// by returning an error, still counts as a failed attempt.
func TestReconnectExhaustedWhenRedialLeavesNoConnection(t *testing.T) {
	t.Parallel()

	h := &handler{live: func() bool { return false }}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())

	if b.TryReconnect(t.Context(), rec.report) {
		t.Fatal("reconnect claimed success with no connection to show for it")
	}
	if got := len(rec.all()); got != 4 {
		t.Errorf("reported %d times, want 4: one per attempt plus one for giving up", got)
	}
	if last := rec.last(); !strings.Contains(last, "failed to reconnect after 3 attempts") {
		t.Errorf("final report %q does not say the retrying ran out", last)
	}
	if first := rec.all()[0]; !strings.Contains(first, ErrVerification.Error()) {
		t.Errorf("first report %q does not say the redial left nothing behind", first)
	}
}

// A connection that keeps dying the instant it is established.

func TestQuickFailuresEndTheRetrying(t *testing.T) {
	t.Parallel()

	clk := newClock()
	h := &handler{receive: func(int) error {
		// The connection dies at once, so no time passes before it fails.
		return closeErr(websocket.StatusPolicyViolation, "Invalid API key")
	}}
	rec := &recorder{}
	b := newTestBase(t, h, clk)

	b.ReceiveTaskHandler(t.Context(), rec.report)

	receives, _, _ := h.counts()
	if receives != b.tracker.MaxConsecutiveFailures() {
		t.Errorf("read %d times, want %d: one per tolerated instant failure",
			receives, b.tracker.MaxConsecutiveFailures())
	}
	reported := rec.all()
	if len(reported) != 1 {
		t.Fatalf("reported %v, want a single report saying the retrying stopped", reported)
	}
	if !strings.Contains(reported[0], "failed 3 times immediately after connecting") {
		t.Errorf("report %q does not say why the retrying stopped", reported[0])
	}
}

func TestStableConnectionResetsTheQuickFailureStreak(t *testing.T) {
	t.Parallel()

	clk := newClock()
	// How long each connection lasts before it fails. The third outlives the
	// stable duration, so the streak before it does not count towards the
	// three that end the retrying.
	lifetimes := []time.Duration{0, time.Second, 10 * time.Second, time.Second, time.Second, time.Second}
	h := &handler{receive: func(call int) error {
		if call <= len(lifetimes) {
			clk.advance(lifetimes[call-1])
		}
		return closeErr(websocket.StatusAbnormalClosure, "Abnormal closure")
	}}
	rec := &recorder{}
	b := newTestBase(t, h, clk)

	b.ReceiveTaskHandler(t.Context(), rec.report)

	if receives, _, _ := h.counts(); receives != len(lifetimes) {
		t.Errorf("read %d times, want %d: the stable connection restarts the count",
			receives, len(lifetimes))
	}
	if got := rec.all(); len(got) != 1 {
		t.Errorf("reported %v, want a single report saying the retrying stopped", got)
	}
}

// Lifecycle and guards.

func TestDeliberateDisconnectPreventsReconnection(t *testing.T) {
	t.Parallel()

	h := &handler{receive: func(int) error {
		return closeErr(websocket.StatusAbnormalClosure, "Abnormal closure")
	}}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())
	b.Disconnect()

	b.ReceiveTaskHandler(t.Context(), rec.report)

	if got := rec.all(); len(got) != 0 {
		t.Errorf("reported %v, want nothing reported for a disconnect we asked for", got)
	}
	if _, connects, _ := h.counts(); connects != 0 {
		t.Errorf("redialed %d times, want none while disconnecting", connects)
	}
}

func TestConnectResetsState(t *testing.T) {
	t.Parallel()

	b := newTestBase(t, &handler{}, newClock())
	b.Disconnect()
	b.tracker.Record(time.Second)
	b.tracker.Record(time.Second)

	b.Connect()

	if b.Disconnecting() {
		t.Error("connecting left the service marked as disconnecting")
	}
	if got := b.tracker.Count(); got != 0 {
		t.Errorf("quick-failure streak after connecting: got %d, want 0", got)
	}
}

func TestReconnectRefusesToRunTwiceAtOnce(t *testing.T) {
	t.Parallel()

	h := &handler{}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())
	// A reconnect entered from within a reconnect is the concurrent attempt the
	// guard exists to refuse.
	var inner bool
	h.connect = func(int) error {
		inner = b.TryReconnect(t.Context(), rec.report)
		return nil
	}

	if !b.TryReconnect(t.Context(), rec.report) {
		t.Fatal("the outer reconnect failed")
	}
	if inner {
		t.Error("a second reconnect ran while the first was still going")
	}
}

// Sending, and the reconnect behind it.

func TestSendWithRetryReconnectsAndSendsAgain(t *testing.T) {
	t.Parallel()

	h := &handler{}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())

	var sends int
	err := b.SendWithRetry(t.Context(), func() error {
		sends++
		if sends == 1 {
			return errPipe
		}
		return nil
	}, rec.report)
	if err != nil {
		t.Fatalf("send failed after a successful reconnect: %v", err)
	}
	if sends != 2 {
		t.Errorf("sent %d times, want 2: the failure and the retry after reconnecting", sends)
	}
	if _, connects, _ := h.counts(); connects != 1 {
		t.Errorf("redialed %d times, want 1", connects)
	}
}

func TestSendWithRetryReturnsTheFailureWhenReconnectingFails(t *testing.T) {
	t.Parallel()

	h := &handler{connect: func(int) error { return errRefused }}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())

	want := errPipe
	var sends int
	err := b.SendWithRetry(t.Context(), func() error {
		sends++
		return want
	}, rec.report)

	if !errors.Is(err, want) {
		t.Errorf("got error %v, want the send failure %v", err, want)
	}
	if sends != 1 {
		t.Errorf("sent %d times, want 1: there was no connection to retry on", sends)
	}
}

func TestSendWithRetryTreatsNoConnectionAsAFailedSend(t *testing.T) {
	t.Parallel()

	var live bool
	h := &handler{
		live:    func() bool { return live },
		connect: func(int) error { live = true; return nil },
	}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())

	var sends int
	err := b.SendWithRetry(t.Context(), func() error {
		sends++
		return nil
	}, rec.report)
	if err != nil {
		t.Fatalf("send failed after reconnecting: %v", err)
	}
	if sends != 1 {
		t.Errorf("sent %d times, want 1: the first attempt had no connection to send on", sends)
	}
	if _, connects, _ := h.counts(); connects != 1 {
		t.Errorf("redialed %d times, want 1", connects)
	}
}

// A handshake the server refused ends the retrying at once: it will refuse the
// next dial the same way, so redialing only delays the news.
func TestPermanentRefusalStopsRetrying(t *testing.T) {
	t.Parallel()

	refused := &wsutil.HandshakeError{StatusCode: 401, Err: errRefused}
	h := &handler{connect: func(int) error { return refused }}
	rec := &recorder{}
	b := newTestBase(t, h, newClock())

	if b.TryReconnect(t.Context(), rec.report) {
		t.Fatal("reconnect claimed success against a refused handshake")
	}
	if _, connects, _ := h.counts(); connects != 1 {
		t.Errorf("redialed %d times, want 1: a refusal is not worth repeating", connects)
	}
	if last := rec.last(); !strings.Contains(last, "giving up") || !strings.Contains(last, "401") {
		t.Errorf("final report %q does not say it gave up, or on what", last)
	}
}

// A refusal the server may not repeat is retried like any other failure.
func TestServerErrorIsRetried(t *testing.T) {
	t.Parallel()

	failed := &wsutil.HandshakeError{StatusCode: 503, Err: errRefused}
	h := &handler{connect: func(int) error { return failed }}
	b := newTestBase(t, h, newClock())

	if b.TryReconnect(t.Context(), nil) {
		t.Fatal("reconnect claimed success with every attempt failing")
	}
	if _, connects, _ := h.counts(); connects != 3 {
		t.Errorf("redialed %d times, want 3: a server-side failure may not repeat", connects)
	}
}
