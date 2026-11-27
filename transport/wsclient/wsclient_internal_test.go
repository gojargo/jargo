package wsclient

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/gojargo/jargo/transport/wsserver"
)

// makeSession builds a session for the test server that counts how often it
// reports the connection established.
func makeSession(t *testing.T) (*Session, *testServer, *atomic.Int32) {
	t.Helper()
	srv := newTestServer(t, nil)
	var connected atomic.Int32
	callbacks := Callbacks{
		OnConnected: func(context.Context, *websocket.Conn) { connected.Add(1) },
	}
	return NewSession(srv.url, Params{}, callbacks, "Test"), srv, &connected
}

// connectTwice connects the session the way an input and an output transport
// sharing it do: both at once.
func connectTwice(t *testing.T, s *Session) {
	t.Helper()
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := range errs {
		wg.Go(func() { errs[i] = s.Connect(t.Context()) })
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
	}
}

// TestConcurrentSetupOpensASingleWebsocket checks the input and output
// transports, which share one session and both connect it, open one socket.
// They are set up concurrently, so a socket opened per caller would leave the
// losing one dialed with nobody reading it.
func TestConcurrentSetupOpensASingleWebsocket(t *testing.T) {
	session, srv, connected := makeSession(t)

	connectTwice(t, session)

	if n := srv.openedCount(); n != 1 {
		t.Errorf("%d websockets opened, so one goes unread", n)
	}
	if n := connected.Load(); n != 1 {
		t.Errorf("OnConnected called %d times, want once", n)
	}

	_ = session.Disconnect(t.Context())
	_ = session.Disconnect(t.Context())
}

// TestTheWebsocketOutlivesTheFirstTransportToDisconnect checks closing on the
// first disconnect does not leave the other transport sending over a closed
// socket.
func TestTheWebsocketOutlivesTheFirstTransportToDisconnect(t *testing.T) {
	session, srv, _ := makeSession(t)
	connectTwice(t, session)
	conn := srv.conn(t, 0)

	_ = session.Disconnect(t.Context())
	if conn.isClosed(100 * time.Millisecond) {
		t.Fatal("the socket closed on the first disconnect")
	}

	_ = session.Disconnect(t.Context())
	if !conn.isClosed(5 * time.Second) {
		t.Error("the socket stayed open after the last disconnect")
	}
}

// coalescingSerializer emits one coalesced payload every third frame, buffering
// the two before it.
type coalescingSerializer struct {
	wsserver.BaseSerializer

	mu   sync.Mutex
	seen int
}

func (*coalescingSerializer) Setup(processor.Setup) error { return nil }

// Serialize emits the accumulated block on every third audio frame.
func (s *coalescingSerializer) Serialize(f frames.Frame) (wsserver.Message, error) {
	af, ok := f.(frames.OutputAudioFrame)
	if !ok {
		return wsserver.Message{}, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seen++
	if s.seen%3 != 0 {
		return wsserver.Message{}, nil
	}
	a := af.AudioData().Audio
	return wsserver.Message{Data: append(append(append([]byte(nil), a...), a...), a...), Binary: true}, nil
}

// Deserialize is unused; only the output transport is exercised here.
func (*coalescingSerializer) Deserialize([]byte) (frames.Frame, error) {
	return nil, nil //nolint:nilnil // nothing inbound is under test
}

// setUpOutput gives the output the sample rate the pipeline would, without
// connecting it.
func setUpOutput(t *testing.T, out *OutputTransport, sampleRate int) {
	t.Helper()
	if err := out.BaseOutput.Setup(t.Context(), processor.Setup{AudioOutSampleRate: sampleRate}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = out.BaseOutput.Cleanup(context.Background()) })
}

// TestEveryFrameIsPacedWhenPayloadsAreCoalesced checks a serializer that buffers
// audio across calls, and so emits no payload on most of them, has every frame
// paced. Those frames have still been written, so pacing follows the frames
// taken rather than the payloads that go out.
func TestEveryFrameIsPacedWhenPayloadsAreCoalesced(t *testing.T) {
	srv := newTestServer(t, nil)
	params := Params{Serializer: &coalescingSerializer{}, AudioOutEnabled: true}
	out := NewTransport(srv.url, params).outputTransport()
	setUpOutput(t, out, 16000)
	if err := out.session.Connect(t.Context()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = out.session.Disconnect(context.Background()) })

	const interval = 10 * time.Millisecond
	out.sendInterval = interval

	frame := frames.NewOutputAudioRawFrame(make([]byte, 320), 16000, 1)
	start := time.Now()
	for i := range 9 {
		written, err := out.WriteAudio(t.Context(), frame)
		if err != nil || !written {
			t.Fatalf("frame %d written = %v, %v; want true", i, written, err)
		}
	}
	elapsed := time.Since(start)

	// The first frame goes out at once and every later one waits an interval.
	// Pacing only the frames that produced a payload would wait three times.
	if want := 8 * interval; elapsed < want {
		t.Errorf("nine frames took %v, want at least %v: not every frame was paced", elapsed, want)
	}
	conn := srv.conn(t, 0)
	for range 3 {
		conn.next(t)
	}
	select {
	case m := <-conn.received:
		t.Errorf("a fourth payload was sent: %d bytes", len(m))
	case <-time.After(50 * time.Millisecond):
	}
}

// TestARefusedUpgradeIsAHandshakeError checks a server that answers the upgrade
// with a refusal is reported as a refused handshake, with the status and the
// start of the body it refused with, so a caller can tell the server refusing
// from a failure on this side.
func TestARefusedUpgradeIsAHandshakeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "spent token", http.StatusForbidden)
	}))
	defer srv.Close()

	s := NewSession("ws"+strings.TrimPrefix(srv.URL, "http"), Params{}, Callbacks{}, "Test")
	err := s.Connect(t.Context())
	var refused *wsutil.HandshakeError
	if !errors.As(err, &refused) {
		t.Fatalf("Connect = %v, want a *wsutil.HandshakeError", err)
	}
	if refused.StatusCode != http.StatusForbidden || !strings.Contains(string(refused.Body), "spent token") {
		t.Errorf("refused with %d %q, want 403 and the body", refused.StatusCode, refused.Body)
	}
}
