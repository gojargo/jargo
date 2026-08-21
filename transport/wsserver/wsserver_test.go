package wsserver_test

// End-to-end tests for the telephony WebSocket transport. This is the shared
// machinery under every provider serializer (Twilio, Telnyx, Plivo, Exotel,
// rtviws), so a fault here is a fault on every phone call the framework serves.
//
// Each test stands up a real HTTP server, dials it with a real WebSocket client,
// and drives a real pipeline; only the wire format is faked, through a test
// Serializer. Nothing is mocked below the socket.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
)

// errSerialize is the failure the test serializer reports on demand.
//
//nolint:gochecknoglobals // sentinel error
var errSerialize = errors.New("serialize failed")

// message is the test wire format: a kind and an opaque payload, which is enough
// to tell audio, clears and hang-ups apart without importing a real provider.
type message struct {
	Kind    string `json:"kind"`
	Payload string `json:"payload"`
}

// testSerializer is a minimal Serializer that records what it was asked to do
// and can be made to fail.
type testSerializer struct {
	mu sync.Mutex

	setupCalled  bool
	setupRate    int
	setupErr     error
	serializeErr error
	// dropAudio makes Serialize return (nil, nil) for audio, the way a provider
	// does before its handshake completes.
	dropAudio bool
	// badMessages counts inbound messages that failed to deserialize.
	badMessages int
	// serialized records the kind of every message Serialize produced.
	serialized []string
}

func (s *testSerializer) Setup(st processor.Setup) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.setupCalled = true
	s.setupRate = st.AudioOutSampleRate
	return s.setupErr
}

func (s *testSerializer) Serialize(f frames.Frame) ([]byte, error) {
	s.mu.Lock()
	serErr, drop := s.serializeErr, s.dropAudio
	s.mu.Unlock()

	var m message
	switch fr := f.(type) {
	case frames.OutputAudioFrame:
		if drop {
			return nil, nil //nolint:nilnil // provider not ready for audio yet
		}
		if serErr != nil {
			return nil, serErr
		}
		m = message{Kind: "audio", Payload: string(fr.AudioData().Audio)}
	case *frames.InterruptionFrame:
		m = message{Kind: "clear"}
	case *frames.EndFrame:
		m = message{Kind: "end"}
	case *frames.CancelFrame:
		m = message{Kind: "cancel"}
	default:
		return nil, nil //nolint:nilnil // frame not sent on this transport
	}
	s.mu.Lock()
	s.serialized = append(s.serialized, m.Kind)
	s.mu.Unlock()
	return json.Marshal(m)
}

// sent reports whether a message of the given kind was produced.
func (s *testSerializer) sent(kind string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Contains(s.serialized, kind)
}

func (s *testSerializer) Deserialize(data []byte) (frames.Frame, error) {
	var m message
	if err := json.Unmarshal(data, &m); err != nil {
		s.mu.Lock()
		s.badMessages++
		s.mu.Unlock()
		return nil, err
	}
	switch m.Kind {
	case "audio":
		return frames.NewInputAudioRawFrame([]byte(m.Payload), 8000, 1), nil
	case "text":
		return frames.NewTranscriptionFrame(m.Payload, "caller", ""), nil
	default:
		return nil, nil //nolint:nilnil // handshake or keepalive; no frame
	}
}

func (s *testSerializer) snapshot() (setupCalled bool, rate, bad int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.setupCalled, s.setupRate, s.badMessages
}

// tap sits between the transport input and output and records every frame that
// passes through, since frames consumed by the output processor never reach the
// pipeline's downstream end.
type tap struct {
	*processor.Base
	mu   sync.Mutex
	seen []frames.Frame
}

func newTap() *tap {
	t := &tap{}
	t.Base = processor.New("Tap", t)
	return t
}

func (t *tap) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := t.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	t.mu.Lock()
	t.seen = append(t.seen, f)
	t.mu.Unlock()
	return t.PushFrame(ctx, f, dir)
}

func (t *tap) frames() []frames.Frame {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]frames.Frame(nil), t.seen...)
}

// call is one live WebSocket session bridged to a pipeline.
type call struct {
	client *websocket.Conn
	task   *pipeline.Worker
	done   chan error
	tr     *wsserver.Transport
	tap    *tap
	cancel context.CancelFunc

	stop sync.Once
}

// dial serves the transport over httptest, dials it, and runs a pipeline whose
// head is the transport input and tail is its output.
func dial(t *testing.T, ser wsserver.Serializer, params transport.Params) *call {
	t.Helper()
	return dialTuned(t, ser, params, nil)
}

// dialTuned is dial with a hook to adjust the task params, for tests that need
// to observe upstream traffic.
func dialTuned(
	t *testing.T, ser wsserver.Serializer, params transport.Params, tune func(*pipeline.WorkerConfig),
) *call {
	t.Helper()

	c := &call{done: make(chan error, 1)}
	ready := make(chan struct{})

	ctx, cancel := context.WithCancel(context.Background())
	c.cancel = cancel
	t.Cleanup(cancel)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tr, err := wsserver.Accept(w, r, ser, params)
		if err != nil {
			t.Errorf("Accept: %v", err)
			close(ready)
			return
		}
		c.tr = tr
		c.tap = newTap()
		// No RTVI: what is under test is the audio the transport writes, and an
		// RTVI processor would put its own client messages on the same socket.
		noRTVI := false
		tp := pipeline.WorkerConfig{
			EnableRTVI: &noRTVI,
			Params: pipeline.Params{
				AudioInSampleRate:  8000,
				AudioOutSampleRate: 8000,
			},
		}
		if tune != nil {
			tune(&tp)
		}
		c.task = pipeline.NewWorker(pipeline.New(tr.Input(), c.tap, tr.Output()), tp)
		close(ready)
		c.done <- c.task.Run(ctx)
	}))
	t.Cleanup(srv.Close)

	conn, resp, err := websocket.Dial(t.Context(), "ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	c.client = conn
	t.Cleanup(func() { _ = conn.CloseNow() })

	select {
	case <-ready:
	case <-time.After(5 * time.Second):
		t.Fatal("the server never accepted the connection")
	}
	return c
}

// send writes one client message onto the socket.
func (c *call) send(t *testing.T, m message) {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := c.client.Write(t.Context(), websocket.MessageText, b); err != nil {
		t.Fatalf("client write: %v", err)
	}
}

// sendRaw writes arbitrary bytes, for the malformed-message cases.
func (c *call) sendRaw(t *testing.T, b []byte) {
	t.Helper()
	if err := c.client.Write(t.Context(), websocket.MessageText, b); err != nil {
		t.Fatalf("client write: %v", err)
	}
}

// expect reads one message from the socket and decodes it.
func (c *call) expect(t *testing.T) message {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	_, data, err := c.client.Read(ctx)
	if err != nil {
		t.Fatalf("client read: %v", err)
	}
	var m message
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unmarshal %q: %v", data, err)
	}
	return m
}

// ready blocks until the transport has begun reading the socket, which the task's
// own StartFrame triggers — the test must not queue a second one, or the input
// starts a competing read loop.
func (c *call) ready(t *testing.T) {
	t.Helper()
	c.send(t, message{Kind: "text", Payload: readyProbe})
	c.waitFor(t, "the read loop to start", func(fs []frames.Frame) bool {
		tf := findTranscription(fs)
		return tf != nil && tf.Text == readyProbe
	})
}

// readyProbe is the sentinel message ready sends to observe the read loop.
const readyProbe = "__ready__"

// waitFor polls the frames that reached the pipeline tail until pred is
// satisfied.
func (c *call) waitFor(t *testing.T, what string, pred func([]frames.Frame) bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if pred(c.tap.frames()) {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func (c *call) shutdown(t *testing.T) {
	t.Helper()
	c.stop.Do(func() {
		c.task.StopWhenDone()
		select {
		case <-c.done:
			return
		case <-time.After(5 * time.Second):
		}
		// The socket may already be gone; cancel the run rather than hanging.
		c.cancel()
		select {
		case <-c.done:
		case <-time.After(5 * time.Second):
			t.Error("pipeline did not finish")
		}
	})
}

// params returns telephony-shaped transport params: 8 kHz mono, both directions.
func params() transport.Params {
	p := transport.DefaultParams()
	p.AudioInSampleRate = 8000
	p.AudioOutSampleRate = 8000
	p.AudioOut10msChunks = 1
	return p
}

// TestSetupRunsBeforeReading checks the serializer is configured from the
// StartFrame before any socket traffic is deserialized — it is where a provider
// learns the sample rate it must encode at.
func TestSetupRunsBeforeReading(t *testing.T) {
	ser := &testSerializer{}
	c := dial(t, ser, params())
	defer c.shutdown(t)

	c.send(t, message{Kind: "text", Payload: "hello"})
	c.waitFor(t, "the transcription to arrive", func(fs []frames.Frame) bool {
		return findTranscription(fs) != nil
	})

	called, rate, _ := ser.snapshot()
	if !called {
		t.Error("Setup was never called")
	}
	if rate != 8000 {
		t.Errorf("Setup saw AudioOutSampleRate = %d, want 8000", rate)
	}
}

// TestSetupErrorEndsTheCall checks a serializer that cannot configure itself
// ends the session instead of stalling it.
//
// The ordering is the point. The StartFrame must reach the rest of the pipeline
// before the serializer is set up; configuring first and failing would swallow
// the StartFrame, leaving every processor downstream uninitialized and the run
// unable to finish. Setup therefore runs in StartReading, after the push and
// before the first inbound message is decoded.
//
// The failure is fatal rather than logged and shrugged off: a session that
// cannot speak the provider's wire format can neither hear nor answer, so the
// pipeline ends and the socket closes.
func TestSetupErrorEndsTheCall(t *testing.T) {
	ser := &testSerializer{setupErr: errSerialize}
	c := dial(t, ser, params())

	// The budget is deliberately tight. Tearing the session down with the normal
	// close handshake would block the frame-processing goroutine for up to five
	// seconds waiting on the peer, wedging the pipeline; aborting takes
	// microseconds. A regression to the handshake fails here rather than passing
	// slowly.
	select {
	case <-c.done:
	case <-time.After(2 * time.Second):
		t.Fatal("the pipeline neither started nor finished after a Setup failure")
	}
	c.stop.Do(func() {}) // already drained

	if called, _, _ := ser.snapshot(); !called {
		t.Error("Setup was never called")
	}

	// Done must close, or an app waiting on it to tear the call down would leak
	// the session: the read loop never started, so nothing else closes it.
	select {
	case <-c.tr.Done():
	case <-time.After(2 * time.Second):
		t.Error("Done was not closed after the Setup failure")
	}

	// The StartFrame reached the pipeline despite the failure.
	var sawStart bool
	for _, f := range c.tap.frames() {
		if _, ok := f.(*frames.StartFrame); ok {
			sawStart = true
		}
	}
	if !sawStart {
		t.Error("the StartFrame never reached the pipeline; Setup must not run before the push")
	}
}

// TestInboundAudioBecomesFrames is the read path: provider media messages must
// arrive downstream as input audio for VAD, STT and turn detection.
func TestInboundAudioBecomesFrames(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	defer c.shutdown(t)

	c.send(t, message{Kind: "audio", Payload: "caller-speech"})

	c.waitFor(t, "inbound audio", func(fs []frames.Frame) bool {
		for _, f := range fs {
			if af, ok := f.(*frames.InputAudioRawFrame); ok && string(af.Audio) == "caller-speech" {
				return true
			}
		}
		return false
	})
}

// TestInboundNonAudioFrames checks messages that deserialize to something other
// than audio still reach the pipeline.
func TestInboundNonAudioFrames(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	defer c.shutdown(t)

	c.send(t, message{Kind: "text", Payload: "press one"})

	c.waitFor(t, "the transcription", func(fs []frames.Frame) bool {
		tf := findTranscription(fs)
		return tf != nil && tf.Text == "press one"
	})
}

// TestMalformedMessagesDoNotKillTheCall is the important resilience property: a
// provider sending one bad message must not end a live phone call. The read loop
// logs and continues.
func TestMalformedMessagesDoNotKillTheCall(t *testing.T) {
	ser := &testSerializer{}
	c := dial(t, ser, params())
	defer c.shutdown(t)

	c.sendRaw(t, []byte("{not json"))
	c.send(t, message{Kind: "keepalive"}) // deserializes to no frame
	c.send(t, message{Kind: "text", Payload: "still here"})

	c.waitFor(t, "traffic after a malformed message", func(fs []frames.Frame) bool {
		tf := findTranscription(fs)
		return tf != nil && tf.Text == "still here"
	})

	if _, _, bad := ser.snapshot(); bad != 1 {
		t.Errorf("deserialize failures = %d, want 1", bad)
	}
}

// TestOutboundAudio is the write path: TTS audio must reach the socket as a
// provider media message.
func TestOutboundAudio(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	defer c.shutdown(t)

	// One 10 ms chunk at 8 kHz mono 16-bit is 160 bytes.
	c.task.QueueFrame(frames.NewTTSAudioRawFrame(make([]byte, 160), 8000, 1))

	if got := c.expect(t); got.Kind != "audio" {
		t.Errorf("message kind = %q, want audio", got.Kind)
	}
}

// TestOutboundAudioDropped checks a serializer returning no message (the
// provider is not ready) writes nothing rather than an empty frame.
func TestOutboundAudioDropped(t *testing.T) {
	c := dial(t, &testSerializer{dropAudio: true}, params())
	defer c.shutdown(t)

	c.task.QueueFrame(frames.NewTTSAudioRawFrame(make([]byte, 160), 8000, 1))
	c.task.QueueFrame(frames.NewInterruptionFrame())

	// The clear arrives first because the audio produced no wire message at all.
	if got := c.expect(t); got.Kind != "clear" {
		t.Errorf("message kind = %q, want clear; the dropped audio should not be sent", got.Kind)
	}
}

// TestOutboundAudioIsPaced checks audio leaves at the rate it plays rather than
// the rate it is produced. Without pacing a whole utterance lands in the
// provider's playout buffer at once, and a barge-in then cuts only audio that
// has not been handed over yet, so the caller keeps hearing the bot. Upstream
// has no test for this.
func TestOutboundAudioIsPaced(t *testing.T) {
	p := params()
	p.AudioOut10msChunks = 10 // 100 ms chunks, long enough to time reliably
	c := dial(t, &testSerializer{}, p)
	defer c.shutdown(t)

	// One 100 ms chunk at 8 kHz mono 16-bit is 1600 bytes. Queue four at once,
	// so pacing is the only thing that can spread them out.
	const chunk = 1600
	c.task.QueueFrame(frames.NewTTSAudioRawFrame(make([]byte, chunk*4), 8000, 1))

	start := time.Now()
	for i := range 4 {
		if got := c.expect(t); got.Kind != "audio" {
			t.Fatalf("message %d kind = %q, want audio", i, got.Kind)
		}
	}
	elapsed := time.Since(start)

	// Each chunk is sent and only then does the clock wait, so the first two go
	// out back to back and every later one waits a full interval. Four chunks
	// therefore take about two intervals to all arrive.
	if want := 150 * time.Millisecond; elapsed < want {
		t.Errorf("four 100 ms chunks arrived in %v, want at least %v: output is not paced", elapsed, want)
	}
}

// TestInterruptionSendsClear covers barge-in: the provider must be told to
// discard the audio it has already buffered, or the bot keeps talking over the
// caller.
func TestInterruptionSendsClear(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	defer c.shutdown(t)

	c.task.QueueFrame(frames.NewInterruptionFrame())

	if got := c.expect(t); got.Kind != "clear" {
		t.Errorf("message kind = %q, want clear", got.Kind)
	}
}

// TestEndFrameReachesSerializer checks the serializer sees the end of the
// pipeline, which is its cue to hang the call up. The assertion is on what the
// serializer produced rather than what arrived on the socket, because the
// transport closes the connection immediately afterwards.
func TestEndFrameReachesSerializer(t *testing.T) {
	ser := &testSerializer{}
	c := dial(t, ser, params())

	c.ready(t)
	c.shutdown(t)

	if !ser.sent("end") {
		t.Error("the serializer never saw the EndFrame; it cannot hang the call up")
	}
}

// TestClientDisconnectClosesDone checks the caller hanging up closes Done, which
// is the signal an app cancels the pipeline context on.
func TestClientDisconnectClosesDone(t *testing.T) {
	c := dial(t, &testSerializer{}, params())

	c.ready(t)

	_ = c.client.Close(websocket.StatusNormalClosure, "caller hung up")

	select {
	case <-c.tr.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("Done was not closed when the client disconnected")
	}
	c.shutdown(t)
}

// TestDoneClosedOnce checks Done is closed exactly once even when the client
// disconnects and the pipeline stops — a double close would panic.
func TestDoneClosedOnce(t *testing.T) {
	c := dial(t, &testSerializer{}, params())

	c.ready(t)

	_ = c.client.Close(websocket.StatusNormalClosure, "caller hung up")
	<-c.tr.Done()
	c.shutdown(t) // stopping the pipeline closes the session a second time

	select {
	case <-c.tr.Done():
	default:
		t.Error("Done should stay closed")
	}
}

// TestInputOutputProcessors checks the transport exposes both ends of the
// pipeline it is meant to sit at.
func TestInputOutputProcessors(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	defer c.shutdown(t)

	if c.tr.Input() == nil {
		t.Error("Input() is nil")
	}
	if c.tr.Output() == nil {
		t.Error("Output() is nil")
	}
	if !strings.HasPrefix(c.tr.Input().Name(), "WSInput#") {
		t.Errorf("Input().Name() = %q, want the WSInput label", c.tr.Input().Name())
	}
	if !strings.HasPrefix(c.tr.Output().Name(), "WSOutput#") {
		t.Errorf("Output().Name() = %q, want the WSOutput label", c.tr.Output().Name())
	}
}

// TestAcceptRejectsPlainHTTP checks a request that is not a WebSocket upgrade is
// refused rather than producing a half-built transport.
func TestAcceptRejectsPlainHTTP(t *testing.T) {
	var acceptErr error
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, acceptErr = wsserver.Accept(w, r, &testSerializer{}, params())
	}))
	defer srv.Close()

	resp, err := srv.Client().Get(srv.URL) //nolint:noctx // httptest client, no timeout needed
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	_ = resp.Body.Close()

	if acceptErr == nil {
		t.Error("Accept accepted a plain HTTP request, want an error")
	}
}

func findTranscription(fs []frames.Frame) *frames.TranscriptionFrame {
	for _, f := range fs {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			return tf
		}
	}
	return nil
}

// teardownBudget is far below the five seconds a WebSocket close handshake may
// spend waiting on a peer, so a regression that starts performing the handshake
// during teardown fails here instead of merely running slowly.
const teardownBudget = time.Second

// TestTeardownIsPromptWhenPeerIsSilent checks ending a call does not wait on the
// close handshake. Stopping the read cancels its context, and the WebSocket
// library closes the connection itself when a read fails, so the deferred Close
// finds the socket already closed and returns immediately rather than sending a
// close frame and waiting for a reply that a silent peer will never send.
func TestTeardownIsPromptWhenPeerIsSilent(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	c.ready(t)

	// The client never reads, so it never auto-replies to a close frame.
	start := time.Now()
	c.shutdown(t)

	if elapsed := time.Since(start); elapsed > teardownBudget {
		t.Errorf("teardown took %v, want under %v; the close handshake is being waited on", elapsed, teardownBudget)
	}
}

// TestTeardownIsPromptWhenPeerVanished is the telephony case that matters: the
// provider tore the call down on its side, so nothing will ever answer a close
// frame. Teardown must not stall on it.
func TestTeardownIsPromptWhenPeerVanished(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	c.ready(t)
	_ = c.client.CloseNow()

	start := time.Now()
	c.shutdown(t)

	if elapsed := time.Since(start); elapsed > teardownBudget {
		t.Errorf("teardown took %v, want under %v", elapsed, teardownBudget)
	}
}

// TestClientConnectedIsReportedOnce checks the caller being on the line reaches
// the pipeline as a frame. The socket is accepted before the pipeline is built,
// so without this nothing tells a processor or an observer that a client is
// there at all: on this transport the connection is a precondition of the run
// rather than an event during it.
func TestClientConnectedIsReportedOnce(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	defer c.shutdown(t)

	c.ready(t)

	var connected int
	for _, f := range c.tap.frames() {
		if _, ok := f.(*frames.ClientConnectedFrame); ok {
			connected++
		}
	}
	if connected != 1 {
		t.Fatalf("ClientConnectedFrames = %d, want exactly one", connected)
	}
}

// TestClientConnectedComesBeforeAnythingRead pins the ordering. The report is
// what a measurement of how long the call took to become answerable is anchored
// to, so it has to land before the traffic it is measuring against.
func TestClientConnectedComesBeforeAnythingRead(t *testing.T) {
	c := dial(t, &testSerializer{}, params())
	defer c.shutdown(t)

	c.ready(t)

	connected, probe := -1, -1
	for i, f := range c.tap.frames() {
		switch fr := f.(type) {
		case *frames.ClientConnectedFrame:
			if connected < 0 {
				connected = i
			}
		case *frames.TranscriptionFrame:
			if probe < 0 && fr.Text == readyProbe {
				probe = i
			}
		}
	}
	if connected < 0 {
		t.Fatal("the client connection was never reported")
	}
	if probe < 0 {
		t.Fatal("the probe message never reached the pipeline")
	}
	if connected > probe {
		t.Errorf("the connection was reported at %d, after the first message read at %d", connected, probe)
	}
}

// TestTheConnectionIsTimedByAnObserver is the end-to-end reason the frame
// exists. An observer measuring how long a call took to become answerable has
// nothing to measure unless the transport says the caller arrived, so this is
// the wiring, not the observer, that the test is about.
func TestTheConnectionIsTimedByAnObserver(t *testing.T) {
	reports := make(chan observers.TransportTimingReport, 4)
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnTransportTimingReport: func(r observers.TransportTimingReport) { reports <- r },
	})

	c := dialTuned(t, &testSerializer{}, params(), func(cfg *pipeline.WorkerConfig) {
		cfg.Observers = append(cfg.Observers, o)
	})
	defer c.shutdown(t)

	select {
	case r := <-reports:
		if r.ClientConnected <= 0 {
			t.Errorf("ClientConnected = %s, want the time it took to get there", r.ClientConnected)
		}
		if r.BotConnected != nil {
			t.Errorf("BotConnected = %s, want none: there is no room for a bot to join", *r.BotConnected)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the transport timing was never reported")
	}
}
