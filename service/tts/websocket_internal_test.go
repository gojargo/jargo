package tts

import (
	"bytes"
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/utils/events"
)

const wsSampleRate = 16000

var errConnectionReset = errors.New("connection reset")

// wsSynth is a websocket service with no socket: connecting marks it connected,
// and the audio of each sentence arrives after RunTTS returns, tagged with the
// sentence's position.
type wsSynth struct {
	ws   *Websocket
	host AudioContextHost

	mu    sync.Mutex
	store settings.TTS
	// connected is whether there is a connection, and calls what was called,
	// in order: connect and disconnect for the service's own, dial for the
	// socket alone.
	connected bool
	calls     []string
	sentences int

	// reconnectAfter is the sentence after whose audio the connection is
	// replaced, or 0 for none.
	reconnectAfter int
	// byDisconnecting replaces it with Disconnect and Connect, as a settings
	// change and a failed send do, rather than through a redial.
	byDisconnecting bool
	// firstAudio is the first sentence's audio, if not the default.
	firstAudio []byte
	// needsConnection opens the connection in the first RunTTS call, the way a
	// service reconnects after the provider closed an idle one.
	needsConnection bool
	// reconnectOnSettings replaces the connection when a setting changes.
	reconnectOnSettings bool
	// deliver says whether RunTTS delivers audio at all.
	deliver bool
	// receiveErr is what ReceiveMessages returns once its context ends.
	receive chan error
}

func newWSSynth() *wsSynth { return &wsSynth{deliver: true, receive: make(chan error)} }

func (s *wsSynth) SampleRate() int { return wsSampleRate }

func (s *wsSynth) SetAudioContextHost(h AudioContextHost) { s.host = h }

func (s *wsSynth) Settings() any { return &s.store }

func (s *wsSynth) UpdateSettings(ctx context.Context, _ settings.Changed) error {
	if s.reconnectOnSettings {
		s.Disconnect(ctx)
		s.Connect(ctx)
	}
	return nil
}

func (s *wsSynth) Start(ctx context.Context) { s.Connect(ctx) }

func (s *wsSynth) Connect(ctx context.Context) {
	s.record("connect")
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

func (s *wsSynth) Disconnect(ctx context.Context) {
	s.record("disconnect")
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

func (s *wsSynth) ConnectWebsocket(context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, "dial")
	s.connected = true
	return nil
}

func (s *wsSynth) DisconnectWebsocket(context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.connected = false
	return nil
}

func (s *wsSynth) ReceiveMessages(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-s.receive:
		return err
	}
}

func (s *wsSynth) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.connected
}

func (s *wsSynth) record(call string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, call)
}

// reconnects lists the service's own connects and disconnects, in order.
func (s *wsSynth) reconnects() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	for _, c := range s.calls {
		if c != "dial" {
			out = append(out, c)
		}
	}
	return out
}

func (s *wsSynth) RunTTS(ctx context.Context, _, contextID string, _ func(frames.Frame) error) error {
	s.mu.Lock()
	s.sentences++
	sentence := s.sentences
	lazy := s.needsConnection
	s.needsConnection = false
	deliver := s.deliver
	s.mu.Unlock()
	if lazy {
		_ = s.ConnectWebsocket(ctx)
	}
	if !deliver {
		return nil
	}
	// The audio arrives after the call has returned, on a context of its own.
	connCtx := context.WithoutCancel(ctx)
	go func() {
		time.Sleep(20 * time.Millisecond)
		audio := bytes.Repeat([]byte{byte(sentence), 0}, 160)
		if sentence == 1 && s.firstAudio != nil {
			audio = s.firstAudio
		}
		f := frames.NewTTSAudioRawFrame(audio, wsSampleRate, 1)
		f.ContextID = contextID
		s.host.AppendToAudioContext(contextID, f)
		if sentence != s.reconnectAfter {
			return
		}
		time.Sleep(50 * time.Millisecond)
		if s.byDisconnecting {
			s.Disconnect(connCtx)
			s.Connect(connCtx)
			return
		}
		s.ws.TryReconnect(connCtx)
	}()
	return nil
}

// newWSService builds a websocket TTS service around syn, interruptible or not.
func newWSService(syn *wsSynth, interruptible bool) *Base {
	b := New("WebsocketTTS", syn)
	b.stopFrameTimeout = 300 * time.Millisecond
	if interruptible {
		syn.ws = NewInterruptibleWebsocket(b, syn, wsservice.Options{})
	} else {
		syn.ws = NewWebsocket(b, syn, wsservice.Options{})
	}
	return b
}

// speakTwoSentences runs a reply of two sentences, with between sent after the
// first one. "One." is released when "Two" arrives, and "Two." when the reply
// ends, so anything that happens after the first sentence comes before the
// second.
func speakTwoSentences(t *testing.T, b *Base, between ...frames.Frame) []frames.Frame {
	t.Helper()
	send := []frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		frames.NewTextFrame("One. "),
		frames.NewTextFrame("Two"),
		sleepFor(150 * time.Millisecond),
	}
	send = append(send, between...)
	send = append(send,
		sleepFor(150*time.Millisecond),
		frames.NewTextFrame("."),
		frames.NewLLMFullResponseEndFrame(),
		sleepFor(600*time.Millisecond),
	)
	return runBase(t, b, send...)
}

// audioBySentence groups the audio frames by the sentence they speak.
func audioBySentence(down []frames.Frame) map[byte][]*frames.TTSAudioRawFrame {
	out := map[byte][]*frames.TTSAudioRawFrame{}
	for _, a := range audioOf(down) {
		out[a.Audio[0]] = append(out[a.Audio[0]], a)
	}
	return out
}

type startStop struct {
	started bool
	context string
}

func assertTurnContinuedUnderNewContext(t *testing.T, down []frames.Frame) {
	t.Helper()
	audio := audioBySentence(down)
	if len(audio) != 2 || audio[1] == nil || audio[2] == nil {
		t.Fatalf("audio by sentence = %v, want sentences 1 and 2", audio)
	}
	first, second := audio[1][0].ContextID, audio[2][0].ContextID
	if first == second {
		t.Fatalf("both sentences on context %q, want the second on a new one", first)
	}
	// The old context is stopped before the new one starts.
	var got []startStop
	for _, f := range down {
		switch fr := f.(type) {
		case *frames.TTSStartedFrame:
			got = append(got, startStop{true, fr.ContextID})
		case *frames.TTSStoppedFrame:
			got = append(got, startStop{false, fr.ContextID})
		}
	}
	want := []startStop{{true, first}, {false, first}, {true, second}, {false, second}}
	if !slices.Equal(got, want) {
		t.Errorf("started and stopped = %v, want %v", got, want)
	}
}

func TestReconnectContinuesTurnUnderNewContext(t *testing.T) {
	syn := newWSSynth()
	syn.reconnectAfter = 1
	down := speakTwoSentences(t, newWSService(syn, false))
	assertTurnContinuedUnderNewContext(t, down)
}

func TestDisconnectAndConnectContinuesTurnUnderNewContext(t *testing.T) {
	syn := newWSSynth()
	syn.reconnectAfter = 1
	syn.byDisconnecting = true
	down := speakTwoSentences(t, newWSService(syn, false))
	assertTurnContinuedUnderNewContext(t, down)
}

func TestSettingsChangeMidTurnContinuesTurnUnderNewContext(t *testing.T) {
	syn := newWSSynth()
	syn.reconnectOnSettings = true
	down := speakTwoSentences(t, newWSService(syn, false),
		frames.NewTTSUpdateSettingsFrame(&settings.TTS{Voice: settings.Set("other-voice")}))
	assertTurnContinuedUnderNewContext(t, down)
}

func TestReconnectFlushesHeldPartialSample(t *testing.T) {
	syn := newWSSynth()
	syn.reconnectAfter = 1
	syn.firstAudio = append(bytes.Repeat([]byte{1, 0}, 160), 1)
	down := speakTwoSentences(t, newWSService(syn, false))

	audio := audioBySentence(down)
	var first []byte
	for _, f := range audio[1] {
		first = append(first, f.Audio...)
	}
	if want := append(slices.Clone(syn.firstAudio), 0); !bytes.Equal(first, want) {
		t.Errorf("first sentence audio = %d bytes, want the held byte padded (%d bytes)", len(first), len(want))
	}
	for _, f := range audio[2] {
		if len(f.Audio)%2 != 0 {
			t.Errorf("second sentence frame of %d bytes, want whole samples", len(f.Audio))
		}
	}
}

func TestLazyConnectionKeepsTheContextItSendsTo(t *testing.T) {
	syn := newWSSynth()
	syn.needsConnection = true
	down := speakTwoSentences(t, newWSService(syn, false))

	audio := audioBySentence(down)
	if len(audio) != 2 || audio[1] == nil || audio[2] == nil {
		t.Fatalf("audio by sentence = %v, want sentences 1 and 2", audio)
	}
	if audio[1][0].ContextID != audio[2][0].ContextID {
		t.Errorf("contexts = %q, %q, want the same one", audio[1][0].ContextID, audio[2][0].ContextID)
	}
}

// wsRun is a running pipeline around a websocket service, whose frames can be
// sent one at a time and waited on.
type wsRun struct {
	t    *testing.T
	task *pipeline.Worker
	done chan error

	mu   sync.Mutex
	down []frames.Frame
	seen chan struct{}
}

func runWS(t *testing.T, b *Base) *wsRun {
	t.Helper()
	r := &wsRun{t: t, done: make(chan error, 1), seen: make(chan struct{}, 1)}
	r.task = pipeline.NewWorker(pipeline.New(b), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		r.down = append(r.down, f)
		r.mu.Unlock()
		select {
		case r.seen <- struct{}{}:
		default:
		}
	})
	go func() { r.done <- r.task.Run(context.Background()) }()
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.StartFrame); return ok })
	return r
}

// waitFor waits until a frame matching match has reached the end of the
// pipeline.
func (r *wsRun) waitFor(match func(frames.Frame) bool) {
	r.t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		r.mu.Lock()
		found := slices.ContainsFunc(r.down, match)
		r.mu.Unlock()
		if found {
			return
		}
		select {
		case <-r.seen:
		case <-deadline:
			r.t.Fatal("the frame waited for never came")
		}
	}
}

// send queues f and waits until it has gone through the service.
func (r *wsRun) send(f frames.Frame) {
	r.t.Helper()
	r.task.QueueFrame(f)
	r.waitFor(func(g frames.Frame) bool { return g == f })
}

// stop ends the pipeline, gracefully.
func (r *wsRun) stop() {
	r.t.Helper()
	r.task.StopWhenDone()
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		r.t.Fatal("the pipeline did not finish")
	}
}

func (r *wsRun) cancel() {
	r.t.Helper()
	r.task.Cancel(context.Background(), "")
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		r.t.Fatal("the pipeline did not finish")
	}
}

func (w *Websocket) flags() (botSpeaking, ttsStarted bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.botSpeaking, w.ttsStarted
}

// startedSpeech says something, and waits for its TTSStartedFrame, which is
// the moment the synthesis has started and the bot has not been heard yet.
func (r *wsRun) startedSpeech() {
	r.t.Helper()
	r.task.QueueFrame(frames.NewTTSSpeakFrame("Hello."))
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.TTSStartedFrame); return ok })
}

func reconnected(calls []string) bool {
	return slices.Equal(calls, []string{"connect", "disconnect", "connect"})
}

func TestReconnectsWhenBotConfirmedSpeaking(t *testing.T) {
	syn := newWSSynth()
	b := newWSService(syn, true)
	r := runWS(t, b)
	defer r.cancel()

	r.send(frames.NewBotStartedSpeakingFrame())
	if speaking, started := syn.ws.flags(); !speaking || started {
		t.Fatalf("bot speaking %v, tts started %v, want true, false", speaking, started)
	}
	r.send(frames.NewInterruptionFrame())
	if calls := syn.reconnects(); !reconnected(calls) {
		t.Errorf("calls = %v, want a disconnect and a connect after the first connect", calls)
	}
}

func TestReconnectsWhenStartedButNotYetConfirmed(t *testing.T) {
	syn := newWSSynth()
	syn.deliver = false
	b := newWSService(syn, true)
	r := runWS(t, b)
	defer r.cancel()

	r.startedSpeech()
	if speaking, started := syn.ws.flags(); speaking || !started {
		t.Fatalf("bot speaking %v, tts started %v, want false, true", speaking, started)
	}
	r.send(frames.NewInterruptionFrame())
	if calls := syn.reconnects(); !reconnected(calls) {
		t.Errorf("calls = %v, want a disconnect and a connect after the first connect", calls)
	}
	// Consumed by the interruption it was needed for.
	if _, started := syn.ws.flags(); started {
		t.Error("tts started is still set after the interruption")
	}
}

func TestNoReconnectWhenBotNeverSpoke(t *testing.T) {
	syn := newWSSynth()
	b := newWSService(syn, true)
	r := runWS(t, b)
	defer r.cancel()

	r.send(frames.NewInterruptionFrame())
	if calls := syn.reconnects(); !slices.Equal(calls, []string{"connect"}) {
		t.Errorf("calls = %v, want only the first connect", calls)
	}
}

func TestTTSStartedClearedOnNewTurn(t *testing.T) {
	syn := newWSSynth()
	syn.deliver = false
	b := newWSService(syn, true)
	r := runWS(t, b)
	defer r.cancel()

	r.startedSpeech()
	if _, started := syn.ws.flags(); !started {
		t.Fatal("tts started is not set")
	}
	// A new turn begins without the previous one ever resolving it.
	r.send(frames.NewLLMFullResponseStartFrame())
	if _, started := syn.ws.flags(); started {
		t.Fatal("tts started leaked into the new turn")
	}
	r.send(frames.NewInterruptionFrame())
	if calls := syn.reconnects(); !slices.Equal(calls, []string{"connect"}) {
		t.Errorf("calls = %v, want only the first connect", calls)
	}
}

// silentWSSynth completes each context with no audio at all.
type silentWSSynth struct{ *wsSynth }

func (s silentWSSynth) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	go func() {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		s.host.AppendToAudioContext(contextID, stopped)
		s.host.RemoveAudioContext(contextID)
	}()
	return nil
}

// TestSilentTurnResumesFrameProcessing checks a turn that plays nothing does
// not leave frame processing paused for an interruptible service that pauses
// it: the context completes with no audio, and no bot speaking frame ever
// comes.
func TestSilentTurnResumesFrameProcessing(t *testing.T) {
	syn := silentWSSynth{newWSSynth()}
	b := New("WebsocketTTS", syn)
	syn.ws = NewInterruptibleWebsocket(b, syn, wsservice.Options{})
	b.SetPauseFrameProcessing(PauseOptions{Enabled: true})
	down := runBase(t, b,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewTextFrame("Hello."),
		frames.NewLLMFullResponseEndFrame(),
		sleepFor(400*time.Millisecond),
		frames.NewOutputTransportMessageFrame("after silence"),
	)
	if !slices.ContainsFunc(down, func(f frames.Frame) bool {
		m, ok := f.(*frames.OutputTransportMessageFrame)
		return ok && m.Message == "after silence"
	}) {
		t.Error("frame processing stayed paused after a turn completed with no audio")
	}
}

func TestReconnectFlagsTrackFullTurn(t *testing.T) {
	syn := newWSSynth()
	syn.deliver = false
	b := newWSService(syn, true)
	r := runWS(t, b)
	defer r.cancel()

	r.send(frames.NewLLMFullResponseStartFrame())
	r.startedSpeech()
	r.send(frames.NewBotStartedSpeakingFrame())
	r.send(frames.NewBotStoppedSpeakingFrame())
	if speaking, started := syn.ws.flags(); speaking || started {
		t.Fatalf("bot speaking %v, tts started %v, want both cleared", speaking, started)
	}
	r.send(frames.NewInterruptionFrame())
	if calls := syn.reconnects(); !slices.Equal(calls, []string{"connect"}) {
		t.Errorf("calls = %v, want no reconnect after a turn that finished cleanly", calls)
	}
}

// streamingWSSynth streams requests in order on one connection and labels its
// audio with the context playing, the way a provider without context IDs does.
// Reconnecting drops the audio still to come.
type streamingWSSynth struct {
	*wsSynth
	b *Base

	smu      sync.Mutex
	requests chan string
	stop     context.CancelFunc
}

func (s *streamingWSSynth) Connect(ctx context.Context) {
	s.wsSynth.Connect(ctx)
	streamCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	requests := make(chan string, 16)
	s.smu.Lock()
	s.requests, s.stop = requests, stop
	s.smu.Unlock()
	go s.stream(streamCtx, requests)
}

func (s *streamingWSSynth) Disconnect(ctx context.Context) {
	s.wsSynth.Disconnect(ctx)
	s.smu.Lock()
	stop := s.stop
	s.stop = nil
	s.smu.Unlock()
	if stop != nil {
		stop()
	}
}

func (s *streamingWSSynth) Start(ctx context.Context) { s.Connect(ctx) }

func (s *streamingWSSynth) stream(ctx context.Context, requests chan string) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-requests:
		}
		for range 4 {
			select {
			case <-ctx.Done():
				return
			case <-time.After(50 * time.Millisecond):
			}
			id := s.b.ActiveAudioContextID()
			f := frames.NewTTSAudioRawFrame(make([]byte, 320), wsSampleRate, 1)
			f.ContextID = id
			s.b.AppendToAudioContext(id, f)
		}
	}
}

func (s *streamingWSSynth) RunTTS(_ context.Context, text, _ string, _ func(frames.Frame) error) error {
	s.smu.Lock()
	requests := s.requests
	s.smu.Unlock()
	requests <- text
	return nil
}

// TestInterruptedReplyStaysInterruptibleBehindUninterruptibleSpeech checks
// uninterruptible speech sent mid-reply does not keep the interrupted reply
// playing: the connection is reset, so none of the reply's audio is labeled
// with the uninterruptible context.
func TestInterruptedReplyStaysInterruptibleBehindUninterruptibleSpeech(t *testing.T) {
	syn := &streamingWSSynth{wsSynth: newWSSynth()}
	b := New("WebsocketTTS", syn)
	b.stopFrameTimeout = 200 * time.Millisecond
	b.SetPauseFrameProcessing(PauseOptions{Enabled: true})
	syn.b = b
	syn.ws = NewInterruptibleWebsocket(b, syn, wsservice.Options{})

	speech := frames.NewTTSSpeakFrame("One moment.")
	speech.AppendToContext = false
	speech.SetInterruptible(false)
	down := runBase(t, b,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewTextFrame("Hello there. "),
		frames.NewTextFrame("How"),
		speech,
		sleepFor(100*time.Millisecond),
		frames.NewBotStartedSpeakingFrame(),
		sleepFor(50*time.Millisecond),
		frames.NewInterruptionFrame(),
		sleepFor(400*time.Millisecond),
	)
	audio := audioOf(down)
	if len(audio) == 0 {
		t.Fatal("no audio")
	}
	for _, f := range audio {
		if !frames.Interruptible(f) {
			t.Errorf("audio frame of context %q is uninterruptible", f.ContextID)
		}
	}
}

// TestEndDisconnects checks the service disconnects when the pipeline ends.
func TestEndDisconnects(t *testing.T) {
	syn := newWSSynth()
	b := newWSService(syn, false)
	r := runWS(t, b)
	if !syn.ws.ReceiveTaskRunning() {
		t.Fatal("no receive loop after connecting")
	}
	r.stop()
	if calls := syn.reconnects(); len(calls) < 2 || calls[1] != "disconnect" {
		t.Errorf("calls = %v, want a disconnect after the connect", calls)
	}
	if syn.Connected() || syn.ws.ReceiveTaskRunning() {
		t.Error("still connected after the pipeline ended")
	}
}

// TestCancelDisconnects checks a canceled pipeline closes the connection.
func TestCancelDisconnects(t *testing.T) {
	syn := newWSSynth()
	b := newWSService(syn, false)
	r := runWS(t, b)
	r.cancel()
	if calls := syn.reconnects(); len(calls) < 2 || calls[1] != "disconnect" {
		t.Errorf("calls = %v, want a disconnect after the connect", calls)
	}
	if syn.Connected() || syn.ws.ReceiveTaskRunning() {
		t.Error("still connected after the pipeline was canceled")
	}
}

// TestCleanupDisconnects checks tearing the service down disconnects it.
func TestCleanupDisconnects(t *testing.T) {
	syn := newWSSynth()
	b := newWSService(syn, false)
	syn.Connect(context.Background())
	if err := b.Cleanup(context.Background()); err != nil {
		t.Fatal(err)
	}
	if syn.Connected() || syn.ws.ReceiveTaskRunning() {
		t.Error("still connected after cleanup")
	}
}

// TestWebsocketServicesReportNoProcessingMetrics checks the processing metric is
// off for a websocket service, whose RunTTS only hands the text off.
func TestWebsocketServicesReportNoProcessingMetrics(t *testing.T) {
	syn := newWSSynth()
	b := newWSService(syn, false)
	if b.reportsProcessingMetrics() {
		t.Error("a websocket service reports processing metrics")
	}
}

// TestALostConnectionIsRestoredAndReported checks a connection the receive loop
// loses is redialed, and a connection error is raised and pushed for each
// redial that fails.
func TestALostConnectionIsRestoredAndReported(t *testing.T) {
	syn := newWSSynth()
	b := newWSService(syn, false)
	syn.ws.ws = wsservice.New(wsHandler{syn.ws}, wsservice.Config{
		ReconnectBackoffMinWait: time.Millisecond, ReconnectBackoffMaxWait: time.Millisecond,
	})
	var mu sync.Mutex
	var reported []string
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		mu.Lock()
		reported = append(reported, msg)
		mu.Unlock()
	})
	r := runWS(t, b)
	defer r.cancel()

	syn.receive <- errConnectionReset
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if calls := syn.reconnectsWithDials(); len(calls) >= 3 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if calls := syn.reconnectsWithDials(); !slices.Equal(calls, []string{"connect", "dial", "dial"}) {
		t.Errorf("calls = %v, want the connection dialed again", calls)
	}
	if !syn.Connected() || !b.Usable() {
		t.Errorf("connected %v, usable %v, want the connection restored", syn.Connected(), b.Usable())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(reported) != 0 {
		t.Errorf("reported %q for a redial that worked", reported)
	}
}

func (s *wsSynth) reconnectsWithDials() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.calls)
}
