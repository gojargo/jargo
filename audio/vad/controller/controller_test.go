package controller_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/audio/vad/controller"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// idleTimeoutTest is short enough to wait out in a test, and long enough that a
// loaded machine does not trip it between two calls made back to back.
const idleTimeoutTest = 100 * time.Millisecond

// errRateUnsupported stands in for a detector refusing a sample rate.
var errRateUnsupported = errors.New("unsupported sample rate")

// mockAnalyzer is a detector whose verdict the test sets, so a test can walk the
// controller through a sequence of states without feeding it real speech.
type mockAnalyzer struct {
	mu     sync.Mutex
	next   vad.State
	params vad.Params
	rates  []int
	closed int
	// accept is the only rate the detector will take, standing in for one whose
	// model runs at fixed rates. 0 takes whatever it is given.
	accept int
	// analyzed is how many bytes of audio have reached the detector in total, so
	// a test can tell resampled audio from audio passed straight through. It is
	// a total rather than the last chunk because a resampler holds a filter
	// length back between calls, so only the total tracks the rate ratio.
	analyzed int
}

func newMockAnalyzer() *mockAnalyzer {
	return &mockAnalyzer{next: vad.StateQuiet, params: vad.DefaultParams()}
}

// setNextState fixes what the next chunk will be heard as.
func (m *mockAnalyzer) setNextState(s vad.State) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.next = s
}

func (m *mockAnalyzer) SetSampleRate(rate int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rates = append(m.rates, rate)
	if m.accept != 0 && rate != m.accept {
		return errRateUnsupported
	}
	return nil
}

func (m *mockAnalyzer) AnalyzeAudio(buffer []byte) vad.State {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.analyzed += len(buffer)
	return m.next
}

func (m *mockAnalyzer) Params() vad.Params {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.params
}

func (m *mockAnalyzer) SetParams(p vad.Params) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.params = p
}

func (m *mockAnalyzer) Reset() {}

func (m *mockAnalyzer) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed++
	return nil
}

// recorder counts what the controller reported, from whichever goroutine
// reported it: the idle watch runs on its own.
type recorder struct {
	mu         sync.Mutex
	started    int
	stopped    int
	activity   int
	pushed     []pushedFrame
	broadcast  []frames.Frame
	stoppedSig chan struct{}
}

type pushedFrame struct {
	frame frames.Frame
	dir   processor.Direction
}

func newRecorder() *recorder {
	return &recorder{stoppedSig: make(chan struct{}, 4)}
}

func (r *recorder) handlers() controller.Handlers {
	return controller.Handlers{
		OnSpeechStarted: func(context.Context) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.started++
		},
		OnSpeechStopped: func(context.Context) {
			r.mu.Lock()
			r.stopped++
			r.mu.Unlock()
			select {
			case r.stoppedSig <- struct{}{}:
			default:
			}
		},
		OnSpeechActivity: func(context.Context) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.activity++
		},
		OnPushFrame: func(_ context.Context, f frames.Frame, dir processor.Direction) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.pushed = append(r.pushed, pushedFrame{frame: f, dir: dir})
		},
		OnBroadcastFrame: func(_ context.Context, build func() frames.Frame) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.broadcast = append(r.broadcast, build())
		},
	}
}

func (r *recorder) counts() (started, stopped, activity int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.started, r.stopped, r.activity
}

// audioChunk is a chunk of silence; what it holds does not matter, since the
// mock analyzer decides what it was heard as.
func audioChunk() *frames.InputAudioRawFrame {
	return frames.NewInputAudioRawFrame(make([]byte, 1024), 16000, 1)
}

// newTestController builds a controller on the mock analyzer, started, with the
// idle watch torn down when the test ends.
func newTestController(t *testing.T, a vad.Analyzer, r *recorder, cfg controller.Config) *controller.Controller {
	t.Helper()
	c := controller.New(a, r.handlers(), cfg)
	t.Cleanup(c.Cleanup)
	if err := c.ProcessFrame(context.Background(), frames.NewStartFrame()); err != nil {
		t.Fatalf("start: %v", err)
	}
	return c
}

// TestControllerReportsSpeechStarting covers the transition into speech: quiet
// audio reports nothing, and the first chunk heard as speech opens the turn.
func TestControllerReportsSpeechStarting(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	c := newTestController(t, a, r, controller.Config{})
	ctx := context.Background()

	a.setNextState(vad.StateQuiet)
	if err := c.ProcessFrame(ctx, audioChunk()); err != nil {
		t.Fatalf("process quiet audio: %v", err)
	}
	if started, _, _ := r.counts(); started != 0 {
		t.Errorf("quiet audio reported speech starting %d times, want 0", started)
	}

	a.setNextState(vad.StateSpeaking)
	if err := c.ProcessFrame(ctx, audioChunk()); err != nil {
		t.Fatalf("process speech: %v", err)
	}
	if started, _, _ := r.counts(); started != 1 {
		t.Errorf("speech reported starting %d times, want 1", started)
	}
}

// TestControllerReportsSpeechStopping covers the transition out of speech: the
// turn closes on the first chunk heard as quiet, and not before.
func TestControllerReportsSpeechStopping(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	c := newTestController(t, a, r, controller.Config{})
	ctx := context.Background()

	a.setNextState(vad.StateSpeaking)
	if err := c.ProcessFrame(ctx, audioChunk()); err != nil {
		t.Fatalf("process speech: %v", err)
	}
	if _, stopped, _ := r.counts(); stopped != 0 {
		t.Errorf("speech reported stopping %d times while still speaking, want 0", stopped)
	}

	a.setNextState(vad.StateQuiet)
	if err := c.ProcessFrame(ctx, audioChunk()); err != nil {
		t.Fatalf("process quiet audio: %v", err)
	}
	if _, stopped, _ := r.counts(); stopped != 1 {
		t.Errorf("going quiet reported stopping %d times, want 1", stopped)
	}
}

// TestControllerReportsEverySpeechChunk covers the activity report, which fires
// for every chunk heard as speech, the one that opened the turn included. What
// counts on the user still being there is kept fed by it.
func TestControllerReportsEverySpeechChunk(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	c := newTestController(t, a, r, controller.Config{})
	ctx := context.Background()

	a.setNextState(vad.StateSpeaking)
	for range 2 {
		if err := c.ProcessFrame(ctx, audioChunk()); err != nil {
			t.Fatalf("process speech: %v", err)
		}
	}

	if _, _, activity := r.counts(); activity != 2 {
		t.Errorf("two chunks of speech reported %d activity, want 2", activity)
	}
}

// TestControllerIgnoresTransitionalStates covers the two states the detector
// passes through while it makes up its mind. Neither opens nor closes a turn:
// acting on them would report speech that the detector has not confirmed.
func TestControllerIgnoresTransitionalStates(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	c := newTestController(t, a, r, controller.Config{})
	ctx := context.Background()

	for _, state := range []vad.State{vad.StateStarting, vad.StateStopping} {
		a.setNextState(state)
		if err := c.ProcessFrame(ctx, audioChunk()); err != nil {
			t.Fatalf("process transitional audio: %v", err)
		}
	}

	started, stopped, _ := r.counts()
	if started != 0 || stopped != 0 {
		t.Errorf("a transitional state reported %d starts and %d stops, want none", started, stopped)
	}
}

// TestControllerReportsParamsOnStart covers the parameters the controller
// announces as the pipeline starts, so that anything sizing its own behavior to
// the detector (speech recognition matching its endpointing, say) is told what
// the detector is running with.
func TestControllerReportsParamsOnStart(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	newTestController(t, a, r, controller.Config{})

	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.broadcast) != 1 {
		t.Fatalf("starting broadcast %d frames, want 1", len(r.broadcast))
	}
	params, ok := r.broadcast[0].(*frames.SpeechControlParamsFrame)
	if !ok {
		t.Fatalf("broadcast a %T, want a SpeechControlParamsFrame", r.broadcast[0])
	}
	if params.VADParams == nil {
		t.Fatal("the reported parameters carried no detection parameters")
	}
	if *params.VADParams != a.Params() {
		t.Errorf("reported %+v, want the detector's own %+v", *params.VADParams, a.Params())
	}
}

// TestControllerAdoptsNewParams covers a parameter change arriving mid-call: the
// detector takes the new values, and they are announced again so that whatever
// sized itself to the old ones hears about it.
func TestControllerAdoptsNewParams(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	c := newTestController(t, a, r, controller.Config{})

	updated := vad.DefaultParams()
	updated.Confidence = 0.9
	updated.StopSecs = 1.25
	if err := c.ProcessFrame(context.Background(), frames.NewVADParamsUpdateFrame(updated)); err != nil {
		t.Fatalf("update params: %v", err)
	}

	if got := a.Params(); got != updated {
		t.Errorf("the detector is running with %+v, want the updated %+v", got, updated)
	}
	if got := c.Params(); got != updated {
		t.Errorf("the controller reports %+v, want the updated %+v", got, updated)
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	// One from starting, one from the update.
	if len(r.broadcast) != 2 {
		t.Fatalf("broadcast %d frames, want 2: starting and the update", len(r.broadcast))
	}
	params, ok := r.broadcast[1].(*frames.SpeechControlParamsFrame)
	if !ok || params.VADParams == nil {
		t.Fatalf("the update broadcast a %T, want a SpeechControlParamsFrame carrying params", r.broadcast[1])
	}
	if *params.VADParams != updated {
		t.Errorf("announced %+v, want the updated %+v", *params.VADParams, updated)
	}
}

// TestControllerFallsBackWhenTheDetectorRejectsTheRate covers a detector that
// only runs at fixed rates. The input rate is preferred, since matching it needs
// no resampling, but a detector that will not take it has to be given one it
// will rather than left unconfigured.
//
// This has no counterpart upstream, which sets the input rate on the detector
// and does not handle it being refused.
func TestControllerFallsBackWhenTheDetectorRejectsTheRate(t *testing.T) {
	a := newMockAnalyzer()
	a.accept = 16000
	r := newRecorder()

	c := controller.New(a, r.handlers(), controller.Config{})
	t.Cleanup(c.Cleanup)

	start := frames.NewStartFrame()
	start.AudioInSampleRate = 44100
	if err := c.ProcessFrame(context.Background(), start); err != nil {
		t.Fatalf("start at a rate the detector refuses: %v", err)
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	want := []int{44100, 16000}
	if len(a.rates) != len(want) || a.rates[0] != want[0] || a.rates[1] != want[1] {
		t.Errorf("the detector was set to %v, want %v: the input rate first, then the fallback", a.rates, want)
	}
}

// TestControllerResamplesToTheDetectorRate covers the audio path that follows
// from that fallback: once the detector is running at a rate the input does not
// match, every chunk has to be converted before it is analyzed, or the detector
// hears an utterance at the wrong speed.
//
// This has no counterpart upstream, which passes the frame's audio to the
// detector unconverted.
func TestControllerResamplesToTheDetectorRate(t *testing.T) {
	a := newMockAnalyzer()
	a.accept = 16000
	r := newRecorder()

	c := controller.New(a, r.handlers(), controller.Config{})
	t.Cleanup(c.Cleanup)

	start := frames.NewStartFrame()
	start.AudioInSampleRate = 48000
	if err := c.ProcessFrame(context.Background(), start); err != nil {
		t.Fatalf("start: %v", err)
	}

	// Three times the detector's rate in, so a third of the bytes out. Fed as a
	// stream of chunks rather than one: a resampler holds a filter length of
	// audio back for the audio it expects to follow, so the ratio only shows
	// over a stream, which is what a detector is given anyway.
	const (
		chunkBytes = 4800 // 50ms at 48kHz
		chunks     = 20
	)
	sent := 0
	for range chunks {
		chunk := frames.NewInputAudioRawFrame(make([]byte, chunkBytes), 48000, 1)
		if err := c.ProcessFrame(context.Background(), chunk); err != nil {
			t.Fatalf("process audio: %v", err)
		}
		sent += len(chunk.Audio)
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	want := sent / 3
	if a.analyzed < want*9/10 || a.analyzed > want*11/10 {
		t.Errorf("the detector was handed %d bytes of %d sent, want about %d: "+
			"the audio should be converted down to its own rate", a.analyzed, sent, want)
	}
}

// TestControllerPushesFrameThroughItsHost covers the way out of the controller:
// it holds no place in the pipeline itself, so a frame it wants sent goes
// through the handler whatever hosts it supplies.
func TestControllerPushesFrameThroughItsHost(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	c := controller.New(a, r.handlers(), controller.Config{})
	t.Cleanup(c.Cleanup)

	f := audioChunk()
	c.PushFrame(context.Background(), f, processor.Downstream)

	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.pushed) != 1 {
		t.Fatalf("pushed %d frames, want 1", len(r.pushed))
	}
	if r.pushed[0].frame != f {
		t.Error("the frame that came out was not the one pushed")
	}
	if r.pushed[0].dir != processor.Downstream {
		t.Errorf("pushed %v, want downstream", r.pushed[0].dir)
	}
}

// TestControllerBroadcastsFrameThroughItsHost covers the other way out: a frame
// that has to reach both directions is built once per direction by the host.
func TestControllerBroadcastsFrameThroughItsHost(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	c := controller.New(a, r.handlers(), controller.Config{})
	t.Cleanup(c.Cleanup)

	c.BroadcastFrame(context.Background(), func() frames.Frame { return audioChunk() })

	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.broadcast) != 1 {
		t.Fatalf("broadcast %d frames, want 1", len(r.broadcast))
	}
	if _, ok := r.broadcast[0].(*frames.InputAudioRawFrame); !ok {
		t.Errorf("broadcast a %T, want the frame the builder made", r.broadcast[0])
	}
}

// TestControllerEndsSpeechWhenAudioStops covers audio that stops arriving
// mid-utterance, a muted microphone being the usual case. The detector only ever
// hears silence as speech ending, and silence is exactly what it stops being
// given, so without the idle watch the user is left speaking for good and the
// turn never closes.
func TestControllerEndsSpeechWhenAudioStops(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	c := newTestController(t, a, r, controller.Config{AudioIdleTimeout: new(idleTimeoutTest)})

	a.setNextState(vad.StateSpeaking)
	if err := c.ProcessFrame(context.Background(), audioChunk()); err != nil {
		t.Fatalf("process speech: %v", err)
	}
	if _, stopped, _ := r.counts(); stopped != 0 {
		t.Fatalf("speech reported stopping %d times while audio was still arriving, want 0", stopped)
	}

	// Nothing more is fed in, so the watch is what has to close the turn.
	select {
	case <-r.stoppedSig:
	case <-time.After(3 * time.Second):
		t.Fatal("audio stopped arriving mid-speech and the turn was never closed")
	}
}

// TestControllerLeavesQuietAloneWhenAudioStops is the other half of that
// contract: audio stopping while nobody is speaking ends nothing, so a quiet
// line does not produce a turn that was never opened.
func TestControllerLeavesQuietAloneWhenAudioStops(t *testing.T) {
	a := newMockAnalyzer()
	r := newRecorder()
	newTestController(t, a, r, controller.Config{AudioIdleTimeout: new(idleTimeoutTest)})

	// No audio at all, and nobody speaking: the watch has to stay quiet through
	// several windows of it.
	time.Sleep(4 * idleTimeoutTest)

	if _, stopped, _ := r.counts(); stopped != 0 {
		t.Errorf("a quiet line reported speech stopping %d times, want 0", stopped)
	}
}
