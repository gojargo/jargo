package vadproc_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/vadproc"
	"github.com/gojargo/jargo/utils/events"
)

// fakeVAD returns scripted states, one per AnalyzeAudio call.
type fakeVAD struct {
	states []vad.State
	i      int
	params vad.Params
}

func newFakeVAD(states ...vad.State) *fakeVAD {
	return &fakeVAD{states: states, params: vad.DefaultParams()}
}

func (f *fakeVAD) SetSampleRate(int) error { return nil }
func (f *fakeVAD) AnalyzeAudio([]byte) vad.State {
	s := f.states[min(f.i, len(f.states)-1)]
	f.i++
	return s
}
func (f *fakeVAD) Params() vad.Params     { return f.params }
func (f *fakeVAD) SetParams(p vad.Params) { f.params = p }
func (f *fakeVAD) Reset()                 {}
func (f *fakeVAD) Close() error           { return nil }

// runVAD drives a VAD processor with the scripted states (one per 20 ms frame at
// 16 kHz) and returns the ordered names of the VAD frames it emitted.
func runVAD(t *testing.T, states []vad.State, nframes int) []string {
	t.Helper()
	p := vadproc.New(vadproc.Config{VAD: newFakeVAD(states...)})

	var mu sync.Mutex
	var seen []string
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch f.(type) {
		case *frames.VADUserStartedSpeakingFrame:
			seen = append(seen, "started")
		case *frames.VADUserStoppedSpeakingFrame:
			seen = append(seen, "stopped")
		case *frames.UserSpeakingFrame:
			seen = append(seen, "speaking")
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	for range nframes {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1)) // 320 samples = 20 ms
	}
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return seen
}

func TestVADStartStop(t *testing.T) {
	states := []vad.State{vad.StateQuiet, vad.StateSpeaking, vad.StateSpeaking, vad.StateQuiet}
	got := runVAD(t, states, 4)
	// Every chunk heard as speech is reported, the one that started it included.
	assertEvents(t, got, []string{"started", "speaking", "speaking", "stopped"})
}

func TestVADReportsEverySpeakingChunk(t *testing.T) {
	states := []vad.State{vad.StateSpeaking, vad.StateSpeaking, vad.StateSpeaking, vad.StateSpeaking, vad.StateQuiet}
	got := runVAD(t, states, 5)
	assertEvents(t, got, []string{"started", "speaking", "speaking", "speaking", "speaking", "stopped"})
}

// TestVADAudioIdleForcesSpeechStop covers audio that stops arriving mid-speech,
// a muted microphone being the usual case. The detector only ever hears silence
// as speech ending, so without a timeout the user is left speaking for good and
// the turn never closes. Ported from upstream's test of the same behavior.
func TestVADAudioIdleForcesSpeechStop(t *testing.T) {
	const idle = 150 * time.Millisecond
	p := vadproc.New(vadproc.Config{
		VAD:              newFakeVAD(vad.StateSpeaking),
		AudioIdleTimeout: new(idle),
	})

	stopped := make(chan struct{}, 4)
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.VADUserStoppedSpeakingFrame); ok {
			stopped <- struct{}{}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// One frame of speech, and then the audio stops altogether.
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))

	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("the speech never ended after the audio stopped arriving")
	}

	task.StopWhenDone()
	<-runDone
}

// TestVADAudioIdleStaysQuiet is the other half: with the user not speaking, the
// idle timeout has nothing to end and must stay silent. Ported from upstream.
func TestVADAudioIdleStaysQuiet(t *testing.T) {
	const idle = 100 * time.Millisecond
	p := vadproc.New(vadproc.Config{
		VAD:              newFakeVAD(vad.StateQuiet),
		AudioIdleTimeout: new(idle),
	})

	stopped := make(chan struct{}, 4)
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.VADUserStoppedSpeakingFrame); ok {
			stopped <- struct{}{}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))

	select {
	case <-stopped:
		t.Error("the idle timeout ended a speech that had never started")
	case <-time.After(4 * idle):
	}

	task.StopWhenDone()
	<-runDone
}

// TestVADReportsParamsOnStart covers the parameters being reported when the
// pipeline starts, so a processor downstream can size its own behavior to them.
// Ported from upstream's test of the same behavior.
func TestVADReportsParamsOnStart(t *testing.T) {
	p := vadproc.New(vadproc.Config{VAD: newFakeVAD(vad.StateQuiet)})

	got := make(chan *frames.SpeechControlParamsFrame, 4)
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if sc, ok := f.(*frames.SpeechControlParamsFrame); ok {
			got <- sc
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case sc := <-got:
		if sc.VADParams == nil {
			t.Fatal("the reported parameters carried no VAD params")
		}
		if sc.VADParams.StopSecs != vad.DefaultParams().StopSecs {
			t.Errorf("StopSecs = %v, want %v", sc.VADParams.StopSecs, vad.DefaultParams().StopSecs)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the parameters were never reported")
	}

	task.StopWhenDone()
	<-runDone
}

// TestVADParamsUpdateTakesEffect covers changing the detection parameters on a
// running pipeline: the analyzer adopts them, and the new values are reported so
// anything sized to the old ones can follow.
func TestVADParamsUpdateTakesEffect(t *testing.T) {
	fake := newFakeVAD(vad.StateQuiet)
	p := vadproc.New(vadproc.Config{VAD: fake})

	got := make(chan *frames.SpeechControlParamsFrame, 8)
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if sc, ok := f.(*frames.SpeechControlParamsFrame); ok {
			got <- sc
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	<-got // the report made on start

	updated := vad.DefaultParams()
	updated.StopSecs = 1.25
	task.QueueFrame(frames.NewVADParamsUpdateFrame(updated))

	select {
	case sc := <-got:
		if sc.VADParams == nil || sc.VADParams.StopSecs != 1.25 {
			t.Errorf("reported params = %+v, want StopSecs 1.25", sc.VADParams)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the updated parameters were never reported")
	}

	if fake.Params().StopSecs != 1.25 {
		t.Errorf("analyzer StopSecs = %v, want 1.25: the update never reached it", fake.Params().StopSecs)
	}

	task.StopWhenDone()
	<-runDone
}

func assertEvents(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("events = %v, want %v", got, want)
		}
	}
}
