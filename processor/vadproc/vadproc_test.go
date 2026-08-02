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
)

// fakeVAD returns scripted states, one per AnalyzeAudio call.
type fakeVAD struct {
	states []vad.State
	i      int
}

func (f *fakeVAD) SetSampleRate(int) error { return nil }
func (f *fakeVAD) AnalyzeAudio([]byte) vad.State {
	s := f.states[min(f.i, len(f.states)-1)]
	f.i++
	return s
}
func (f *fakeVAD) Params() vad.Params { return vad.DefaultParams() }
func (f *fakeVAD) Reset()             {}
func (f *fakeVAD) Close() error       { return nil }

// runVAD drives a VAD processor with the scripted states (one per 20 ms frame at
// 16 kHz) and returns the ordered names of the VAD frames it emitted.
func runVAD(t *testing.T, states []vad.State, period time.Duration, nframes int) []string {
	t.Helper()
	p := vadproc.New(vadproc.Config{VAD: &fakeVAD{states: states}, SpeechActivityPeriod: period})

	var mu sync.Mutex
	var events []string
	task := pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch f.(type) {
			case *frames.VADUserStartedSpeakingFrame:
				events = append(events, "started")
			case *frames.VADUserStoppedSpeakingFrame:
				events = append(events, "stopped")
			case *frames.UserSpeakingFrame:
				events = append(events, "speaking")
			}
		},
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
	return events
}

func TestVADStartStop(t *testing.T) {
	states := []vad.State{vad.StateQuiet, vad.StateSpeaking, vad.StateSpeaking, vad.StateQuiet}
	got := runVAD(t, states, -1, 4) // keepalive disabled
	assertEvents(t, got, []string{"started", "stopped"})
}

func TestVADPeriodicSpeaking(t *testing.T) {
	// Four speaking frames then quiet; a 40 ms period emits a keepalive every two
	// 20 ms frames.
	states := []vad.State{vad.StateSpeaking, vad.StateSpeaking, vad.StateSpeaking, vad.StateSpeaking, vad.StateQuiet}
	got := runVAD(t, states, 40*time.Millisecond, 5)
	assertEvents(t, got, []string{"started", "speaking", "speaking", "stopped"})
}

// TestVADAudioIdleForcesSpeechStop covers audio that stops arriving mid-speech,
// a muted microphone being the usual case. The detector only ever hears silence
// as speech ending, so without a timeout the user is left speaking for good and
// the turn never closes. Ported from upstream's test of the same behavior.
func TestVADAudioIdleForcesSpeechStop(t *testing.T) {
	const idle = 150 * time.Millisecond
	p := vadproc.New(vadproc.Config{
		VAD:                  &fakeVAD{states: []vad.State{vad.StateSpeaking}},
		SpeechActivityPeriod: -1, // keepalive off, so only the speech events show
		AudioIdleTimeout:     idle,
	})

	stopped := make(chan struct{}, 4)
	task := pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.VADUserStoppedSpeakingFrame); ok {
				stopped <- struct{}{}
			}
		},
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
		VAD:                  &fakeVAD{states: []vad.State{vad.StateQuiet}},
		SpeechActivityPeriod: -1,
		AudioIdleTimeout:     idle,
	})

	stopped := make(chan struct{}, 4)
	task := pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.VADUserStoppedSpeakingFrame); ok {
				stopped <- struct{}{}
			}
		},
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
