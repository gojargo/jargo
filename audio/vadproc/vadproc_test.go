package vadproc_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/audio/vadproc"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
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
