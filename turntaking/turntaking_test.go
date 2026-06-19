package turntaking_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/turntaking"
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

// fakeTurn returns scripted AppendAudio and AnalyzeEndOfTurn results.
type fakeTurn struct {
	appends  []turn.EndOfTurnState
	ai       int
	analyses []turn.EndOfTurnState
	ni       int
}

func (f *fakeTurn) SetSampleRate(int) {}
func (f *fakeTurn) AppendAudio([]byte, bool) turn.EndOfTurnState {
	if len(f.appends) == 0 {
		return turn.Incomplete
	}
	s := f.appends[min(f.ai, len(f.appends)-1)]
	f.ai++
	return s
}

func (f *fakeTurn) AnalyzeEndOfTurn() (turn.EndOfTurnState, float64, error) {
	if len(f.analyses) == 0 {
		return turn.Complete, 1, nil
	}
	s := f.analyses[min(f.ni, len(f.analyses)-1)]
	f.ni++
	return s, 1, nil
}
func (f *fakeTurn) UpdateVADStartSecs(float64) {}
func (f *fakeTurn) Clear()                     {}
func (f *fakeTurn) Close() error               { return nil }

// runScenario drives a detector with the given fakes through one frame per VAD
// state and returns the ordered names of the speaking/interruption frames it
// emitted.
func runScenario(t *testing.T, v *fakeVAD, tr turn.Analyzer, nframes int) []string {
	t.Helper()
	d := turntaking.New(turntaking.Config{VAD: v, Turn: tr})

	var mu sync.Mutex
	var events []string
	task := pipeline.NewTask(pipeline.New(d), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			switch f.(type) {
			case *frames.UserStartedSpeakingFrame:
				mu.Lock()
				events = append(events, "started")
				mu.Unlock()
			case *frames.InterruptionFrame:
				mu.Lock()
				events = append(events, "interruption")
				mu.Unlock()
			case *frames.UserStoppedSpeakingFrame:
				mu.Lock()
				events = append(events, "stopped")
				mu.Unlock()
			}
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for range nframes {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
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

func TestCompleteTurn(t *testing.T) {
	v := &fakeVAD{states: []vad.State{vad.StateQuiet, vad.StateSpeaking, vad.StateSpeaking, vad.StateQuiet}}
	tr := &fakeTurn{analyses: []turn.EndOfTurnState{turn.Complete}}

	events := runScenario(t, v, tr, 4)
	want := []string{"started", "interruption", "stopped"}
	assertEvents(t, events, want)
}

func TestMidTurnPauseDoesNotEndTurn(t *testing.T) {
	// Speak, pause (model says incomplete), resume, then stop (model says
	// complete). Only one start/interruption and one stop should be emitted.
	v := &fakeVAD{states: []vad.State{
		vad.StateSpeaking, vad.StateQuiet, vad.StateSpeaking, vad.StateQuiet,
	}}
	tr := &fakeTurn{analyses: []turn.EndOfTurnState{turn.Incomplete, turn.Complete}}

	events := runScenario(t, v, tr, 4)
	want := []string{"started", "interruption", "stopped"}
	assertEvents(t, events, want)
}

func TestSilenceSafetyNetEndsTurn(t *testing.T) {
	// VAD stays speaking, but the turn analyzer's AppendAudio reports complete
	// (its silence safety net), which must end the turn.
	v := &fakeVAD{states: []vad.State{vad.StateSpeaking, vad.StateSpeaking, vad.StateSpeaking}}
	tr := &fakeTurn{appends: []turn.EndOfTurnState{turn.Incomplete, turn.Incomplete, turn.Complete}}

	events := runScenario(t, v, tr, 3)
	want := []string{"started", "interruption", "stopped"}
	assertEvents(t, events, want)
}

func TestVADOnlyMode(t *testing.T) {
	// With no turn analyzer, a VAD stop ends the turn immediately.
	v := &fakeVAD{states: []vad.State{vad.StateSpeaking, vad.StateQuiet}}

	events := runScenario(t, v, nil, 2)
	want := []string{"started", "interruption", "stopped"}
	assertEvents(t, events, want)
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
