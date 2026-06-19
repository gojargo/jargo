package vad

import "testing"

// fakeConfidencer returns scripted confidences, one per analysis frame, so the
// state machine can be tested without a model.
type fakeConfidencer struct {
	frameSamples int
	confs        []float64
	i            int
}

func (f *fakeConfidencer) numFramesRequired() int { return f.frameSamples }

func (f *fakeConfidencer) voiceConfidence([]byte) float64 {
	c := f.confs[f.i%len(f.confs)]
	f.i++
	return c
}

// newTestMachine builds a state machine at 16 kHz with frames sized so that one
// frame is 0.1 s: start needs 2 speaking frames (0.2 s), stop needs 3 silent
// frames (0.3 s).
func newTestMachine(confs []float64) *stateMachine {
	fc := &fakeConfidencer{frameSamples: 1600, confs: confs}
	m := newStateMachine(fc, Params{Confidence: 0.7, StartSecs: 0.2, StopSecs: 0.3})
	m.setSampleRate(16000)
	return m
}

const frameBytes = 1600 * 2

func feed(m *stateMachine) State { return m.AnalyzeAudio(make([]byte, frameBytes)) }

func TestStateMachineConfirmsSpeechAfterStartFrames(t *testing.T) {
	m := newTestMachine([]float64{0.9, 0.9, 0.9})

	if got := feed(m); got != StateStarting {
		t.Fatalf("frame 1 state = %v, want starting", got)
	}
	if got := feed(m); got != StateSpeaking {
		t.Fatalf("frame 2 state = %v, want speaking (start needs 2 frames)", got)
	}
	if got := feed(m); got != StateSpeaking {
		t.Fatalf("frame 3 state = %v, want speaking", got)
	}
}

func TestStateMachineConfirmsSilenceAfterStopFrames(t *testing.T) {
	// Two speaking frames confirm speech, then silence must persist 3 frames.
	m := newTestMachine([]float64{0.9, 0.9, 0.1, 0.1, 0.1})

	feed(m) // starting
	if got := feed(m); got != StateSpeaking {
		t.Fatalf("want speaking after 2 frames, got %v", got)
	}
	if got := feed(m); got != StateStopping {
		t.Fatalf("first silent frame state = %v, want stopping", got)
	}
	if got := feed(m); got != StateStopping {
		t.Fatalf("second silent frame state = %v, want stopping", got)
	}
	if got := feed(m); got != StateQuiet {
		t.Fatalf("third silent frame state = %v, want quiet (stop needs 3 frames)", got)
	}
}

func TestStateMachineDebouncesBriefSpeech(t *testing.T) {
	// A single speaking frame followed by silence must not confirm speech.
	m := newTestMachine([]float64{0.9, 0.1})

	if got := feed(m); got != StateStarting {
		t.Fatalf("frame 1 state = %v, want starting", got)
	}
	if got := feed(m); got != StateQuiet {
		t.Fatalf("frame 2 state = %v, want quiet (brief speech debounced)", got)
	}
}

func TestStateMachineBuffersPartialFrames(t *testing.T) {
	m := newTestMachine([]float64{0.9})
	// Less than one frame: no analysis yet, stays quiet.
	if got := m.AnalyzeAudio(make([]byte, frameBytes/2)); got != StateQuiet {
		t.Fatalf("partial frame state = %v, want quiet", got)
	}
	// The remaining half completes one frame and is analyzed.
	if got := m.AnalyzeAudio(make([]byte, frameBytes/2)); got != StateStarting {
		t.Fatalf("completed frame state = %v, want starting", got)
	}
}
