package vad

import (
	"encoding/binary"
	"math"
	"testing"
)

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

// loudFrame is one analysis frame of a 440 Hz tone at the given amplitude, loud
// enough at 8000 to carry well past the volume gate.
func loudFrame(amp float64) []byte {
	pcm := make([]byte, frameBytes)
	for i := range frameBytes / 2 {
		t := float64(i) / 16000
		s := int16(amp * math.Sin(2*math.Pi*440*t))
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(s))
	}
	return pcm
}

// TestStateMachineGatesOnVolume covers the volume gate: a model confident about
// something barely audible must not open a turn, because a confident guess at
// near-silence is how a room's own noise gets mistaken for speech.
func TestStateMachineGatesOnVolume(t *testing.T) {
	confident := func() *stateMachine {
		fc := &fakeConfidencer{frameSamples: 1600, confs: []float64{1.0}}
		m := newStateMachine(fc, Params{Confidence: 0.7, StartSecs: 0.2, StopSecs: 0.3, MinVolume: 0.6})
		m.setSampleRate(16000)
		return m
	}

	t.Run("silence stays quiet however confident", func(t *testing.T) {
		m := confident()
		for i := range 5 {
			if got := m.AnalyzeAudio(make([]byte, frameBytes)); got != StateQuiet {
				t.Fatalf("frame %d state = %v, want quiet: silence passed the volume gate", i+1, got)
			}
		}
	})

	t.Run("audible speech gets through", func(t *testing.T) {
		m := confident()
		frame := loudFrame(8000)

		// Each reading is smoothed against the one before it, starting from
		// silence, so the measured volume climbs toward the tone's rather than
		// arriving at it. Speech is confirmed once it has climbed past the gate
		// and held there for StartSecs.
		var frames int
		for range 20 {
			frames++
			if m.AnalyzeAudio(frame) == StateSpeaking {
				break
			}
		}
		if m.state != StateSpeaking {
			t.Fatalf("state = %v after 20 frames, want speaking: audible speech was gated out", m.state)
		}
		if frames < 3 {
			t.Errorf("speech confirmed after %d frames; the volume gate cannot have been applied", frames)
		}
		t.Logf("speech confirmed after %d frames of 0.1 s", frames)
	})

	t.Run("the gate can be turned off", func(t *testing.T) {
		fc := &fakeConfidencer{frameSamples: 1600, confs: []float64{1.0}}
		m := newStateMachine(fc, Params{Confidence: 0.7, StartSecs: 0.2, StopSecs: 0.3})
		m.setSampleRate(16000)
		for range 3 {
			m.AnalyzeAudio(make([]byte, frameBytes))
		}
		if m.state != StateSpeaking {
			t.Fatalf("state = %v, want speaking with no minimum volume set", m.state)
		}
	})
}
