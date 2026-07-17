package kyutai

import (
	"testing"

	"github.com/gojargo/jargo/service/stt"
)

func TestPCMRoundTrip(t *testing.T) {
	// int16 -> float32 -> int16 should preserve samples (within rounding).
	in := []int16{0, 1, -1, 32767, -32768, 12345, -12345}
	b := make([]byte, len(in)*2)
	for i, s := range in {
		b[i*2] = byte(uint16(s))
		b[i*2+1] = byte(uint16(s) >> 8)
	}
	f := int16BytesToFloat32(nil, b)
	if len(f) != len(in) {
		t.Fatalf("got %d samples, want %d", len(f), len(in))
	}
	out := float32ToInt16Bytes(f)
	for i, want := range in {
		got := int16(uint16(out[i*2]) | uint16(out[i*2+1])<<8)
		if got-want > 1 || want-got > 1 {
			t.Errorf("sample %d: got %d, want %d", i, got, want)
		}
	}
}

// TestSTTStateMachine drives handle through a two-word utterance ending in a
// semantic-VAD pause, then a second utterance, asserting cumulative interims and
// a single finalized end-of-turn transcript per utterance.
func TestSTTStateMachine(t *testing.T) {
	s := &stream{}

	// Word "je" -> cumulative interim, not final.
	r, ok := s.handle(sttMsg{Type: "Word", Text: "je"})
	if !ok || r.Text != "je" || r.Final {
		t.Fatalf("word1: got %+v ok=%v", r, ok)
	}
	// Word "tombe" -> cumulative interim "je tombe".
	r, ok = s.handle(sttMsg{Type: "Word", Text: "tombe"})
	if !ok || r.Text != "je tombe" || r.Final {
		t.Fatalf("word2: got %+v ok=%v", r, ok)
	}
	// Step with a low pause prediction -> no emit.
	if _, ok := s.handle(sttMsg{Type: "Step", Prs: []float64{0.1, 0.1, 0.2, 0.1}}); ok {
		t.Fatal("low pause prediction should not emit")
	}
	// Step above threshold on the 2 s head -> finalized end-of-turn.
	r, ok = s.handle(sttMsg{Type: "Step", Prs: []float64{0.1, 0.2, 0.9, 0.3}})
	if !ok || r.Text != "je tombe" || !r.Final || !r.EndOfTurn {
		t.Fatalf("pause: got %+v ok=%v", r, ok)
	}
	// A second Step should not re-emit an empty utterance.
	if _, ok := s.handle(sttMsg{Type: "Step", Prs: []float64{0.1, 0.2, 0.9, 0.3}}); ok {
		t.Fatal("pause with empty utterance should not emit")
	}
	// Next word starts a fresh utterance.
	r, ok = s.handle(sttMsg{Type: "Word", Text: "oui"})
	if !ok || r.Text != "oui" {
		t.Fatalf("next utterance: got %+v ok=%v", r, ok)
	}
}

// TestSTTStepBeforeSpeech ensures a pause prediction before any word is ignored.
func TestSTTStepBeforeSpeech(t *testing.T) {
	s := &stream{}
	if _, ok := s.handle(sttMsg{Type: "Step", Prs: []float64{0.9, 0.9, 0.9, 0.9}}); ok {
		t.Fatal("pause before any word should not emit")
	}
}

var _ stt.Result // keep the stt import meaningful if assertions above change
