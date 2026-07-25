package cartesia

import (
	"testing"

	"github.com/gojargo/jargo/service/tts"
)

// timedSynthesizer must satisfy tts.WordTimestamps so the base takes the
// word-aligned path when a caller opts in.
func TestTimedSynthesizerImplementsWordTimestamps(t *testing.T) {
	var _ tts.WordTimestamps = &timedSynthesizer{synthesizer: &synthesizer{}}
}

// A plain Cartesia service must not implement the word-timestamp interface, so
// existing callers keep the unchanged behavior.
func TestPlainConfigDoesNotOptIn(t *testing.T) {
	var s tts.Synthesizer = &synthesizer{cfg: Config{}}
	if _, ok := s.(tts.WordTimestamps); ok {
		t.Fatal("plain synthesizer must not implement tts.WordTimestamps")
	}
}

func TestEmitWordTimingsMergesAndForwards(t *testing.T) {
	in := &wsWordTimings{
		Words: []string{"Hello", ",", " ", "world"},
		Start: []float64{0.0, 0.4, 0.45, 0.5},
	}
	type wt struct {
		word   string
		offset float64
	}
	var got []wt
	err := emitWordTimings(in, func(word string, offset float64) error {
		got = append(got, wt{word, offset})
		return nil
	})
	if err != nil {
		t.Fatalf("emitWordTimings: %v", err)
	}
	want := []wt{{"Hello,", 0.0}, {"world", 0.5}}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestEmitWordTimingsNilIsNoop(t *testing.T) {
	called := false
	if err := emitWordTimings(nil, func(string, float64) error { called = true; return nil }); err != nil {
		t.Fatalf("emitWordTimings(nil): %v", err)
	}
	if called {
		t.Fatal("word callback invoked for nil timings")
	}
}
