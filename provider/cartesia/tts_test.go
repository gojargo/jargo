package cartesia

import (
	"testing"

	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
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

// TestEmitWordTimingsForwardsRawAndAsksForMerging checks the provider hands its
// tokens over as they came and asks for them to be normalized, rather than
// normalizing them itself. This API splits punctuation and spaces into tokens of
// their own, so the merging matters; doing it here rather than in the base is
// what left other providers reporting tokens nothing downstream expects.
func TestEmitWordTimingsForwardsRawAndAsksForMerging(t *testing.T) {
	in := &wsWordTimings{
		Words: []string{"Hello", ",", " ", "world"},
		Start: []float64{0.0, 0.4, 0.45, 0.5},
	}
	type wt struct {
		word   string
		offset float64
	}
	var got []wt
	var asked tts.WordTimingOptions
	err := emitWordTimings(in, func(words []uctx.WordTiming, opts tts.WordTimingOptions) error {
		asked = opts
		for _, w := range words {
			got = append(got, wt{w.Word, w.Offset})
		}
		return nil
	})
	if err != nil {
		t.Fatalf("emitWordTimings: %v", err)
	}
	if !asked.PreMergeTokens {
		t.Error("the provider did not ask for its punctuation tokens to be merged")
	}
	// Forwarded as they came: the merging happens in the base.
	want := []wt{{"Hello", 0.0}, {",", 0.4}, {" ", 0.45}, {"world", 0.5}}
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
	emit := func([]uctx.WordTiming, tts.WordTimingOptions) error { called = true; return nil }
	if err := emitWordTimings(nil, emit); err != nil {
		t.Fatalf("emitWordTimings(nil): %v", err)
	}
	if called {
		t.Fatal("word callback invoked for nil timings")
	}
}
