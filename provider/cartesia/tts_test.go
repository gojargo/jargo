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

// The tokens are reported as this provider timed them, with the markup it was
// given stripped back off and anything that was only markup or spacing dropped.
// Punctuation the writing separates from its word stays a token of its own,
// which is what the text says it is.
func TestEmitWordTimingsStripsMarkupAndKeepsTokens(t *testing.T) {
	in := &wsWordTimings{
		Words: []string{"<spell>Hello</spell>", ",", " ", "world"},
		Start: []float64{0.0, 0.4, 0.45, 0.5},
	}
	type wt struct {
		word   string
		offset float64
	}
	var got []wt
	var asked tts.WordTimingOptions
	err := emitWordTimings(in, false, func(words []uctx.WordTiming, opts tts.WordTimingOptions) error {
		asked = opts
		for _, w := range words {
			got = append(got, wt{w.Word, w.Offset})
		}
		return nil
	})
	if err != nil {
		t.Fatalf("emitWordTimings: %v", err)
	}
	if asked.IncludesInterFrameSpaces {
		t.Error("a language written with spaces was reported as carrying its own")
	}
	want := []wt{{"Hello", 0.0}, {",", 0.4}, {"world", 0.5}}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v: the markup comes off and the spacing token is dropped", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d = %v, want %v", i, got[i], want[i])
		}
	}
}

// A language written without spaces has its characters reported separately but
// grouped into one message. The message is the unit a reader recognizes, so it
// is joined into one token at the offset of its first character, and it says it
// carries its own spacing so nothing adds any.
func TestEmitWordTimingsJoinsASpacelessMessage(t *testing.T) {
	in := &wsWordTimings{
		Words: []string{"こ", "ん", "に", "ち", "は", "。"},
		Start: []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5},
	}
	var got []uctx.WordTiming
	var asked tts.WordTimingOptions
	err := emitWordTimings(in, true, func(words []uctx.WordTiming, opts tts.WordTimingOptions) error {
		asked = opts
		got = append(got, words...)
		return nil
	})
	if err != nil {
		t.Fatalf("emitWordTimings: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d tokens, want the message joined into one", len(got))
	}
	if got[0].Word != "こんにちは。" {
		t.Fatalf("token = %q, want the whole message", got[0].Word)
	}
	if got[0].Offset != 0.0 {
		t.Fatalf("offset = %v, want the first character's, 0", got[0].Offset)
	}
	if !asked.IncludesInterFrameSpaces {
		t.Error("the message was reported as needing spacing supplied around it")
	}
}

// A message that was nothing but markup reports nothing.
func TestEmitWordTimingsDropsAMessageOfOnlyMarkup(t *testing.T) {
	in := &wsWordTimings{Words: []string{"<break/>", " "}, Start: []float64{0.0, 0.1}}
	called := false
	err := emitWordTimings(in, false, func([]uctx.WordTiming, tts.WordTimingOptions) error {
		called = true
		return nil
	})
	if err != nil {
		t.Fatalf("emitWordTimings: %v", err)
	}
	if called {
		t.Fatal("a message holding no words was reported")
	}
}

func TestEmitWordTimingsNilIsNoop(t *testing.T) {
	called := false
	emit := func([]uctx.WordTiming, tts.WordTimingOptions) error { called = true; return nil }
	if err := emitWordTimings(nil, false, emit); err != nil {
		t.Fatalf("emitWordTimings(nil): %v", err)
	}
	if called {
		t.Fatal("word callback invoked for nil timings")
	}
}
