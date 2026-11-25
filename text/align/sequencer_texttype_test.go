package align_test

import (
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/text/align"
)

// promotedType streams text of the given type into a fresh streaming sequencer
// and returns the text type of the sentence it promotes.
func promotedType(t *testing.T, typ frames.TextType, buildTracker bool) []frames.TextType {
	t.Helper()
	s := align.NewAggregatedFrameSequencer("Test", true)
	out := s.RegisterSpoken(frames.NewAggregatedTextFrame("One moment.", typ), "c1", "One moment.",
		false, buildTracker, false, "")
	out = append(out, s.Finalize("c1")...)
	var types []frames.TextType
	for _, f := range out {
		switch fr := f.(type) {
		case *frames.TTSTextFrame:
			types = append(types, fr.TextType)
		case *frames.AggregatedTextFrame:
			types = append(types, fr.TextType)
		}
	}
	return types
}

// TestAPromotedSentenceKeepsItsTextType checks a sentence promoted from
// streamed text keeps the type the text was given with, such as a
// TTSSpeakFrame's, with and without word timestamps.
func TestAPromotedSentenceKeepsItsTextType(t *testing.T) {
	for _, tracker := range []bool{true, false} {
		types := promotedType(t, "status", tracker)
		if len(types) == 0 {
			t.Fatalf("tracker %t: nothing was promoted", tracker)
		}
		for _, typ := range types {
			if typ != "status" {
				t.Errorf("tracker %t: text type = %q, want status", tracker, typ)
			}
		}
	}
}

// TestPromotedTokensAreASentence checks streamed tokens are promoted as a
// sentence.
func TestPromotedTokensAreASentence(t *testing.T) {
	for _, typ := range promotedType(t, frames.TextTypeToken, true) {
		if typ != frames.TextTypeSentence {
			t.Errorf("text type = %q, want sentence", typ)
		}
	}
}
