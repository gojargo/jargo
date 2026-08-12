package text

import (
	"testing"

	"github.com/gojargo/jargo/frames"
)

// A run that closed but is still sitting in the buffer must not be taken for
// the unclosed one: its opening delimiter, and any delimiter nested inside what
// it matched, is stepped over so the cut lands on the run that really is open.
//
// The buffer is seeded directly because aggregating takes a closed run out as
// soon as it completes, so a flush only ever sees one in the corner cases where
// the scan offset has gone stale.
func TestPatternPairFlushStepsOverAClosedRun(t *testing.T) {
	tok, err := NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	a := NewPatternPairAggregator(frames.AggregationSentence, tok)
	if err := a.AddPattern("test_pattern", "<test>", "</test>", MatchRemove); err != nil {
		t.Fatal(err)
	}

	a.SimpleAggregator.text = "Start <test>closed</test> middle <test>unclosed"

	rest, ok := a.Flush()
	want := "Start <test>closed</test> middle"
	if !ok || rest.Text != want {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, want)
	}
}
