package text_test

import (
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/text"
)

func newAggregator(t *testing.T, by frames.AggregationType) *text.SimpleAggregator {
	t.Helper()
	tok, err := text.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	return text.NewSimpleAggregator(by, tok)
}

// A sentence is only completed once a non-whitespace character confirms the
// boundary, so text that merely ends on a period stays buffered.
func TestSimpleAggregatorWaitsForLookahead(t *testing.T) {
	a := newAggregator(t, frames.AggregationSentence)

	if got := a.Aggregate("Hello world."); len(got) != 0 {
		t.Fatalf("a trailing period should wait for lookahead, got %+v", got)
	}
	if got := a.Aggregate(" "); len(got) != 0 {
		t.Fatalf("whitespace is not lookahead, got %+v", got)
	}
	got := a.Aggregate("A")
	if len(got) != 1 || got[0].Text != "Hello world." {
		t.Fatalf("aggregations = %+v, want one sentence %q", got, "Hello world.")
	}
	if got[0].Type != frames.AggregationSentence {
		t.Fatalf("type = %q, want %q", got[0].Type, frames.AggregationSentence)
	}
}

// A period inside an amount is not a boundary, which is what the lookahead is
// there to establish.
func TestSimpleAggregatorKeepsAmountsWhole(t *testing.T) {
	a := newAggregator(t, frames.AggregationSentence)
	got := a.Aggregate("Your balance is $29.50 today. Next")
	if len(got) != 1 || got[0].Text != "Your balance is $29.50 today." {
		t.Fatalf("aggregations = %+v, want the amount kept whole", got)
	}
}

// One chunk carrying several boundaries completes several sentences.
func TestSimpleAggregatorCompletesSeveralSentences(t *testing.T) {
	a := newAggregator(t, frames.AggregationSentence)
	got := a.Aggregate("One. Two. Three. X")
	if len(got) != 3 {
		t.Fatalf("aggregations = %+v, want three sentences", got)
	}
	for i, want := range []string{"One.", "Two.", "Three."} {
		if got[i].Text != want {
			t.Fatalf("sentence %d = %q, want %q", i, got[i].Text, want)
		}
	}
}

// Text that never terminates is returned by the flush at the end of a response.
func TestSimpleAggregatorFlushesTrailingText(t *testing.T) {
	a := newAggregator(t, frames.AggregationSentence)
	if got := a.Aggregate("no period here"); len(got) != 0 {
		t.Fatalf("aggregations = %+v, want none", got)
	}
	rest, ok := a.Flush()
	if !ok || rest.Text != "no period here" {
		t.Fatalf("flush = %+v (%v), want the buffered text", rest, ok)
	}
	if _, ok := a.Flush(); ok {
		t.Fatal("a second flush should find nothing")
	}
}

// Aggregating by token buffers nothing: each chunk goes straight through.
func TestSimpleAggregatorTokenModePassesThrough(t *testing.T) {
	a := newAggregator(t, frames.AggregationToken)
	got := a.Aggregate("Hel")
	if len(got) != 1 || got[0].Text != "Hel" || got[0].Type != frames.AggregationToken {
		t.Fatalf("aggregations = %+v, want the token as-is", got)
	}
	if _, ok := a.Flush(); ok {
		t.Fatal("token aggregation buffers nothing to flush")
	}
}

// The aggregator reports how it was built and what it is still holding, which
// is how a caller running channels of text in parallel tells whether its own
// buffer still mirrors this one.
func TestSimpleAggregatorReportsItsState(t *testing.T) {
	a := newAggregator(t, frames.AggregationSentence)
	if a.Type() != frames.AggregationSentence {
		t.Errorf("Type() = %v, want sentence aggregation", a.Type())
	}

	if a.Buffer() != "" {
		t.Errorf("a fresh aggregator buffers %q, want nothing", a.Buffer())
	}

	// Partial text stays buffered, with the raw text kept untrimmed.
	a.Aggregate("  Hello wor")
	if got := a.Buffer(); got != "  Hello wor" {
		t.Errorf("Buffer() = %q, want the raw text", got)
	}
	if got := a.Text(); got.Text != "Hello wor" {
		t.Errorf("Text() = %q, want it trimmed", got.Text)
	}
	if got := a.Text(); got.Type != frames.AggregationSentence {
		t.Errorf("Text() type = %v, want sentence aggregation", got.Type)
	}

	a.Reset()
	if a.Buffer() != "" {
		t.Errorf("Buffer() = %q after a reset, want nothing", a.Buffer())
	}
}

// An interruption discards what was half-gathered, the pending lookahead
// included: a mark waiting to be confirmed belongs to the turn that was cut off,
// so the next turn must not be confirmed by the character that follows it.
func TestSimpleAggregatorHandleInterruptionDropsWhatWasGathered(t *testing.T) {
	a := newAggregator(t, frames.AggregationSentence)

	if got := a.Aggregate("Hello world."); len(got) != 0 {
		t.Fatalf("a trailing period should wait for lookahead, got %+v", got)
	}
	a.HandleInterruption()
	if a.Buffer() != "" {
		t.Errorf("Buffer() = %q after an interruption, want nothing", a.Buffer())
	}

	// The lookahead the period was waiting for is gone with it, so this text
	// completes nothing until it ends a sentence of its own.
	if got := a.Aggregate("Next"); len(got) != 0 {
		t.Fatalf("a pending lookahead survived the interruption, got %+v", got)
	}
	if got := a.Buffer(); got != "Next" {
		t.Errorf("Buffer() = %q, want the new turn's text alone", got)
	}
}
