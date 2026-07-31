package context_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	uctx "github.com/gojargo/jargo/utils/context"
	ttstext "github.com/gojargo/jargo/utils/text"
)

func newSeq(t *testing.T, streaming bool) *uctx.AggregatedFrameSequencer {
	t.Helper()
	tok, err := ttstext.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	return uctx.NewAggregatedFrameSequencer("Test", streaming, tok)
}

func texts(fs []frames.Frame) []string {
	out := make([]string, 0, len(fs))
	for _, f := range fs {
		switch fr := f.(type) {
		case *frames.TTSTextFrame:
			out = append(out, "word:"+fr.Text)
		case *frames.AggregatedTextFrame:
			out = append(out, "agg:"+fr.Text)
		case *frames.AggregatedTextProgressFrame:
			out = append(out, "progress:"+fr.AccumulatedText)
		}
	}
	return out
}

// A word emits its frame and a progress frame reporting how much of the slot
// has been spoken.
func TestSequencerEmitsWordAndProgress(t *testing.T) {
	s := newSeq(t, false)
	frame := frames.NewAggregatedTextFrame("Hello world.", frames.AggregationSentence)
	s.RegisterSpoken(frame, "c1", "Hello world.", true, true, false)

	got := texts(s.ProcessWord("Hello", 100, "c1", false))
	if len(got) != 2 || got[0] != "word:Hello" {
		t.Fatalf("frames = %v, want a word frame then a progress frame", got)
	}
	if got[1] != "progress:Hello" {
		t.Fatalf("progress = %q, want the text spoken so far", got[1])
	}
}

// A skipped frame waits at its position until every spoken slot ahead of it has
// been spoken, so the conversation keeps the order it was written in.
func TestSequencerHoldsSkippedFrameBehindSpokenOne(t *testing.T) {
	s := newSeq(t, false)
	spoken := frames.NewAggregatedTextFrame("Here it is.", frames.AggregationSentence)
	s.RegisterSpoken(spoken, "c1", "Here it is.", true, true, false)

	skipped := frames.NewAggregatedTextFrame("code block", frames.AggregationSentence)
	if got := texts(s.RegisterSkipped(skipped, "c1", "")); len(got) != 0 {
		t.Fatalf("the skipped frame should wait, got %v", got)
	}

	var out []frames.Frame
	for _, w := range []string{"Here", "it", "is."} {
		out = append(out, s.ProcessWord(w, 100, "c1", false)...)
	}
	got := texts(out)
	last := got[len(got)-1]
	if last != "agg:code block" {
		t.Fatalf("frames = %v, want the skipped frame released last", got)
	}
}

// A provider that drops word timings leaves a slot unfinished; ending the
// context emits the unspoken remainder so the conversation is still complete.
func TestSequencerForceCompleteEmitsRemainder(t *testing.T) {
	s := newSeq(t, false)
	frame := frames.NewAggregatedTextFrame("One two three.", frames.AggregationSentence)
	s.RegisterSpoken(frame, "c1", "One two three.", true, true, false)
	s.ProcessWord("One", 100, "c1", false)

	got := texts(s.ForceComplete("c1", 200))
	if len(got) == 0 || got[0] != "word:two three." {
		t.Fatalf("frames = %v, want the unspoken remainder", got)
	}
	// The context is finished, so a word arriving now is stale.
	if late := texts(s.ProcessWord("two", 300, "c1", false)); len(late) != 0 {
		t.Fatalf("a word after the context ended must be dropped, got %v", late)
	}
}

// An interruption drops everything, so no word of the abandoned turn is written.
func TestSequencerClearDropsEverything(t *testing.T) {
	s := newSeq(t, false)
	frame := frames.NewAggregatedTextFrame("Hello world.", frames.AggregationSentence)
	s.RegisterSpoken(frame, "c1", "Hello world.", true, true, false)
	s.Clear()
	if got := texts(s.ProcessWord("Hello", 100, "c1", false)); len(got) != 0 {
		t.Fatalf("frames after clear = %v, want none", got)
	}
}

// Streaming a turn token by token, a sentence becomes a slot only once its
// boundary is confirmed, and the words buffered meanwhile are replayed against
// it rather than lost.
func TestSequencerStreamingPromotesOnBoundary(t *testing.T) {
	s := newSeq(t, true)
	send := func(tok string) []frames.Frame {
		f := frames.NewAggregatedTextFrame(tok, frames.AggregationToken)
		return s.RegisterSpoken(f, "c1", tok, true, true, false)
	}
	if got := texts(send("Hello ")); len(got) != 0 {
		t.Fatalf("no boundary yet, got %v", got)
	}
	if got := texts(send("world.")); len(got) != 0 {
		t.Fatalf("the boundary needs lookahead, got %v", got)
	}
	// A word arrives before its sentence has been promoted.
	if got := texts(s.ProcessWord("Hello", 100, "c1", false)); len(got) != 0 {
		t.Fatalf("the word should be buffered, got %v", got)
	}
	// The next sentence's first non-whitespace character confirms the boundary.
	got := texts(send(" Next"))
	if len(got) == 0 || got[0] != "agg:Hello world." {
		t.Fatalf("frames = %v, want the sentence promoted first", got)
	}
	// The buffered word is replayed against the slot it belongs to.
	joined := strings.Join(got, " ")
	if !strings.Contains(joined, "word:Hello") {
		t.Fatalf("frames = %v, want the buffered word replayed", got)
	}
}
