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

// Ported from upstream's sequencer suite.
//
// A unit the synthesizer never speaks, a code block held back from it, still
// belongs in the conversation, and it belongs where it was written. It waits at
// its place in the queue until every spoken unit ahead of it has been said.

func spokenFrame(text string) *frames.AggregatedTextFrame {
	return frames.NewAggregatedTextFrame(text, frames.AggregationSentence)
}

func skippedFrame(text string) *frames.AggregatedTextFrame {
	return frames.NewAggregatedTextFrame(text, "code")
}

func TestRegisterSkipped(t *testing.T) {
	t.Run("with nothing ahead of it, it goes at once", func(t *testing.T) {
		seq := newSeq(t, false)
		frame := skippedFrame("code block")
		out := seq.RegisterSkipped(frame, "ctx1", "")
		if len(out) != 1 || out[0] != frames.Frame(frame) {
			t.Fatalf("got %v, want the frame itself, released at once", texts(out))
		}
	})

	t.Run("it is marked as belonging in the conversation", func(t *testing.T) {
		seq := newSeq(t, false)
		frame := skippedFrame("code")
		seq.RegisterSkipped(frame, "ctx1", "")
		if !frame.AppendToContext {
			t.Fatal("the frame was released saying it does not belong in the conversation")
		}
	})

	t.Run("it carries the context it was written in", func(t *testing.T) {
		seq := newSeq(t, false)
		frame := skippedFrame("code")
		seq.RegisterSkipped(frame, "ctx42", "")
		if frame.ContextID != "ctx42" {
			t.Fatalf("frame context = %q, want %q", frame.ContextID, "ctx42")
		}
	})

	t.Run("it carries where it is bound", func(t *testing.T) {
		seq := newSeq(t, false)
		out := seq.RegisterSkipped(skippedFrame("code"), "ctx1", "dest-A")
		if len(out) != 1 {
			t.Fatalf("got %v, want the frame released", texts(out))
		}
		if got := out[0].Base().TransportDestination(); got != "dest-A" {
			t.Fatalf("destination = %q, want %q", got, "dest-A")
		}
	})

	t.Run("speech still to come holds it back", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello world"), "ctx1", "hello world", true, true, false)
		if out := seq.RegisterSkipped(skippedFrame("code"), "ctx2", ""); len(out) != 0 {
			t.Fatalf("got %v, want nothing: the speech before it has not been said", texts(out))
		}
	})

	t.Run("speech already said does not hold it back", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hi"), "ctx1", "hi", true, false, false)
		seq.CompleteSpokenSlot()
		if out := seq.RegisterSkipped(skippedFrame("code"), "ctx2", ""); len(out) != 1 {
			t.Fatalf("got %v, want the frame released", texts(out))
		}
	})

	t.Run("several with no speech between them all go", func(t *testing.T) {
		seq := newSeq(t, false)
		first := seq.RegisterSkipped(skippedFrame("code1"), "ctx1", "")
		second := seq.RegisterSkipped(skippedFrame("code2"), "ctx2", "")
		if len(first) != 1 || len(second) != 1 {
			t.Fatalf("got %v and %v, want each released", texts(first), texts(second))
		}
	})
}

// A service with no word timings completes its slot by saying so, rather than by
// its words arriving. What that unblocks has to be released then.
func TestCompleteSpokenSlot(t *testing.T) {
	t.Run("with nothing queued it does nothing", func(t *testing.T) {
		if out := newSeq(t, false).CompleteSpokenSlot(); len(out) != 0 {
			t.Fatalf("got %v, want nothing", texts(out))
		}
	})

	t.Run("it completes the slot and releases what waited", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, false, false)
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")

		out := seq.CompleteSpokenSlot()
		if len(out) != 1 || out[0] != frames.Frame(skipped) {
			t.Fatalf("got %v, want the held frame released", texts(out))
		}
		if !skipped.AppendToContext {
			t.Fatal("the released frame says it does not belong in the conversation")
		}
	})

	t.Run("it completes only the first slot waiting", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("one"), "ctx1", "one", true, false, false)
		seq.RegisterSpoken(spokenFrame("two"), "ctx2", "two", true, false, false)
		seq.RegisterSkipped(skippedFrame("code"), "ctx3", "")

		if out := seq.CompleteSpokenSlot(); len(out) != 0 {
			t.Fatalf("got %v, want nothing: the second unit has still not been said", texts(out))
		}
	})

	t.Run("what waited goes once all the speech before it has", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("one"), "ctx1", "one", true, false, false)
		seq.RegisterSpoken(spokenFrame("two"), "ctx2", "two", true, false, false)
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx3", "")

		seq.CompleteSpokenSlot()
		out := seq.CompleteSpokenSlot()
		if len(out) != 1 || out[0] != frames.Frame(skipped) {
			t.Fatalf("got %v, want the held frame released", texts(out))
		}
	})
}

// Flushing walks the queue and releases what is no longer held back.
func TestFlush(t *testing.T) {
	t.Run("an empty queue releases nothing", func(t *testing.T) {
		if out := newSeq(t, false).Flush(0); len(out) != 0 {
			t.Fatalf("got %v, want nothing", texts(out))
		}
	})

	t.Run("it stops at speech that has not been said", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, false, false)
		seq.RegisterSkipped(skippedFrame("code"), "ctx2", "")
		if out := seq.Flush(0); len(out) != 0 {
			t.Fatalf("got %v, want nothing", texts(out))
		}
	})

	t.Run("what is released lands after the last word spoken", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, true, false)
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")

		// The word completes the speech ahead of it, which releases it.
		out := seq.ProcessWord("hello", 77, "ctx1", false)
		var released []*frames.AggregatedTextFrame
		for _, f := range out {
			if fr, ok := f.(*frames.AggregatedTextFrame); ok && fr.Text == "code" {
				released = append(released, fr)
			}
		}
		if len(released) != 1 {
			t.Fatalf("got %v, want the held frame released", texts(out))
		}
		if pts, has := released[0].PTS(); !has || pts != 77 {
			t.Fatalf("released at %d (set=%v), want the moment of the last word, 77", pts, has)
		}
	})
}

// seqWithSpoken is a sequencer holding one unit of speech, tracked word by word.
func seqWithSpoken(t *testing.T, text, ctx string, appendToContext bool) *uctx.AggregatedFrameSequencer {
	t.Helper()
	seq := newSeq(t, false)
	seq.RegisterSpoken(spokenFrame(text), ctx, text, appendToContext, true, false)
	return seq
}

// Each word spoken produces the text of that word and a report of how far
// through the unit it is.
func TestProcessWordBasic(t *testing.T) {
	t.Run("a word produces its text and its progress", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello", "ctx1", true)
		out := seq.ProcessWord("hello", 100, "ctx1", false)
		if len(out) != 2 {
			t.Fatalf("got %v, want the word and its progress", texts(out))
		}
		if _, ok := out[0].(*frames.TTSTextFrame); !ok {
			t.Fatalf("first frame is %T, want the word", out[0])
		}
		if _, ok := out[1].(*frames.AggregatedTextProgressFrame); !ok {
			t.Fatalf("second frame is %T, want the progress", out[1])
		}
	})

	t.Run("the word carries its text and the moment it was spoken", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello", "ctx1", true)
		out := seq.ProcessWord("hello", 100, "ctx1", false)
		word, ok := out[0].(*frames.TTSTextFrame)
		if !ok {
			t.Fatalf("first frame is %T, want the word", out[0])
		}
		if word.Text != "hello" {
			t.Fatalf("word = %q, want %q", word.Text, "hello")
		}
		if pts, has := word.PTS(); !has || pts != 100 {
			t.Fatalf("spoken at %d (set=%v), want 100", pts, has)
		}
	})

	t.Run("the word carries the context it was spoken on", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello", "ctx99", true)
		out := seq.ProcessWord("hello", 1, "ctx99", false)
		word, ok := out[0].(*frames.TTSTextFrame)
		if !ok {
			t.Fatalf("first frame is %T, want the word", out[0])
		}
		if got := word.ContextID; got != "ctx99" {
			t.Fatalf("context = %q, want %q", got, "ctx99")
		}
	})

	t.Run("the word says whether it belongs in the conversation", func(t *testing.T) {
		for _, belongs := range []bool{true, false} {
			seq := seqWithSpoken(t, "hello", "ctx1", belongs)
			out := seq.ProcessWord("hello", 1, "ctx1", false)
			word, ok := out[0].(*frames.TTSTextFrame)
			if !ok {
				t.Fatalf("first frame is %T, want the word", out[0])
			}
			if got := word.AppendToContext; got != belongs {
				t.Fatalf("the word says append=%v, want %v", got, belongs)
			}
		}
	})

	t.Run("a word that does not finish the unit releases nothing", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello world", "ctx1", true)
		seq.RegisterSkipped(skippedFrame("code"), "ctx2", "")
		out := seq.ProcessWord("hello", 10, "ctx1", false)
		if len(out) != 2 {
			t.Fatalf("got %v, want only the word and its progress", texts(out))
		}
	})

	t.Run("the word that finishes the unit releases what waited", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello", "ctx1", true)
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")
		out := seq.ProcessWord("hello", 50, "ctx1", false)
		if len(out) != 3 || out[2] != frames.Frame(skipped) {
			t.Fatalf("got %v, want the word, its progress, then the held frame", texts(out))
		}
	})

	t.Run("it is the last word that releases, not an earlier one", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello world", "ctx1", true)
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")
		seq.ProcessWord("hello", 10, "ctx1", false)
		out := seq.ProcessWord("world", 20, "ctx1", false)
		var released bool
		for _, f := range out {
			if f == frames.Frame(skipped) {
				released = true
			}
		}
		if !released {
			t.Fatalf("got %v, want the held frame released on the last word", texts(out))
		}
	})
}

// Which unit a word lands on, and what happens when none claims it.
func TestProcessWordContextRouting(t *testing.T) {
	t.Run("a word on no context at all passes through", func(t *testing.T) {
		// A service with no audio contexts names none, and its words are still
		// the turn's words.
		seq := newSeq(t, false)
		out := seq.ProcessWord("hello", 1, "", false)
		if len(out) != 1 {
			t.Fatalf("got %v, want the word alone", texts(out))
		}
		word, ok := out[0].(*frames.TTSTextFrame)
		if !ok {
			t.Fatalf("first frame is %T, want the word", out[0])
		}
		if word.Text != "hello" || word.ContextID != "" {
			t.Fatalf("word = %q on context %q, want %q on none", word.Text, word.ContextID, "hello")
		}
		if !word.AppendToContext {
			t.Fatal("the word says it does not belong in the conversation")
		}
	})

	t.Run("a word on a context nobody registered is dropped", func(t *testing.T) {
		// A provider can deliver timings long after its context was abandoned.
		// Emitting them would interleave them into the turn now being spoken.
		seq := newSeq(t, false)
		if out := seq.ProcessWord("hello", 1, "ctx-unknown", false); len(out) != 0 {
			t.Fatalf("got %v, want nothing: the context is not one this turn knows", texts(out))
		}
	})

	t.Run("a word no unit claims passes through", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello world"), "ctx1", "hello world", true, true, false)
		out := seq.ProcessWord("zzz", 5, "ctx1", false)
		passed, ok := out[0].(*frames.TTSTextFrame)
		if len(out) != 1 || !ok || passed.Text != "zzz" {
			t.Fatalf("got %v, want the word passed through", texts(out))
		}
	})

	t.Run("an unnamed word still lands on the unit waiting for it", func(t *testing.T) {
		// A provider may register its units under a context and then report the
		// words without naming one. They are still that unit's words.
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, true, false)
		out := seq.ProcessWord("hello", 1, "", false)
		var words []*frames.TTSTextFrame
		for _, f := range out {
			if w, ok := f.(*frames.TTSTextFrame); ok {
				words = append(words, w)
			}
		}
		if len(words) != 1 || words[0].Text != "hello" {
			t.Fatalf("got %v, want the one word", texts(out))
		}
		if words[0].ContextID != "ctx1" {
			t.Fatalf("the word names context %q, want the unit's own %q", words[0].ContextID, "ctx1")
		}
	})
}
