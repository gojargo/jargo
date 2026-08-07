package context_test

import (
	"slices"
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

// wordFrames picks the word frames out of what a call returned.
func wordFrames(fs []frames.Frame) []*frames.TTSTextFrame {
	var out []*frames.TTSTextFrame
	for _, f := range fs {
		if w, ok := f.(*frames.TTSTextFrame); ok {
			out = append(out, w)
		}
	}
	return out
}

// contains reports whether want is among fs.
func contains(fs []frames.Frame, want frames.Frame) bool {
	return slices.Contains(fs, want)
}

// Each word carries the span of written text it stands for, so what reaches the
// conversation is what the model wrote rather than what the synthesizer said.
func TestProcessWordRawText(t *testing.T) {
	t.Run("the written span is split across the words", func(t *testing.T) {
		seq := newSeq(t, false)
		frame := spokenFrame("4111 1111")
		frame.RawText = "<card>4111 1111</card>"
		seq.RegisterSpoken(frame, "ctx1", "4111 1111", true, true, false)

		first := seq.ProcessWord("4111", 10, "ctx1", false)
		second := seq.ProcessWord("1111", 20, "ctx1", false)

		if got := wordFrames(first)[0].RawText; got != "<card>4111" {
			t.Fatalf("the first word stands for %q, want the opening tag with it", got)
		}
		if got := wordFrames(second)[0].RawText; got != "1111</card>" {
			t.Fatalf("the last word stands for %q, want the closing tag with it", got)
		}
	})

	t.Run("with nothing else written, a word stands for itself", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello", "ctx1", true)
		out := seq.ProcessWord("hello", 1, "ctx1", false)
		if got := wordFrames(out)[0].RawText; got != "hello" {
			t.Fatalf("the word stands for %q, want %q", got, "hello")
		}
	})
}

// One token can span two units of a turn, since the sentences of a turn share a
// context. It is split between them, and each half lands on its own unit.
func TestProcessWordOverflow(t *testing.T) {
	twoUnits := func(t *testing.T, first, second string) *uctx.AggregatedFrameSequencer {
		t.Helper()
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame(first), "ctx1", first, true, true, false)
		seq.RegisterSpoken(spokenFrame(second), "ctx1", second, true, true, false)
		return seq
	}

	t.Run("it produces a word for each unit", func(t *testing.T) {
		seq := twoUnits(t, "abc", "def")
		got := wordFrames(seq.ProcessWord("abcdef", 100, "ctx1", false))
		if len(got) != 2 || got[0].Text != "abc" || got[1].Text != "def" {
			t.Fatalf("got %v, want a word for each unit", texts(seq.ProcessWord("", 0, "ctx1", false)))
		}
	})

	t.Run("both halves name the context they were spoken on", func(t *testing.T) {
		seq := twoUnits(t, "abc", "def")
		for i, w := range wordFrames(seq.ProcessWord("abcdef", 100, "ctx1", false)) {
			if w.ContextID != "ctx1" {
				t.Fatalf("half %d names context %q, want %q", i+1, w.ContextID, "ctx1")
			}
		}
	})

	t.Run("the half that finishes the second unit releases what waited", func(t *testing.T) {
		seq := twoUnits(t, "abc", "def")
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")
		if !contains(seq.ProcessWord("abcdef", 100, "ctx1", false), skipped) {
			t.Fatal("the held frame was not released once both units had been said")
		}
	})

	t.Run("a half that leaves the second unit unfinished releases nothing", func(t *testing.T) {
		seq := twoUnits(t, "abc", "def ghi")
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")
		if contains(seq.ProcessWord("abcdef", 100, "ctx1", false), skipped) {
			t.Fatal("the held frame was released with a word of the second unit still unsaid")
		}
	})
}

// A word belonging to the unit after this one means the synthesizer dropped what
// was in between. The unit it skipped is closed out and the word goes where it
// belongs.
func TestProcessWordForcesComplete(t *testing.T) {
	t.Run("a word for the next unit closes this one out", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, true, false)
		seq.RegisterSpoken(spokenFrame("world"), "ctx1", "world", true, true, false)

		var said []string
		for _, w := range wordFrames(seq.ProcessWord("world", 50, "ctx1", false)) {
			said = append(said, w.Text)
		}
		if !slices.Contains(said, "world") {
			t.Fatalf("got %v, want the word among them", said)
		}
	})

	t.Run("closing out and finishing the next releases what waited", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, true, false)
		seq.RegisterSpoken(spokenFrame("world"), "ctx1", "world", true, true, false)
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")

		if !contains(seq.ProcessWord("world", 50, "ctx1", false), skipped) {
			t.Fatal("the held frame was not released")
		}
	})

	t.Run("a unit of nothing but space emits no word of its own", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame(" "), "ctx1", " ", true, true, false)
		seq.RegisterSpoken(spokenFrame("World"), "ctx1", "World", true, true, false)

		got := wordFrames(seq.ProcessWord("World", 10, "ctx1", false))
		if len(got) != 1 {
			var said []string
			for _, w := range got {
				said = append(said, w.Text)
			}
			t.Fatalf("got %v, want one word: a unit holding only space has nothing to say", said)
		}
		if got[0].Text != "World" || got[0].ContextID != "ctx1" {
			t.Fatalf("word = %q on %q, want %q on %q", got[0].Text, got[0].ContextID, "World", "ctx1")
		}
	})
}

// A context whose audio has ended closes out whatever of it went unspoken, for a
// synthesizer that drops word timings, so the conversation still records the
// text and whatever waited behind it is released.
func TestForceComplete(t *testing.T) {
	t.Run("what went unspoken is emitted", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello world", "ctx1", true)
		seq.ProcessWord("hello", 10, "ctx1", false) // "world" never arrives

		got := wordFrames(seq.ForceComplete("ctx1", 50))
		if len(got) != 1 || got[0].Text != "world" {
			t.Fatalf("got %v, want the unspoken remainder", texts(seq.ForceComplete("ctx1", 50)))
		}
		if pts, has := got[0].PTS(); !has || pts != 50 {
			t.Fatalf("emitted at %d (set=%v), want the last word's moment, 50", pts, has)
		}
	})

	t.Run("with nothing spoken at all, the whole unit is emitted", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello world", "ctx1", true)
		got := wordFrames(seq.ForceComplete("ctx1", 0))
		if len(got) != 1 || got[0].Text != "hello world" {
			t.Fatal("the whole unit was not emitted")
		}
	})

	t.Run("a unit already said emits nothing", func(t *testing.T) {
		seq := seqWithSpoken(t, "hi", "ctx1", true)
		seq.ProcessWord("hi", 5, "ctx1", false)
		if out := seq.ForceComplete("ctx1", 10); len(out) != 0 {
			t.Fatalf("got %v, want nothing: the unit had been said in full", texts(out))
		}
	})

	t.Run("what waited behind it is released", func(t *testing.T) {
		seq := seqWithSpoken(t, "hello", "ctx1", true)
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")

		out := seq.ForceComplete("ctx1", 20)
		if !contains(out, skipped) {
			t.Fatal("the held frame was not released")
		}
		if !skipped.AppendToContext {
			t.Fatal("the released frame says it does not belong in the conversation")
		}
	})

	t.Run("the unspoken remainder carries its written form", func(t *testing.T) {
		seq := newSeq(t, false)
		frame := spokenFrame("4111 1111")
		frame.RawText = "<card>4111 1111</card>"
		seq.RegisterSpoken(frame, "ctx1", "4111 1111", true, true, false)
		seq.ProcessWord("4111", 10, "ctx1", false) // "1111" never arrives

		got := wordFrames(seq.ForceComplete("ctx1", 20))
		if len(got) == 0 || got[0].Text != "1111" {
			t.Fatal("the unspoken remainder was not emitted")
		}
		if got[0].RawText != "1111</card>" {
			t.Fatalf("it stands for %q, want the closing tag with it", got[0].RawText)
		}
	})

	t.Run("a written form that disagrees with the text is dropped", func(t *testing.T) {
		seq := newSeq(t, false)
		frame := spokenFrame("abc")
		frame.RawText = "xyz"
		seq.RegisterSpoken(frame, "ctx1", "abc", true, true, false)

		got := wordFrames(seq.ForceComplete("ctx1", 0))
		if len(got) != 1 || got[0].Text != "abc" {
			t.Fatal("the unit was not emitted")
		}
		if got[0].RawText != "" {
			t.Fatalf("it stands for %q, want nothing: the two texts disagree, so the "+
				"written one cannot be trusted", got[0].RawText)
		}
	})

	t.Run("a unit with no word tracking is simply completed", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, false, false)
		skipped := skippedFrame("code")
		seq.RegisterSkipped(skipped, "ctx2", "")

		out := seq.ForceComplete("ctx1", 0)
		if got := wordFrames(out); len(got) != 0 {
			t.Fatalf("got %v, want no words: this unit was never tracked word by word", texts(out))
		}
		if !contains(out, skipped) {
			t.Fatal("the held frame was not released")
		}
	})

	t.Run("it closes out only the context it names", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, true, false)
		seq.RegisterSpoken(spokenFrame("world"), "ctx2", "world", true, true, false)

		first := wordFrames(seq.ForceComplete("ctx1", 0))
		if len(first) != 1 || first[0].Text != "hello" {
			t.Fatal("closing one context out did not emit its own text alone")
		}
		// The other is still in flight and closes out on its own afterwards.
		second := wordFrames(seq.ForceComplete("ctx2", 10))
		if len(second) != 1 || second[0].Text != "world" {
			t.Fatal("the other context was left unable to close itself out")
		}
	})

	t.Run("a word arriving afterwards is dropped", func(t *testing.T) {
		// Some providers deliver timings seconds late. Emitting one now would
		// interleave it into the turn being spoken.
		seq := seqWithSpoken(t, "hello", "ctx1", true)
		seq.ForceComplete("ctx1", 0)
		if out := seq.ProcessWord("hello", 5, "ctx1", false); len(out) != 0 {
			t.Fatalf("got %v, want nothing: the context is closed", texts(out))
		}
	})

	t.Run("closing the same context out twice emits nothing the second time", func(t *testing.T) {
		// The service closes a context out on its stop frame and again when its
		// audio has drained.
		seq := seqWithSpoken(t, "hello", "ctx1", true)
		first := wordFrames(seq.ForceComplete("ctx1", 0))
		if len(first) != 1 || first[0].Text != "hello" {
			t.Fatal("the first close did not emit the unspoken text")
		}
		if out := seq.ForceComplete("ctx1", 10); len(out) != 0 {
			t.Fatalf("got %v, want nothing the second time", texts(out))
		}
	})
}

// stream feeds tokens through a streaming sequencer and returns what they
// pushed. A token whose text is the same on both sides stands for a turn with
// no transform applied to it.
func stream(t *testing.T, seq *uctx.AggregatedFrameSequencer, ctx string, tokens ...string) []frames.Frame {
	t.Helper()
	var out []frames.Frame
	for _, tok := range tokens {
		out = append(out, seq.RegisterSpoken(spokenFrame(tok), ctx, tok, true, true, false)...)
	}
	return out
}

// aggFrames picks the aggregated-text frames out of what a call returned. A
// promoted sentence announces itself with one, so their absence is how a test
// says nothing was promoted.
func aggFrames(fs []frames.Frame) []*frames.AggregatedTextFrame {
	var out []*frames.AggregatedTextFrame
	for _, f := range fs {
		if a, ok := f.(*frames.AggregatedTextFrame); ok {
			out = append(out, a)
		}
	}
	return out
}

// A turn streamed token by token is put back together into sentences: a slot
// appears only once a boundary is confirmed, so the conversation records
// sentences rather than the arbitrary pieces the model happened to send.
func TestRegisterSpokenStreaming(t *testing.T) {
	t.Run("tokens mid-sentence promote nothing", func(t *testing.T) {
		seq := newSeq(t, true)
		if out := stream(t, seq, "ctx1", "Hi", " there"); len(aggFrames(out)) != 0 {
			t.Fatalf("got %v, want nothing promoted mid-sentence", texts(out))
		}
	})

	t.Run("a terminal token alone promotes nothing", func(t *testing.T) {
		// The token after it is what confirms the boundary.
		seq := newSeq(t, true)
		if out := stream(t, seq, "ctx1", "Hi", " there", "!"); len(aggFrames(out)) != 0 {
			t.Fatalf("got %v, want nothing promoted until the boundary is confirmed", texts(out))
		}
	})

	t.Run("the next sentence's first token promotes the one before it", func(t *testing.T) {
		seq := newSeq(t, true)
		out := stream(t, seq, "ctx1", "Hi", " there", "!", " How")
		got := aggFrames(out)
		if len(got) != 1 || got[0].Text != "Hi there!" {
			t.Fatalf("got %v, want the first sentence promoted whole, without the token "+
				"that confirmed it", texts(out))
		}
		if got[0].AggregatedBy != frames.AggregationSentence {
			t.Fatalf("promoted as %q, want a sentence", got[0].AggregatedBy)
		}
	})

	t.Run("a promoted sentence takes its words normally", func(t *testing.T) {
		seq := newSeq(t, true)
		stream(t, seq, "ctx1", "Hi", " there", "!", " How")

		out := seq.ProcessWord("Hi", 10, "ctx1", false)
		out = append(out, seq.ProcessWord("there!", 20, "ctx1", false)...)

		var said []string
		for _, w := range wordFrames(out) {
			said = append(said, w.Text)
		}
		if len(said) != 2 || said[0] != "Hi" || said[1] != "there!" {
			t.Fatalf("said %v, want the sentence's own words", said)
		}
		var progress []*frames.AggregatedTextProgressFrame
		for _, f := range out {
			if p, ok := f.(*frames.AggregatedTextProgressFrame); ok {
				progress = append(progress, p)
			}
		}
		if len(progress) == 0 {
			t.Fatal("no progress was reported")
		}
		last := progress[len(progress)-1]
		if last.AccumulatedText != "Hi there!" || last.RemainingText != "" {
			t.Fatalf("progress ends at %q with %q left, want the whole sentence said",
				last.AccumulatedText, last.RemainingText)
		}
	})

	t.Run("the tokens' own spacing does not follow them into the words", func(t *testing.T) {
		// A model's tokens carry their spacing; the words a synthesizer reports
		// for them do not, and joining the words as though they did would run
		// them together.
		seq := newSeq(t, true)
		stream(t, seq, "ctx1", "Hi", " there", "!", " How")

		out := seq.ProcessWord("Hi", 10, "ctx1", false)
		out = append(out, seq.ProcessWord("there!", 20, "ctx1", false)...)

		var parts []ttstext.Part
		for _, w := range wordFrames(out) {
			parts = append(parts, ttstext.Part{
				Text:                    w.Text,
				IncludesInterPartSpaces: w.IncludesInterFrameSpaces,
			})
		}
		if got := ttstext.Concatenate(parts); got != "Hi there!" {
			t.Fatalf("the words assemble to %q, want %q", got, "Hi there!")
		}
	})

	t.Run("each sentence promotes on its own, the last on finalizing", func(t *testing.T) {
		seq := newSeq(t, true)
		out := stream(t, seq, "ctx1", "Hi", " there", "!", " How", " are", " you", "?")
		var promoted []string
		for _, a := range aggFrames(out) {
			promoted = append(promoted, a.Text)
		}
		if len(promoted) != 1 || promoted[0] != "Hi there!" {
			t.Fatalf("promoted %v, want only the first sentence so far", promoted)
		}

		for _, a := range aggFrames(seq.Finalize("ctx1")) {
			promoted = append(promoted, a.Text)
		}
		if len(promoted) != 2 || promoted[1] != " How are you?" {
			t.Fatalf("promoted %v, want the trailing sentence on finalizing", promoted)
		}
	})

	t.Run("a transform survives the promotion", func(t *testing.T) {
		seq := newSeq(t, true)
		// What is spoken differs from what was written: "$5" said as "five dollars".
		for _, tok := range []struct{ spoken, written string }{
			{"five dollars", "$5"}, {".", "."}, {" Ok", " Ok"},
		} {
			seq.RegisterSpoken(spokenFrame(tok.written), "ctx1", tok.spoken, true, true, false)
		}
		// The promoted sentence carries the written form.
		out := seq.ProcessWord("five", 10, "ctx1", false)
		if got := wordFrames(out); len(got) == 0 || got[0].Text != "five" {
			t.Fatal("the promoted sentence did not take the spoken word")
		}
	})
}

// A word can arrive before the sentence it belongs to has been recognized, since
// a turn streamed token by token has no sentence until a boundary confirms one.
// Such a word waits rather than being lost or emitted on its own.
func TestRegisterSpokenBufferedWords(t *testing.T) {
	t.Run("a word for a sentence not yet recognized waits", func(t *testing.T) {
		seq := newSeq(t, true)
		stream(t, seq, "ctx1", "Hi", " there")
		if out := seq.ProcessWord("Hi", 10, "ctx1", false); len(out) != 0 {
			t.Fatalf("got %v, want nothing yet: the sentence it belongs to is not one yet",
				texts(out))
		}
	})

	t.Run("it is replayed once the sentence is recognized", func(t *testing.T) {
		seq := newSeq(t, true)
		stream(t, seq, "ctx1", "Hi", " there", "!")
		if out := seq.ProcessWord("Hi", 10, "ctx1", false); len(out) != 0 {
			t.Fatalf("got %v, want the word held", texts(out))
		}
		// The next sentence's first token confirms the boundary.
		out := stream(t, seq, "ctx1", " How")
		var said []string
		for _, w := range wordFrames(out) {
			said = append(said, w.Text)
		}
		if len(said) != 1 || said[0] != "Hi" {
			t.Fatalf("said %v, want the held word replayed", said)
		}
	})

	t.Run("a word for a sentence still to come waits again", func(t *testing.T) {
		seq := newSeq(t, true)
		stream(t, seq, "ctx1", "Hi", " there", "!", " How")
		// This belongs to the second sentence, which is not recognized yet.
		if out := seq.ProcessWord("How", 5, "ctx1", false); len(out) != 0 {
			t.Fatalf("got %v, want the word held", texts(out))
		}
		// The first sentence finishes on its own words.
		seq.ProcessWord("Hi", 10, "ctx1", false)
		seq.ProcessWord("there!", 20, "ctx1", false)
		// Finalizing promotes the second, and the held word matches it at last.
		var said []string
		for _, w := range wordFrames(seq.Finalize("ctx1")) {
			said = append(said, w.Text)
		}
		if !slices.Contains(said, "How") {
			t.Fatalf("said %v, want the held word among them", said)
		}
	})

	t.Run("without streaming, an unclaimed word still passes through", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello world"), "ctx1", "hello world", true, true, false)
		out := seq.ProcessWord("zzz", 5, "ctx1", false)
		if len(out) != 1 || wordFrames(out)[0].Text != "zzz" {
			t.Fatalf("got %v, want the word passed through", texts(out))
		}
	})
}

// A unit the synthesizer never speaks arriving mid-sentence forces the sentence
// so far into a slot, so the unit waits behind speech that really precedes it
// rather than jumping ahead of it.
func TestRegisterSkippedForcesFinalize(t *testing.T) {
	t.Run("the pending sentence is promoted ahead of it", func(t *testing.T) {
		seq := newSeq(t, true)
		stream(t, seq, "ctx1", "Hi", " there")
		skipped := skippedFrame("code")
		out := seq.RegisterSkipped(skipped, "ctx1", "")

		promoted := aggFrames(out)
		if len(promoted) == 0 || promoted[0].Text != "Hi there" {
			t.Fatalf("got %v, want the sentence so far promoted first", texts(out))
		}
		if contains(out, skipped) {
			t.Fatal("the held frame went out ahead of speech that precedes it")
		}
	})

	t.Run("with nothing pending it goes at once", func(t *testing.T) {
		seq := newSeq(t, true)
		frame := skippedFrame("code")
		out := seq.RegisterSkipped(frame, "ctx1", "")
		if len(out) != 1 || out[0] != frames.Frame(frame) {
			t.Fatalf("got %v, want the frame released at once", texts(out))
		}
	})
}

// An interruption clears everything. What matters afterwards is that the words
// still arriving for the turn that was cut off do not leak into the next one.
func TestClear(t *testing.T) {
	t.Run("what waited is dropped and the next unit is not held", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, true, false)
		seq.RegisterSkipped(skippedFrame("code"), "ctx2", "")
		seq.Clear()

		frame := skippedFrame("code")
		if out := seq.RegisterSkipped(frame, "ctx3", ""); len(out) != 1 {
			t.Fatalf("got %v, want the frame released: nothing precedes it any more", texts(out))
		}
	})

	t.Run("a word for the cleared turn is dropped", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("hello"), "ctx1", "hello", true, true, false)
		seq.Clear()
		if out := seq.ProcessWord("hello", 1, "ctx1", false); len(out) != 0 {
			t.Fatalf("got %v, want nothing: that turn is over", texts(out))
		}
	})

	t.Run("the words of an interrupted turn do not reach the next one", func(t *testing.T) {
		// A provider goes on delivering timings for a turn that was cut off. They
		// must not be threaded into the turn that replaced it.
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame("I just wanted to follow up"), "ctxA",
			"I just wanted to follow up", true, true, false)
		seq.Clear()
		seq.RegisterSpoken(spokenFrame("Hello"), "ctxB", "Hello", true, true, false)

		for _, stale := range []string{"I", "just", "wanted", "to", "follow", "up"} {
			if out := seq.ProcessWord(stale, 1, "ctxA", false); len(out) != 0 {
				t.Fatalf("the word %q from the interrupted turn reached the new one", stale)
			}
		}

		out := seq.ProcessWord("Hello", 2, "ctxB", false)
		if len(out) != 2 {
			t.Fatalf("got %v, want the new turn's word and its progress", texts(out))
		}
		word := wordFrames(out)[0]
		if word.Text != "Hello" || word.ContextID != "ctxB" || !word.AppendToContext {
			t.Fatalf("the new turn's word came out as %q on %q (append=%v)",
				word.Text, word.ContextID, word.AppendToContext)
		}
	})
}

// assembleContext joins the words the way the conversation does, so a test can
// assert what the turn reads as rather than inspecting each frame's flags.
func assembleContext(fs []frames.Frame) string {
	var parts []ttstext.Part
	for _, w := range wordFrames(fs) {
		parts = append(parts, ttstext.Part{
			Text:                    w.Text,
			IncludesInterPartSpaces: w.IncludesInterFrameSpaces,
		})
	}
	return ttstext.Concatenate(parts)
}

// The answer about spacing arrives with the words, not with the unit: the
// service registers the unit knowing nothing about the language, and the
// provider says per batch of timings whether its tokens carry their own
// spacing. It has to reach the frames either way.
func TestCJKSpacingReachesTheFrames(t *testing.T) {
	const sentence = "どんなことでも気軽に話しかけてくださいね。"

	t.Run("the answer given with a word reaches its frame", func(t *testing.T) {
		seq := newSeq(t, false)
		// Registered without it, the way the service does.
		seq.RegisterSpoken(spokenFrame(sentence), "ctx1", sentence, true, true, false)

		out := seq.ProcessWord("どんなことでも気", 100, "ctx1", true)
		got := wordFrames(out)
		if len(got) != 1 {
			t.Fatalf("got %v, want one word", texts(out))
		}
		if !got[0].IncludesInterFrameSpaces {
			t.Fatal("the word says it needs spacing supplied, though it was reported as " +
				"carrying its own")
		}
	})

	t.Run("two chunks assemble with nothing between them", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame(sentence), "ctx1", sentence, true, true, false)

		out := seq.ProcessWord("どんなことでも気", 100, "ctx1", true)
		out = append(out, seq.ProcessWord("軽に話しかけてくださいね。", 200, "ctx1", true)...)

		if got := assembleContext(out); got != sentence {
			t.Fatalf("the turn assembles to %q, want %q", got, sentence)
		}
	})

	t.Run("closing the unit out keeps the answer", func(t *testing.T) {
		seq := newSeq(t, false)
		seq.RegisterSpoken(spokenFrame(sentence), "ctx1", sentence, true, true, false)

		seq.ProcessWord("どんなことでも気", 100, "ctx1", true)
		// The rest is never reported, so the unit is closed out with what is left.
		got := wordFrames(seq.ForceComplete("ctx1", 200))
		if len(got) != 1 {
			t.Fatalf("got %d frames, want the unspoken remainder", len(got))
		}
		if !got[0].IncludesInterFrameSpaces {
			t.Fatal("the remainder says it needs spacing supplied, losing what the words " +
				"of this unit had already established")
		}
	})
}
