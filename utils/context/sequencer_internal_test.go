package context

import (
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// What a context ending has to account for: a buffered word no slot ever
// matched, which is dropped where sentence mode would have dropped it on
// arrival. The buffer is the sequencer's own state, so these read it directly.

// streamedSeq builds a streaming sequencer and registers each token on it.
func streamedSeq(t *testing.T, tokens ...string) *AggregatedFrameSequencer {
	t.Helper()
	s := NewAggregatedFrameSequencer("Test", true)
	for _, token := range tokens {
		s.RegisterSpoken(
			frames.NewAggregatedTextFrame(token, frames.AggregationToken), "ctx1", token, true, true, false, "")
	}
	return s
}

// bufferedWords is what the buffer holds, for an assertion to read.
func bufferedWords(s *AggregatedFrameSequencer) []string {
	out := make([]string, 0, len(s.buffered))
	for _, w := range s.buffered {
		out = append(out, w.word)
	}
	return out
}

// "。" is taken by "好" as trailing punctuation, so the slot is already complete
// when the provider reports the mark on its own, and the mark is buffered. It
// will never match anything now, which only the end of the context can say.
func TestBufferedWordIsDroppedAtContextEnd(t *testing.T) {
	s := streamedSeq(t, "您好", "。", "谢谢")
	for i, ch := range []string{"您", "好", "。"} {
		s.ProcessWord(ch, int64(i+1)*10, "ctx1", false)
	}
	if got := bufferedWords(s); len(got) != 1 || got[0] != "。" {
		t.Fatalf("buffered = %v, want the mark held back", got)
	}

	for _, f := range s.ForceComplete("ctx1", 30) {
		if w, ok := f.(*frames.TTSTextFrame); ok {
			t.Errorf("force-complete emitted %q, want nothing: the slot was already spoken", w.Text)
		}
	}
	if got := bufferedWords(s); len(got) != 0 {
		t.Errorf("buffered = %v, want the word dropped with the context it belonged to", got)
	}
}

// Contexts can be in flight at once, so a word buffered under another one is
// still waiting for a sentence of its own.
func TestBufferedWordForAnotherContextIsLeftAlone(t *testing.T) {
	s := streamedSeq(t, "您好", "。", "谢谢")
	for i, ch := range []string{"您", "好", "。"} {
		s.ProcessWord(ch, int64(i+1)*10, "ctx1", false)
	}
	s.buffered[0].contextID = "ctx2"

	s.ForceComplete("ctx1", 30)
	if got := bufferedWords(s); len(got) != 1 || got[0] != "。" {
		t.Errorf("buffered = %v, want the other context's word left for its own turn", got)
	}
}

// slotTexts is the text of every slot, trimmed.
func slotTexts(s *AggregatedFrameSequencer) []string {
	out := make([]string, 0, len(s.slots))
	for _, slot := range s.slots {
		out = append(out, strings.TrimSpace(slot.frame.Text))
	}
	return out
}

// The three channels are sliced at the boundaries of one language.
func TestParallelChannelsUseTheSameLanguageForSlicing(t *testing.T) {
	p := newParallelSentenceAggregator("de")
	text := "Das ist bzw. wichtig. Weiter geht es. N"
	result := p.aggregate(text, text, text)
	got := make([]string, 0, len(result))
	for _, a := range result {
		got = append(got, strings.TrimSpace(a.tts))
		if a.tts != a.llm || a.llm != a.userFacing {
			t.Errorf("channels differ: %+v", a)
		}
	}
	if want := []string{"Das ist bzw. wichtig.", "Weiter geht es."}; !slices.Equal(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func feedToken(s *AggregatedFrameSequencer, contextID, text, language string) {
	s.RegisterSpoken(frames.NewAggregatedTextFrame(text, frames.AggregationToken), contextID, text,
		true, true, false, language)
}

// A context keeps its own language after another context starts.
func TestStreamingContextRetainsLanguageAfterAnotherContextStarts(t *testing.T) {
	s := NewAggregatedFrameSequencer("Test", true)
	feedToken(s, "de", "Das ist bzw.", "de")
	feedToken(s, "en", "Hello. Next", "en")
	feedToken(s, "de", " wichtig. Weiter", "de")
	if got, want := slotTexts(s), []string{"Hello.", "Das ist bzw. wichtig."}; !slices.Equal(got, want) {
		t.Fatalf("slots = %q, want %q", got, want)
	}
}

// A context's language follows the text registered on it.
func TestStreamingContextLanguageUpdatesWithIncomingText(t *testing.T) {
	s := NewAggregatedFrameSequencer("Test", true)
	feedToken(s, "context", "Das ist bzw.", "de")
	feedToken(s, "context", " wichtig. Weiter", "en")
	if got, want := slotTexts(s), []string{"Das ist bzw.", "wichtig."}; !slices.Equal(got, want) {
		t.Fatalf("slots = %q, want %q", got, want)
	}
}
