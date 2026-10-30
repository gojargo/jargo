package context

import (
	"reflect"
	"regexp"
	"strings"
	"testing"
)

// Tags such as an expressive conversation's are in the LLM's text alone: the
// cursor into it passes over them, so a word is never attributed half a tag.
const (
	markupLLMText    = "<<calm>> It's [[laugh]] fine now."
	markupTTSText    = "It's [laughter] fine now."
	markupUserFacing = "It's fine now."
)

func expressionTags() *regexp.Regexp { return regexp.MustCompile(`<<[^<>]*>>|\[\[[^\[\]]*\]\]`) }

func TestTagsOnlyTheLLMTextCarriesAreAttributedWhole(t *testing.T) {
	tracker := NewWordCompletionTrackerWithMarkup(markupTTSText, markupUserFacing, markupLLMText, expressionTags())
	var consumed []string
	for _, word := range []string{"It's", "[laughter]", "fine", "now."} {
		tracker.AddWord(word)
		raw, ok := tracker.RawText()
		if !ok {
			raw = "<none>"
		}
		consumed = append(consumed, raw)
	}

	if want := []string{"<<calm>> It's", "<none>", "[[laugh]] fine", "now."}; !reflect.DeepEqual(consumed, want) {
		t.Fatalf("consumed = %q, want %q", consumed, want)
	}
}

func TestWhatIsLeftStartsAfterTheWordsSpoken(t *testing.T) {
	// The service's markup takes the space after it, as the tag does.
	tracker := NewWordCompletionTrackerWithMarkup(
		`<emotion value="calm"/> I'm so sorry.`, "I'm so sorry.", "<<calm>> I'm so sorry.", expressionTags())
	tracker.AddWord("I'm")

	if got := tracker.RemainingRawText(); got != "so sorry." {
		t.Fatalf("remaining = %q", got)
	}
}

func TestAFrameThatIsOnlyMarkupTakesNoWord(t *testing.T) {
	tracker := NewWordCompletionTrackerWithMarkup(`<break time="400ms"/>`, "", "[[pause 400]]", expressionTags())

	if tracker.WordBelongsHere("Bye") {
		t.Fatal("a word belongs to a frame that is only markup")
	}
}

func TestCaseOnlyReplacementLinesUpWithTheOriginal(t *testing.T) {
	// "SQL" -> "sql": only case differs, so the word lines up with the original
	// and stays in an equal segment with the words around it.
	m := NewTextSegmentMap("Contact sql support today.", "Contact SQL support today.", "")
	var seg *textSegment
	for i := range m.segments {
		if strings.Contains(m.segments[i].original(), "SQL") {
			seg = &m.segments[i]
		}
	}
	if seg == nil {
		t.Fatal("no segment holds SQL")
	}
	if seg.tts() != "Contact sql support today." {
		t.Fatalf("tts side = %q", seg.tts())
	}
	if seg.isTransformed() {
		t.Fatal("a case-only replacement is flagged transformed")
	}
}

func TestACaseOnlyWordAfterATagLinesUpWithTheOriginal(t *testing.T) {
	// A stressed word in capitals right after a tag is matched on its own, not
	// folded into one all-or-nothing segment with the tag.
	tracker := NewWordCompletionTracker(`<break time="600ms"/> REALLY good.`, "Really good.",
		"[[pause 600]] ^^Really^^ good.")

	if !tracker.WordBelongsHere("REALLY") {
		t.Fatal("REALLY does not belong")
	}
	if tracker.AddWord("REALLY") {
		t.Fatal("complete after REALLY")
	}
	if !tracker.AddWord("good.") {
		t.Fatal("not complete after good.")
	}
}
