package context

import (
	"strings"
	"testing"
)

// spoken drives a tracker (with the original text used for both the user-facing
// and LLM sides, as the TTS base wires it) through a sequence of spoken words,
// returning the original-form text recorded for the context: each accepted word
// contributes its mapped original span, and intermediate words of a transformed
// span are suppressed.
func spoken(ttsText, originalText string, words []string) (context string, completedAt int) {
	t := NewWordCompletionTracker(ttsText, originalText, originalText)
	var parts []string
	completedAt = -1
	for i, w := range words {
		done := t.AddWord(w)
		if !t.Suppress() {
			if raw, ok := t.RawText(); ok && raw != "" {
				parts = append(parts, raw)
			}
		}
		if done && completedAt < 0 {
			completedAt = i
		}
	}
	return strings.Join(parts, " "), completedAt
}

func TestTrackerAtomicCurrencySpan(t *testing.T) {
	// "$42.50" is spoken as several words but must map back to the single
	// original token, held atomic until the span completes.
	tts := "Your balance is forty two dollars and fifty cents"
	orig := "Your balance is $42.50"
	words := []string{"Your", "balance", "is", "forty", "two", "dollars", "and", "fifty", "cents"}

	got, completedAt := spoken(tts, orig, words)
	if got != "Your balance is $42.50" {
		t.Fatalf("context = %q, want %q", got, "Your balance is $42.50")
	}
	if completedAt != len(words)-1 {
		t.Fatalf("completed at word %d, want %d", completedAt, len(words)-1)
	}

	// Intermediate words of the transformed span must be suppressed and carry no
	// original text; only the completing word does.
	tr := NewWordCompletionTracker(tts, orig, orig)
	for _, w := range []string{"Your", "balance", "is", "forty", "two"} {
		tr.AddWord(w)
	}
	if !tr.Suppress() {
		t.Fatal("expected suppression mid transformed span (after 'two')")
	}
	if raw, ok := tr.RawText(); ok && raw != "" {
		t.Fatalf("mid-span raw text = %q, want none", raw)
	}
	tr.AddWord("dollars")
	tr.AddWord("and")
	tr.AddWord("fifty")
	tr.AddWord("cents")
	if raw, _ := tr.RawText(); raw != "$42.50" {
		t.Fatalf("completing-word raw text = %q, want %q", raw, "$42.50")
	}
	if tr.Suppress() {
		t.Fatal("completing word must not be suppressed")
	}
}

func TestTrackerAcronymLetterSpacing(t *testing.T) {
	// "API" is letter-spaced to "A P I": same alphanumeric content but a
	// different word count, so it is still atomic.
	got, _ := spoken("Call the A P I now", "Call the API now",
		[]string{"Call", "the", "A", "P", "I", "now"})
	if got != "Call the API now" {
		t.Fatalf("context = %q, want %q", got, "Call the API now")
	}
}

func TestTrackerSSMLMarkupInOutput(t *testing.T) {
	// The synthesizer wraps a word in SSML; the spoken word maps back to the
	// untagged original.
	got, _ := spoken("Say <spell>SQL</spell> please", "Say SQL please",
		[]string{"Say", "SQL", "please"})
	if got != "Say SQL please" {
		t.Fatalf("context = %q, want %q", got, "Say SQL please")
	}
}

func TestTrackerTerminalPunctuationOwnToken(t *testing.T) {
	// A terminal '?' set off by a space arrives as its own word-timestamp token,
	// so the frame must stay open until it lands.
	tr := NewWordCompletionTracker("Comment ça va ?", "Comment ça va ?", "Comment ça va ?")
	for _, w := range []string{"Comment", "ça", "va"} {
		tr.AddWord(w)
	}
	if tr.IsComplete() {
		t.Fatal("frame complete before the separated '?' token arrived")
	}
	if done := tr.AddWord("?"); !done {
		t.Fatal("frame not complete after the '?' token")
	}
}

func TestTrackerPlainSentenceReconstructs(t *testing.T) {
	// With no transforms, the recorded context is the original text verbatim.
	got, at := spoken("Hello there, friend.", "Hello there, friend.",
		[]string{"Hello", "there,", "friend."})
	if got != "Hello there, friend." {
		t.Fatalf("context = %q, want verbatim original", got)
	}
	if at != 2 {
		t.Fatalf("completed at %d, want 2", at)
	}
}

func TestTrackerRemainingRawTextClosesFrame(t *testing.T) {
	// When a synthesizer under-reports words, the remaining original text closes
	// out the frame so the full sentence still reaches the context.
	tr := NewWordCompletionTracker("Hello brave new world", "Hello brave new world", "Hello brave new world")
	tr.AddWord("Hello")
	tr.AddWord("brave")
	if tr.IsComplete() {
		t.Fatal("unexpectedly complete after two of four words")
	}
	if rem := tr.RemainingRawText(); rem != "new world" {
		t.Fatalf("remaining raw text = %q, want %q", rem, "new world")
	}
}
