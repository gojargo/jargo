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

// Ported from upstream. A provider that reports punctuation with the following
// word (", I") rather than the preceding one ("Yeah,") presents it twice:
// punctuation trailing a word is swept into that word's span already. The
// duplicate is dropped from the frame word, and the attribution is kept, because
// discarding it would lose the written form of the word entirely.
func TestPunctuationLeadingTheNextToken(t *testing.T) {
	const sentence = "Yeah, I can do that. "

	t.Run("the duplicate is dropped from the frame word", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, sentence, sentence)

		tr.AddWord("Yeah")
		if raw, ok := tr.RawText(); !ok || raw != "Yeah," {
			t.Fatalf("first word attributed to %q (set=%v), want %q: the comma belongs to "+
				"the word it trails", raw, ok, "Yeah,")
		}

		tr.AddWord(", I")
		if word, _ := tr.FrameWord(); word != "I" {
			t.Fatalf("frame word = %q, want %q: the comma must not be emitted twice", word, "I")
		}
		if raw, ok := tr.RawText(); !ok || raw != "I" {
			t.Fatalf("attributed span = %q (set=%v), want %q", raw, ok, "I")
		}
	})

	t.Run("the words after it still track", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, sentence, sentence)
		for _, word := range []string{"Yeah", ", I", "can", "do"} {
			if !tr.WordBelongsHere(word) {
				t.Fatalf("%q was rejected as not belonging to this frame", word)
			}
			tr.AddWord(word)
		}
		if !tr.AddWord("that") {
			t.Fatal("the frame did not complete on its last word")
		}
	})
}

// Ported from upstream. A synthesizer sometimes reports one token spanning the
// boundary between two frames ("helloworld" where this frame ends at "hello").
// The part belonging here is emitted for this frame and the rest is handed to
// the next one, split by counting letters and digits rather than bytes.
func TestOverflowAcrossTheFrameBoundary(t *testing.T) {
	t.Run("a word straddling the boundary splits", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello", "", "")
		if !tr.AddWord("helloworld") {
			t.Fatal("the frame did not complete")
		}
		if word, _ := tr.FrameWord(); word != "hello" {
			t.Fatalf("frame word = %q, want %q", word, "hello")
		}
		if over, ok := tr.OverflowWord(); !ok || over != "world" {
			t.Fatalf("overflow = %q (set=%v), want %q", over, ok, "world")
		}
	})

	t.Run("a word that fits exactly overflows nothing", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello", "", "")
		if !tr.AddWord("hello") {
			t.Fatal("the frame did not complete")
		}
		if over, ok := tr.OverflowWord(); ok && over != "" {
			t.Fatalf("overflow = %q, want none", over)
		}
	})

	t.Run("the split counts letters and digits, not bytes", func(t *testing.T) {
		tr := NewWordCompletionTracker("4111", "", "")
		if !tr.AddWord("41111111") {
			t.Fatal("the frame did not complete")
		}
		if word, _ := tr.FrameWord(); word != "4111" {
			t.Fatalf("frame word = %q, want %q", word, "4111")
		}
		if over, _ := tr.OverflowWord(); over != "1111" {
			t.Fatalf("overflow = %q, want %q", over, "1111")
		}
	})

	t.Run("the overflow completes the next frame", func(t *testing.T) {
		first := NewWordCompletionTracker("hello", "", "")
		second := NewWordCompletionTracker("world", "", "")

		first.AddWord("helloworld")
		over, _ := first.OverflowWord()
		if !second.AddWord(over) {
			t.Fatal("the next frame did not complete on the overflow")
		}
		if word, _ := second.FrameWord(); word != "world" {
			t.Fatalf("next frame word = %q, want %q", word, "world")
		}
	})

	t.Run("a normal word leaves no overflow behind", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello world", "", "")
		tr.AddWord("hello")
		if over, ok := tr.OverflowWord(); ok && over != "" {
			t.Fatalf("overflow = %q after a word that fits, want none", over)
		}
	})
}

// Ported from upstream. A synthesizer that drops a word-timestamp event reports
// a word belonging to the next frame while this one is unfinished. The frame is
// closed out with whatever of it went unspoken, so the conversation still
// records it, and the word is handed on.
func TestAWordThatDoesNotBelongClosesTheFrameOut(t *testing.T) {
	t.Run("with nothing spoken yet, the whole frame is closed out", func(t *testing.T) {
		tr := NewWordCompletionTracker("number is", "", "")
		if !tr.AddWord("4111") {
			t.Fatal("the frame was not closed out")
		}
		if word, _ := tr.FrameWord(); word != "number is" {
			t.Fatalf("frame word = %q, want the whole unspoken text %q", word, "number is")
		}
		if over, _ := tr.OverflowWord(); over != "4111" {
			t.Fatalf("overflow = %q, want the word handed on %q", over, "4111")
		}
	})

	t.Run("partway through, only what is left is closed out", func(t *testing.T) {
		tr := NewWordCompletionTracker("number is", "", "")
		tr.AddWord("number")
		if !tr.AddWord("4111") {
			t.Fatal("the frame was not closed out")
		}
		if word, _ := tr.FrameWord(); word != "is" {
			t.Fatalf("frame word = %q, want the unspoken remainder %q", word, "is")
		}
	})

	t.Run("after several words, the remainder is closed out", func(t *testing.T) {
		tr := NewWordCompletionTracker("Your credit card number is", "", "")
		for _, w := range []string{"Your", "credit", "card"} {
			tr.AddWord(w)
		}
		if !tr.AddWord("4111") {
			t.Fatal("the frame was not closed out")
		}
		if word, _ := tr.FrameWord(); word != "number is" {
			t.Fatalf("frame word = %q, want %q", word, "number is")
		}
	})

	t.Run("the word handed on completes the next frame", func(t *testing.T) {
		first := NewWordCompletionTracker("number is", "", "")
		second := NewWordCompletionTracker("4111 1111", "", "")

		first.AddWord("4111")
		over, _ := first.OverflowWord()
		if second.AddWord(over) {
			t.Fatal("the next frame completed on its first word, which is only half of it")
		}
		if !second.AddWord("1111") {
			t.Fatal("the next frame did not complete")
		}
	})

	t.Run("closing out completes the frame however little was spoken", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello world", "", "")
		tr.AddWord("hello")
		if tr.IsComplete() {
			t.Fatal("the frame reported itself complete with a word still unspoken")
		}
		tr.AddWord("4111")
		if !tr.IsComplete() {
			t.Fatal("the frame did not complete after being closed out")
		}
	})

	t.Run("the next frame starts clean", func(t *testing.T) {
		first := NewWordCompletionTracker("ab", "", "")
		second := NewWordCompletionTracker("cd", "", "")

		first.AddWord("xyz")
		if over, ok := first.OverflowWord(); !ok || over == "" {
			t.Fatal("the word that did not belong was not handed on")
		}
		if !second.AddWord("cd") {
			t.Fatal("the next frame did not complete")
		}
		if over, ok := second.OverflowWord(); ok && over != "" {
			t.Fatalf("the next frame reported overflow %q from a word that fit it", over)
		}
	})
}

// Ported from upstream. A synthesizer may report a word with terminal
// punctuation the written text does not have: reading a list, it speaks each
// line as its own sentence and reports "account." where the text has "account"
// followed by a newline. A tracker that rejected such a word would desync on it
// and reject every word after it, losing the rest of the turn.
func TestPunctuationTheSynthesizerAddedIsTolerated(t *testing.T) {
	const sentence = "To better assist you, please choose one of the following options:\n\n" +
		"        A. I need help with my account\n" +
		"        B. I have a question about billing\n" +
		"        C. I need technical support\n\n" +
		"        Which option fits your needs best?"

	// The words as a provider reports them, a period appended to the last word
	// of each line.
	spokenWords := []string{
		"To", "better", "assist", "you,", "please", "choose", "one", "of", "the",
		"following", "options:",
		"A.", "I", "need", "help", "with", "my", "account.",
		"B.", "I", "have", "a", "question", "about", "billing.",
		"C.", "I", "need", "technical", "support.",
		"Which", "option", "fits", "your", "needs", "best?",
	}

	t.Run("every word is recognized", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, sentence, sentence)
		for _, w := range spokenWords {
			if !tr.WordBelongsHere(w) {
				t.Fatalf("%q was rejected over punctuation the synthesizer added", w)
			}
			tr.AddWord(w)
		}
	})

	t.Run("the frame completes on its last word", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, sentence, sentence)
		for _, w := range spokenWords[:len(spokenWords)-1] {
			if tr.AddWord(w) {
				t.Fatalf("the frame completed early, on %q", w)
			}
		}
		if !tr.AddWord(spokenWords[len(spokenWords)-1]) {
			t.Fatal("the frame did not complete on its last word")
		}
	})

	t.Run("one such word does not desync the rest", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, sentence, sentence)
		for _, w := range spokenWords[:17] { // up to "my", just before "account."
			tr.AddWord(w)
		}
		for _, w := range spokenWords[17:] { // "account." onward
			if !tr.WordBelongsHere(w) {
				t.Fatalf("%q was rejected after the word carrying the added period", w)
			}
			tr.AddWord(w)
		}
	})
}

// Ported from upstream. A synthesizer may report a word in a different case
// than the text has it, an acronym spoken back as "sql" for "SQL". The word is
// still the word.
func TestACaseDifferenceIsTolerated(t *testing.T) {
	const sentence = "Please open the SQL database now."
	spokenWords := []string{"Please", "open", "the", "sql", "database", "now."}

	tr := NewWordCompletionTracker(sentence, "", "")
	for _, w := range spokenWords[:len(spokenWords)-1] {
		if !tr.WordBelongsHere(w) {
			t.Fatalf("%q was rejected over a difference of case", w)
		}
		if tr.AddWord(w) {
			t.Fatalf("the frame completed early, on %q", w)
		}
	}
	last := spokenWords[len(spokenWords)-1]
	if !tr.WordBelongsHere(last) {
		t.Fatalf("%q was rejected over a difference of case", last)
	}
	if !tr.AddWord(last) {
		t.Fatal("the frame did not complete on its last word")
	}
}

// Ported from upstream. A synthesizer may strip the diacritics from a word,
// reporting "cafe" for "café".
func TestAnAccentDifferenceIsTolerated(t *testing.T) {
	const sentence = "Bienvenue au café parisien."
	spokenWords := []string{"Bienvenue", "au", "cafe", "parisien."}

	tr := NewWordCompletionTracker(sentence, "", "")
	for _, w := range spokenWords[:len(spokenWords)-1] {
		if !tr.WordBelongsHere(w) {
			t.Fatalf("%q was rejected over a missing accent", w)
		}
		if tr.AddWord(w) {
			t.Fatalf("the frame completed early, on %q", w)
		}
	}
	last := spokenWords[len(spokenWords)-1]
	if !tr.AddWord(last) {
		t.Fatal("the frame did not complete on its last word")
	}
}

// Ported from upstream. Scripts written without spaces between words are
// reported differently by different providers: Korean word by word, Japanese
// and Chinese as whole groups of characters merged into one token. Tracking has
// to hold for both, and the cursor has to land in the right place so what is
// left of a frame can still be emitted if the rest never arrives.
func TestCJKCompletion(t *testing.T) {
	t.Run("Korean, one word at a time", func(t *testing.T) {
		const sentence = "저는 여러분의 AI 어시스턴트입니다."
		words := []string{"저는", "여러분의", "AI", "어시스턴트입니다."}
		tr := NewWordCompletionTracker(sentence, "", "")
		for _, w := range words[:len(words)-1] {
			if tr.AddWord(w) {
				t.Fatalf("the frame completed early, on %q", w)
			}
		}
		if !tr.AddWord(words[len(words)-1]) {
			t.Fatal("the frame did not complete on its last word")
		}
	})

	t.Run("Korean, one syllable counts as one character", func(t *testing.T) {
		// Decomposing a Hangul syllable into its parts would make the
		// normalized text longer than the original and walk the cursor past the
		// end of every frame.
		for _, text := range []string{"저는여러분의", "안녕하세요", "어시스턴트"} {
			var raw int
			for _, r := range text {
				if isAlnum(r) {
					raw++
				}
			}
			if got := len([]rune(normalize(text))); got != raw {
				t.Fatalf("normalize(%q) is %d characters, want %d", text, got, raw)
			}
		}
	})

	t.Run("Korean, what is left of the frame reads verbatim", func(t *testing.T) {
		const sentence = "저는 여러분의 AI 어시스턴트입니다."
		tr := NewWordCompletionTracker(sentence, "", "")
		tr.AddWord("저는")
		if got := tr.RemainingTTSText(true); got != "여러분의 AI 어시스턴트입니다." {
			t.Fatalf("what is left = %q, want %q", got, "여러분의 AI 어시스턴트입니다.")
		}
	})

	t.Run("Korean mixed with Latin", func(t *testing.T) {
		tr := NewWordCompletionTracker("AI 어시스턴트입니다.", "", "")
		if tr.AddWord("AI") {
			t.Fatal("the frame completed on its first word")
		}
		if !tr.AddWord("어시스턴트입니다.") {
			t.Fatal("the frame did not complete")
		}
	})

	t.Run("Korean, a word belongs only once the one before it is spoken", func(t *testing.T) {
		tr := NewWordCompletionTracker("저는 여러분의", "", "")
		if !tr.WordBelongsHere("저는") {
			t.Fatal("the first word was rejected")
		}
		if tr.WordBelongsHere("여러분의") {
			t.Fatal("the second word was accepted before the first was spoken")
		}
		tr.AddWord("저는")
		if !tr.WordBelongsHere("여러분의") {
			t.Fatal("the second word was rejected once the first had been spoken")
		}
	})

	t.Run("Japanese, one merged group", func(t *testing.T) {
		tr := NewWordCompletionTracker("こんにちは。", "", "")
		if !tr.AddWord("こんにちは。") {
			t.Fatal("the frame did not complete on the group covering all of it")
		}
	})

	t.Run("Japanese, two groups for one frame", func(t *testing.T) {
		const sentence = "こんにちは、私はあなたの"
		tr := NewWordCompletionTracker(sentence, "", "")
		if tr.AddWord("こんにちは、私") {
			t.Fatal("the frame completed on the first of two groups")
		}
		if !tr.AddWord("はあなたの") {
			t.Fatal("the frame did not complete on the second group")
		}
	})

	t.Run("Japanese, the cursor lands between the groups", func(t *testing.T) {
		const sentence = "こんにちは、私はあなたの"
		tr := NewWordCompletionTracker(sentence, "", "")
		tr.AddWord("こんにちは、私")
		if got := tr.RemainingTTSText(true); got != "はあなたの" {
			t.Fatalf("what is left = %q, want %q", got, "はあなたの")
		}
	})
}
