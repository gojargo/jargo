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

// Ported from upstream. A transformed span whose written form ends in
// punctuation the synthesizer never speaks ("$42.50," read as "forty two
// dollars and fifty cents"): the word after it has to be placed past
// punctuation the cursor reached by the span's jump rather than by speaking it.
func TestATransformFollowedByPunctuationNobodySpeaks(t *testing.T) {
	const spoken = "Your balance is forty two dollars and fifty cents, and it is ready"
	const written = "Your balance is $42.50, and it is ready"

	t.Run("the words after it are still attributed", func(t *testing.T) {
		tr := NewWordCompletionTracker(spoken, written, written)
		words := []string{
			"Your", "balance", "is", "forty", "two", "dollars", "and", "fifty", "cents",
			"and", "it", "is", "ready",
		}
		for _, w := range words {
			if !tr.WordBelongsHere(w) {
				t.Fatalf("%q was rejected", w)
			}
			tr.AddWord(w)
		}
		if !tr.IsComplete() {
			t.Fatal("the frame did not complete")
		}
	})

	t.Run("the span is attributed once, on the word completing it", func(t *testing.T) {
		tr := NewWordCompletionTracker(spoken, written, written)
		for _, w := range []string{"Your", "balance", "is", "forty", "two", "dollars", "and", "fifty"} {
			tr.AddWord(w)
		}
		// The words inside the expansion carry no written form of their own.
		if raw, ok := tr.RawText(); ok && raw != "" {
			t.Fatalf("a word inside the expansion was attributed %q, want nothing", raw)
		}

		tr.AddWord("cents")
		if raw, _ := tr.RawText(); raw != "$42.50," {
			t.Fatalf("the completing word was attributed %q, want the whole written span %q",
				raw, "$42.50,")
		}

		tr.AddWord("and")
		if raw, _ := tr.RawText(); raw != "and" {
			t.Fatalf("the word after the span was attributed %q, want %q", raw, "and")
		}
	})
}

// Ported from upstream. French and other typographies put a space before ?, !,
// : and ;. The cursor stops at that space, so once the last word is spoken it
// still has to reach the end of the text. Otherwise what is left never empties,
// the segment stays open, and a client that commits its caption when the
// segment completes drops the sentence as the next one begins.
func TestPunctuationSeparatedFromItsWordByASpace(t *testing.T) {
	t.Run("a question mark completes the frame", func(t *testing.T) {
		tr := NewWordCompletionTracker("Comment ça va ?", "", "")
		tr.AddWord("Comment")
		tr.AddWord("ça")
		if tr.AddWord("va") {
			t.Fatal("the frame completed before its punctuation was spoken")
		}
		if got := tr.RemainingUserFacingText(false); got != " ?" {
			t.Fatalf("what is left = %q, want %q", got, " ?")
		}
		if got := tr.AccumulatedUserFacingText(); got != "Comment ça va" {
			t.Fatalf("what has been said = %q, want %q", got, "Comment ça va")
		}
		if !tr.AddWord("?") {
			t.Fatal("the frame did not complete on the punctuation")
		}
	})

	t.Run("an exclamation mark completes the frame", func(t *testing.T) {
		tr := NewWordCompletionTracker("Bonjour !", "", "")
		if tr.AddWord("Bonjour") {
			t.Fatal("the frame completed before its punctuation was spoken")
		}
		if got := tr.RemainingUserFacingText(false); got != " !" {
			t.Fatalf("what is left = %q, want %q", got, " !")
		}
		if !tr.AddWord("!") {
			t.Fatal("the frame did not complete on the punctuation")
		}
	})

	t.Run("the space before it may be a non-breaking one", func(t *testing.T) {
		tr := NewWordCompletionTracker("Ça va ?", "", "")
		tr.AddWord("Ça")
		if tr.AddWord("va") {
			t.Fatal("the frame completed before its punctuation was spoken")
		}
		if got := tr.AccumulatedUserFacingText(); got != "Ça va" {
			t.Fatalf("what has been said = %q, want %q", got, "Ça va")
		}
		if !tr.AddWord(" ?") {
			t.Fatal("the frame did not complete on the punctuation")
		}
	})

	t.Run("English, with no space, is unchanged", func(t *testing.T) {
		tr := NewWordCompletionTracker("How are you?", "", "")
		tr.AddWord("How")
		tr.AddWord("are")
		if !tr.AddWord("you") {
			t.Fatal("the frame did not complete on its last word")
		}
		if got := tr.RemainingUserFacingText(false); got != "" {
			t.Fatalf("what is left = %q, want nothing", got)
		}
		if got := tr.AccumulatedUserFacingText(); got != "How are you?" {
			t.Fatalf("what has been said = %q, want the whole sentence", got)
		}
	})

	t.Run("a colon mid-sentence appears as soon as it is spoken", func(t *testing.T) {
		tr := NewWordCompletionTracker("Attention : ceci est un test", "", "")
		tr.AddWord("Attention")
		if got := tr.AccumulatedUserFacingText(); got != "Attention" {
			t.Fatalf("what has been said = %q, want %q", got, "Attention")
		}
		tr.AddWord(":")
		if got := tr.AccumulatedUserFacingText(); got != "Attention :" {
			t.Fatalf("what has been said = %q, want %q: a colon mid-sentence is followed by "+
				"more of the same segment, so it cannot lag behind", got, "Attention :")
		}
		tr.AddWord("ceci")
		if got := tr.AccumulatedUserFacingText(); got != "Attention : ceci" {
			t.Fatalf("what has been said = %q, want %q", got, "Attention : ceci")
		}
	})

	t.Run("a question mark between two sentences drains on its own word", func(t *testing.T) {
		tr := NewWordCompletionTracker("Comment ça va ? Bien merci.", "", "")
		for _, w := range []string{"Comment", "ça", "va"} {
			tr.AddWord(w)
		}
		if got := tr.AccumulatedUserFacingText(); got != "Comment ça va" {
			t.Fatalf("what has been said = %q, want %q", got, "Comment ça va")
		}
		if tr.AddWord("?") {
			t.Fatal("the frame completed with a sentence still to come")
		}
		if got := tr.AccumulatedUserFacingText(); got != "Comment ça va ?" {
			t.Fatalf("what has been said = %q, want %q", got, "Comment ça va ?")
		}
		tr.AddWord("Bien")
		if got := tr.AccumulatedUserFacingText(); got != "Comment ça va ? Bien" {
			t.Fatalf("what has been said = %q, want %q", got, "Comment ça va ? Bien")
		}
	})
}

// Ported from upstream. A transformed span that ends the utterance completes on
// its last spoken word, and that word will not appear in the written text: the
// synthesizer says "dollars" for "$5". The written form has to survive anyway,
// or the conversation records the expansion rather than what the model wrote.
func TestATransformThatEndsTheUtterance(t *testing.T) {
	t.Run("the written form survives the last word", func(t *testing.T) {
		tr := NewWordCompletionTracker("Your total is five dollars",
			"Your total is $5", "Your total is $5")
		for _, w := range []string{"Your", "total", "is", "five"} {
			tr.AddWord(w)
		}
		if !tr.AddWord("dollars") {
			t.Fatal("the frame did not complete on its last word")
		}
		raw, ok := tr.RawText()
		if !ok || raw != "$5" {
			t.Fatalf("attributed %q (set=%v), want the written form %q", raw, ok, "$5")
		}
	})

	t.Run("a closing tag around it is swept in", func(t *testing.T) {
		tr := NewWordCompletionTracker("Total is fifty percent",
			"Total is 50%", "<price>Total is 50%</price>")
		for _, w := range []string{"Total", "is", "fifty"} {
			tr.AddWord(w)
		}
		if !tr.AddWord("percent") {
			t.Fatal("the frame did not complete on its last word")
		}
		raw, ok := tr.RawText()
		if !ok || !strings.Contains(raw, "50%") {
			t.Fatalf("attributed %q (set=%v), want it to carry %q", raw, ok, "50%")
		}
	})

	t.Run("a transform mid-sentence is unaffected", func(t *testing.T) {
		tr := NewWordCompletionTracker("Your balance is five dollars due now",
			"Your balance is $5 due now", "Your balance is $5 due now")
		for _, w := range []string{"Your", "balance", "is", "five"} {
			tr.AddWord(w)
		}
		if raw, ok := tr.RawText(); ok && raw != "" {
			t.Fatalf("a word inside the expansion was attributed %q, want nothing", raw)
		}
		tr.AddWord("dollars")
		if raw, _ := tr.RawText(); raw != "$5" {
			t.Fatalf("the completing word was attributed %q, want %q", raw, "$5")
		}
		tr.AddWord("due")
		if raw, ok := tr.RawText(); !ok || raw == "" {
			t.Fatal("the word after the expansion was attributed nothing")
		}
		tr.AddWord("now")
		if !tr.IsComplete() {
			t.Fatal("the frame did not complete")
		}
	})
}

// Ported from upstream. A pronunciation replacement can change how the text is
// broken into words, or only its case or its connector. Whichever it does, the
// conversation has to record what the model wrote rather than the respelling
// the synthesizer was handed.
func TestReplacementsThatChangeTheShapeOfAWord(t *testing.T) {
	t.Run("one word split into two commits the original on the second", func(t *testing.T) {
		const written = "Try BODYPUMP on Monday morning."
		const spoken = "Try body pump on Monday morning."
		tr := NewWordCompletionTracker(spoken, written, written)

		tr.AddWord("Try")
		tr.AddWord("body")
		if raw, ok := tr.RawText(); ok && raw != "" {
			t.Fatalf("the first half was attributed %q, want nothing", raw)
		}
		if !tr.Suppress() {
			t.Fatal("the first half was not held back from the conversation")
		}

		tr.AddWord("pump")
		if raw, _ := tr.RawText(); raw != "BODYPUMP" {
			t.Fatalf("the second half was attributed %q, want the written word %q", raw, "BODYPUMP")
		}
		if tr.Suppress() {
			t.Fatal("the word completing the replacement was held back")
		}

		for _, w := range []string{"on", "Monday", "morning."} {
			tr.AddWord(w)
		}
		if !tr.IsComplete() {
			t.Fatal("the frame did not complete")
		}
	})

	t.Run("a difference of case alone keeps the written form", func(t *testing.T) {
		const written = "Contact SQL support today."
		const spoken = "Contact sql support today."
		tr := NewWordCompletionTracker(spoken, written, written)

		tr.AddWord("Contact")
		tr.AddWord("sql")
		if raw, _ := tr.RawText(); raw != "SQL" {
			t.Fatalf("attributed %q, want the written form %q", raw, "SQL")
		}
		for _, w := range []string{"support", "today."} {
			tr.AddWord(w)
		}
		if !tr.IsComplete() {
			t.Fatal("the frame did not complete")
		}
	})

	t.Run("a hyphen standing in for nothing keeps the written form", func(t *testing.T) {
		const written = "Try BODYPUMP on Monday morning."
		const spoken = "Try body-pump on Monday morning."
		tr := NewWordCompletionTracker(spoken, written, written)

		tr.AddWord("Try")
		tr.AddWord("body-pump")
		if raw, _ := tr.RawText(); raw != "BODYPUMP" {
			t.Fatalf("attributed %q, want the written form %q", raw, "BODYPUMP")
		}
	})

	t.Run("the cursor ends past the whole written sentence", func(t *testing.T) {
		const written = "Try BODYPUMP on Monday morning."
		const spoken = "Try body pump on Monday morning."
		tr := NewWordCompletionTracker(spoken, written, written)
		for w := range strings.FieldsSeq(spoken) {
			tr.AddWord(w)
		}
		got, ok := tr.AccumulatedRawText()
		if !ok || got != written {
			t.Fatalf("consumed %q (set=%v), want the whole written sentence %q", got, ok, written)
		}
	})

	t.Run("a respelling of a different length still maps back", func(t *testing.T) {
		const written = "The leisure center opens at six."
		const spoken = "The lezher center opens at six."
		tr := NewWordCompletionTracker(spoken, written, written)
		tr.AddWord("The")
		tr.AddWord("lezher")
		if raw, _ := tr.RawText(); !strings.Contains(raw, "leisure") {
			t.Fatalf("attributed %q, want it to carry the written word %q", raw, "leisure")
		}
	})

	t.Run("an inline pronunciation tag does not steal the next word's letters", func(t *testing.T) {
		const written = "The leisure center opens at six."
		const spoken = "The <<l|ɛ|ʒ|ə|r>> center opens at six."
		tr := NewWordCompletionTracker(spoken, written, written)

		tr.AddWord("The")
		tr.AddWord("<<l|ɛ|ʒ|ə|r>>")
		tr.AddWord("center")
		if raw, _ := tr.RawText(); raw != "center" {
			t.Fatalf("the word after the tag was attributed %q, want its own span %q", raw, "center")
		}
	})
}

// Ported from upstream. A sentence carrying emoji, alongside tags and currency
// in the written text. An emoji contributes no letters or digits, so it moves
// no cursor of its own and has to be found where it sits; the words after it
// must not lose their own span to it.
func TestEmojiInASentence(t *testing.T) {
	// step is one spoken word and what it should produce.
	type step struct {
		word      string
		frameWord string
		raw       string
		rawSet    bool
	}
	run := func(t *testing.T, spoken, written string, steps []step) {
		t.Helper()
		tr := NewWordCompletionTracker(spoken, "", written)
		for i, s := range steps {
			complete := tr.AddWord(s.word)
			if word, _ := tr.FrameWord(); word != s.frameWord {
				t.Fatalf("word %d (%q): frame word = %q, want %q", i+1, s.word, word, s.frameWord)
			}
			raw, ok := tr.RawText()
			if ok != s.rawSet || (s.rawSet && raw != s.raw) {
				t.Fatalf("word %d (%q): attributed %q (set=%v), want %q (set=%v)",
					i+1, s.word, raw, ok, s.raw, s.rawSet)
			}
			if last := i == len(steps)-1; complete != last {
				t.Fatalf("word %d (%q): complete = %v, want %v", i+1, s.word, complete, last)
			}
		}
	}

	t.Run("an emoji mid-sentence, inside tags", func(t *testing.T) {
		const sentence = "Great job! 🎉 Well done."
		run(t, sentence, "<praise>"+sentence+"</praise>", []step{
			{"Great", "Great", "<praise>Great", true},
			{"job!", "job!", "job!", true},
			{"🎉", "🎉", "🎉", true},
			{"Well", "Well", "Well", true},
			{"done.", "done.", "done.</praise>", true},
		})
	})

	t.Run("an emoji beside currency", func(t *testing.T) {
		const sentence = "Pay $50 😊 today!"
		run(t, sentence, "<promo>"+sentence+"</promo>", []step{
			{"Pay", "Pay", "<promo>Pay", true},
			{"$50", "$50", "$50", true},
			{"😊", "😊", "😊", true},
			{"today!", "today!", "today!</promo>", true},
		})
	})

	t.Run("several emoji, no tags", func(t *testing.T) {
		const sentence = "Hello 😊 world 🎉 there!"
		run(t, sentence, sentence, []step{
			{"Hello", "Hello", "Hello", true},
			{"😊", "😊", "😊", true},
			{"world", "world", "world", true},
			{"🎉", "🎉", "🎉", true},
			{"there!", "there!", "there!", true},
		})
	})

	t.Run("an emoji the written text does not have is attributed nothing", func(t *testing.T) {
		tr := NewWordCompletionTracker("See you soon 😊", "", "<note>See you soon</note>")
		for _, w := range []string{"See", "you", "soon"} {
			tr.AddWord(w)
		}
		tr.AddWord("😊")
		if word, _ := tr.FrameWord(); word != "😊" {
			t.Fatalf("frame word = %q, want the emoji: it was spoken, so it is part of the frame", word)
		}
		if raw, ok := tr.RawText(); ok && raw != "" {
			t.Fatalf("attributed %q, want nothing: the written text does not carry it", raw)
		}
		if !tr.IsComplete() {
			t.Fatal("the frame did not complete")
		}
	})
}

// Ported from upstream. The tracker as the sequencer drives it: several frames
// of one turn, with words arriving in order, dropped, or straddling a boundary,
// and the written text carrying tags around a span the model produced.
func TestSeveralFramesOfOneTurn(t *testing.T) {
	t.Run("two plain frames, words in order", func(t *testing.T) {
		first := NewWordCompletionTracker("Your credit card", "", "")
		second := NewWordCompletionTracker("number is 42", "", "")

		for _, w := range []string{"Your", "credit", "card"} {
			first.AddWord(w)
		}
		if !first.IsComplete() {
			t.Fatal("the first frame did not complete")
		}
		for _, w := range []string{"number", "is"} {
			if second.AddWord(w) {
				t.Fatalf("the second frame completed early, on %q", w)
			}
		}
		if !second.AddWord("42") {
			t.Fatal("the second frame did not complete")
		}
	})

	t.Run("tags around the written span are kept with its ends", func(t *testing.T) {
		first := NewWordCompletionTracker("Your credit card number is", "", "")
		second := NewWordCompletionTracker("4111 1111 1111 1111", "",
			"<card>4111 1111 1111 1111</card>")

		for _, w := range []string{"Your", "credit", "card", "number", "is"} {
			first.AddWord(w)
		}

		second.AddWord("4111")
		if raw, _ := second.RawText(); raw != "<card>4111" {
			t.Fatalf("the first word was attributed %q, want the opening tag with it", raw)
		}
		second.AddWord("1111")
		second.AddWord("1111")
		if !second.AddWord("1111") {
			t.Fatal("the frame did not complete on its last word")
		}
		if raw, _ := second.RawText(); raw != "1111</card>" {
			t.Fatalf("the last word was attributed %q, want the closing tag with it", raw)
		}
	})

	t.Run("dropped words close the frame out and the next one carries on", func(t *testing.T) {
		first := NewWordCompletionTracker("Your credit card number is", "", "")
		second := NewWordCompletionTracker("4111 1111 1111 1111", "",
			"<card>4111 1111 1111 1111</card>")

		for _, w := range []string{"Your", "credit", "card"} {
			first.AddWord(w)
		}
		// "number" and "is" never arrive; the next frame's first word does.
		if !first.AddWord("4111") {
			t.Fatal("the first frame was not closed out")
		}
		if word, _ := first.FrameWord(); word != "number is" {
			t.Fatalf("frame word = %q, want the unspoken remainder %q", word, "number is")
		}
		over, _ := first.OverflowWord()

		second.AddWord(over)
		if raw, _ := second.RawText(); raw != "<card>4111" {
			t.Fatalf("the handed-on word was attributed %q in the next frame", raw)
		}
		second.AddWord("1111")
		second.AddWord("1111")
		if !second.AddWord("1111") {
			t.Fatal("the second frame did not complete")
		}
	})

	t.Run("a straddling word carries its written span with it", func(t *testing.T) {
		first := NewWordCompletionTracker("4111 1111", "", "<card>4111 1111</card>")
		second := NewWordCompletionTracker("And", "", "")

		first.AddWord("4111")
		if raw, _ := first.RawText(); raw != "<card>4111" {
			t.Fatalf("attributed %q, want the opening tag with the first word", raw)
		}
		if !first.AddWord("1111And") {
			t.Fatal("the frame did not complete on the straddling word")
		}
		if raw, _ := first.RawText(); raw != "1111</card>" {
			t.Fatalf("attributed %q, want the closing tag with the last word", raw)
		}
		over, _ := first.OverflowWord()
		if over != "And" {
			t.Fatalf("overflow = %q, want %q", over, "And")
		}
		if !second.AddWord(over) {
			t.Fatal("the next frame did not complete on the overflow")
		}
	})

	t.Run("a whole frame of dropped words is closed out at once", func(t *testing.T) {
		first := NewWordCompletionTracker("one two three", "", "")
		second := NewWordCompletionTracker("four five", "", "")

		// Nothing of the first frame is ever reported.
		if !first.AddWord("four") {
			t.Fatal("the frame was not closed out")
		}
		if word, _ := first.FrameWord(); word != "one two three" {
			t.Fatalf("frame word = %q, want the whole frame %q", word, "one two three")
		}
		if second.AddWord("four") {
			t.Fatal("the next frame completed on its first of two words")
		}
		if !second.AddWord("five") {
			t.Fatal("the next frame did not complete")
		}
	})
}

// The shape of the contract: a frame is complete once its text has been
// accounted for, and the call that completes it says so.
func TestTrackerBasics(t *testing.T) {
	t.Run("a frame with words still to come is not complete", func(t *testing.T) {
		if NewWordCompletionTracker("Hello world", "", "").IsComplete() {
			t.Fatal("the frame reported itself complete before a word was spoken")
		}
	})

	t.Run("it completes when its last word is spoken", func(t *testing.T) {
		tr := NewWordCompletionTracker("Hello world", "", "")
		if tr.AddWord("Hello") || tr.IsComplete() {
			t.Fatal("the frame completed on its first of two words")
		}
		if !tr.AddWord("world") || !tr.IsComplete() {
			t.Fatal("the frame did not complete on its last word")
		}
	})

	t.Run("one word can be a whole frame", func(t *testing.T) {
		if !NewWordCompletionTracker("Hello", "", "").AddWord("Hello") {
			t.Fatal("the frame did not complete")
		}
	})

	t.Run("it stays complete once it is", func(t *testing.T) {
		tr := NewWordCompletionTracker("Hi", "", "")
		tr.AddWord("Hi")
		if !tr.AddWord("extra") || !tr.IsComplete() {
			t.Fatal("a word arriving after the frame was done unsettled it")
		}
	})

	t.Run("a word arriving late is dropped, not handed on", func(t *testing.T) {
		// Handing it on would put it into the next sentence, which never
		// contained it: the same word would be recorded twice.
		tr := NewWordCompletionTracker("Hello world", "", "")
		tr.AddWord("Hello")
		tr.AddWord("world")

		if !tr.AddWord("extra") {
			t.Fatal("the frame stopped reporting itself complete")
		}
		if over, ok := tr.OverflowWord(); ok && over != "" {
			t.Fatalf("the late word was handed on as %q, which would put it in the "+
				"next sentence", over)
		}
		if word, ok := tr.FrameWord(); ok && word != "" {
			t.Fatalf("the late word was emitted as %q", word)
		}
	})
}

// Resetting returns a tracker to its beginning without changing what it expects,
// so the same frame can be spoken again.
func TestTrackerReset(t *testing.T) {
	t.Run("it clears what was spoken", func(t *testing.T) {
		tr := NewWordCompletionTracker("Hello world", "", "")
		tr.AddWord("Hello")
		tr.AddWord("world")
		tr.Reset()
		if tr.IsComplete() {
			t.Fatal("the frame is still complete after being reset")
		}
	})

	t.Run("the frame can be spoken again", func(t *testing.T) {
		tr := NewWordCompletionTracker("Hello world", "", "")
		tr.AddWord("Hello")
		tr.AddWord("world")
		tr.Reset()
		if tr.AddWord("Hello") {
			t.Fatal("the frame completed on its first word")
		}
		if !tr.AddWord("world") {
			t.Fatal("the frame did not complete the second time through")
		}
	})

	t.Run("the written text starts over too", func(t *testing.T) {
		const written = "<card>4111</card>"
		tr := NewWordCompletionTracker("4111", "", written)
		tr.AddWord("4111")
		if raw, _ := tr.RawText(); raw != written {
			t.Fatalf("attributed %q, want %q", raw, written)
		}
		tr.Reset()
		tr.AddWord("4111")
		if raw, _ := tr.RawText(); raw != written {
			t.Fatalf("after the reset, attributed %q, want %q again", raw, written)
		}
	})

	t.Run("closing out after a reset sees the whole frame again", func(t *testing.T) {
		tr := NewWordCompletionTracker("number is", "", "")
		tr.AddWord("number")
		tr.AddWord("4111") // closes the frame out with " is" unspoken
		if word, _ := tr.FrameWord(); word != "is" {
			t.Fatalf("frame word = %q, want %q", word, "is")
		}

		tr.Reset()
		tr.AddWord("4111")
		if word, _ := tr.FrameWord(); word != "number is" {
			t.Fatalf("after the reset, frame word = %q, want the whole frame", word)
		}
	})
}

// The edges: nothing to say, nothing said, and a token that covers more than it
// was asked to.
func TestTrackerEdges(t *testing.T) {
	t.Run("a frame with no text is complete already", func(t *testing.T) {
		if !NewWordCompletionTracker("", "", "").IsComplete() {
			t.Fatal("an empty frame is waiting for something to be spoken")
		}
	})

	t.Run("an empty word says nothing", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello", "", "")
		tr.AddWord("")
		if tr.IsComplete() {
			t.Fatal("an empty word completed the frame")
		}
	})

	t.Run("one token can cover the whole frame", func(t *testing.T) {
		if !NewWordCompletionTracker("ab", "", "").AddWord("ab") {
			t.Fatal("the frame did not complete")
		}
	})

	t.Run("a token saying more than the frame holds still completes it", func(t *testing.T) {
		if !NewWordCompletionTracker("Hi", "", "").AddWord("Hieveryone") {
			t.Fatal("the frame did not complete")
		}
	})
}

// What is left of a frame is what closes it out when the synthesizer stops
// reporting, so the conversation still records the text nobody heard reported.
func TestRemainingText(t *testing.T) {
	t.Run("before anything is spoken, all of it is left", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello world", "", "")
		if got := tr.RemainingTTSText(true); got != "hello world" {
			t.Fatalf("what is left = %q, want the whole frame", got)
		}
	})

	t.Run("partway through, the rest is left", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello world", "", "")
		tr.AddWord("hello")
		if got := tr.RemainingTTSText(true); got != "world" {
			t.Fatalf("what is left = %q, want %q", got, "world")
		}
	})

	t.Run("once it is all spoken, nothing is left", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello world", "", "")
		tr.AddWord("hello")
		tr.AddWord("world")
		if got := tr.RemainingTTSText(true); got != "" {
			t.Fatalf("what is left = %q, want nothing", got)
		}
	})

	t.Run("punctuation inside it is kept", func(t *testing.T) {
		tr := NewWordCompletionTracker("Hello, world!", "", "")
		tr.AddWord("Hello,")
		if got := tr.RemainingTTSText(true); got != "world!" {
			t.Fatalf("what is left = %q, want %q", got, "world!")
		}
	})

	t.Run("closing the frame out leaves the text it was closed out with", func(t *testing.T) {
		tr := NewWordCompletionTracker("number is", "", "")
		tr.AddWord("4111")
		if !tr.IsComplete() {
			t.Fatal("the frame was not closed out")
		}
		if got := tr.RemainingTTSText(true); got != "number is" {
			t.Fatalf("what is left = %q, want the text nobody spoke", got)
		}
	})

	t.Run("a reset puts it all back", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello world", "", "")
		tr.AddWord("hello")
		tr.AddWord("world")
		tr.Reset()
		if got := tr.RemainingTTSText(true); got != "hello world" {
			t.Fatalf("what is left = %q, want the whole frame again", got)
		}
	})
}

// The same for the written text, which is what a closed-out frame contributes to
// the conversation.
func TestRemainingWrittenText(t *testing.T) {
	t.Run("with no written text there is none to report", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello world", "", "")
		if _, ok := tr.RemainingRawTextOnly(); ok {
			t.Fatal("a frame with no written text reported some")
		}
	})

	t.Run("before anything is spoken, all of it is left", func(t *testing.T) {
		const written = "<card>4111 1111</card>"
		tr := NewWordCompletionTracker("4111 1111", "", written)
		got, ok := tr.RemainingRawTextOnly()
		if !ok || got != written {
			t.Fatalf("what is left = %q (set=%v), want %q", got, ok, written)
		}
	})

	t.Run("partway through, the tail is left", func(t *testing.T) {
		tr := NewWordCompletionTracker("4111 1111", "", "<card>4111 1111</card>")
		tr.AddWord("4111")
		got, _ := tr.RemainingRawTextOnly()
		if got != "1111</card>" {
			t.Fatalf("what is left = %q, want %q", got, "1111</card>")
		}
	})

	t.Run("once it is all spoken, nothing is left", func(t *testing.T) {
		tr := NewWordCompletionTracker("4111 1111", "", "<card>4111 1111</card>")
		tr.AddWord("4111")
		tr.AddWord("1111")
		if got, _ := tr.RemainingRawTextOnly(); got != "" {
			t.Fatalf("what is left = %q, want nothing", got)
		}
	})

	t.Run("closing the frame out takes all of it", func(t *testing.T) {
		tr := NewWordCompletionTracker("4111 1111 1111 1111", "", "<card>4111 1111 1111 1111</card>")
		tr.AddWord("4111")
		tr.AddWord("WRONG") // closes the frame out
		if !tr.IsComplete() {
			t.Fatal("the frame was not closed out")
		}
		if got, _ := tr.RemainingRawTextOnly(); got != "" {
			t.Fatalf("what is left = %q, want nothing: closing out sweeps the rest", got)
		}
	})

	t.Run("a reset puts it all back", func(t *testing.T) {
		const written = "<card>hello world</card>"
		tr := NewWordCompletionTracker("hello world", "", written)
		tr.AddWord("hello")
		tr.AddWord("world")
		tr.Reset()
		got, _ := tr.RemainingRawTextOnly()
		if got != written {
			t.Fatalf("what is left = %q, want the whole written text again", got)
		}
	})
}

// What a client shows as the bot speaks is the user-facing text, which carries
// neither the synthesis markup nor the model's delimiters. Its cursor is what a
// live caption is drawn from, so what has been said plus what is left has to
// reconstruct the sentence exactly at every step.
func TestUserFacingText(t *testing.T) {
	const spoken = "<spell>4111 1111 1111 1111</spell>"
	const written = "<card>4111 1111 1111 1111</card>"
	const shown = "4111 1111 1111 1111"
	words := []string{"4111", "1111", "1111", "1111"}

	newTracker := func() *WordCompletionTracker {
		return NewWordCompletionTracker(spoken, shown, written)
	}

	t.Run("with none given, it follows the text sent to the synthesizer", func(t *testing.T) {
		tr := NewWordCompletionTracker("hello world", "", "")
		tr.AddWord("hello")
		if got, want := tr.AccumulatedUserFacingText(), tr.AccumulatedTTSText(); got != want {
			t.Fatalf("shown %q, want it to follow the spoken text %q", got, want)
		}
	})

	t.Run("before anything is said, nothing is shown", func(t *testing.T) {
		tr := newTracker()
		if got := tr.AccumulatedUserFacingText(); got != "" {
			t.Fatalf("shown %q, want nothing yet", got)
		}
		if got := tr.RemainingUserFacingText(true); got != shown {
			t.Fatalf("left to say = %q, want the whole sentence", got)
		}
	})

	t.Run("the markup is not shown", func(t *testing.T) {
		tr := newTracker()
		tr.AddWord("4111")
		if got := tr.AccumulatedUserFacingText(); got != "4111" {
			t.Fatalf("shown %q, want %q", got, "4111")
		}
		if got := tr.AccumulatedTTSText(); got != "<spell>4111" {
			t.Fatalf("spoken text consumed = %q, want the markup with it", got)
		}
	})

	t.Run("it advances word by word", func(t *testing.T) {
		tr := newTracker()
		want := []struct{ shown, left string }{
			{"4111", " 1111 1111 1111"},
			{"4111 1111", " 1111 1111"},
			{"4111 1111 1111", " 1111"},
			{"4111 1111 1111 1111", ""},
		}
		for i, w := range words {
			tr.AddWord(w)
			if got := tr.AccumulatedUserFacingText(); got != want[i].shown {
				t.Fatalf("after word %d, shown %q, want %q", i+1, got, want[i].shown)
			}
			if got := tr.RemainingUserFacingText(false); got != want[i].left {
				t.Fatalf("after word %d, left %q, want %q", i+1, got, want[i].left)
			}
		}
	})

	t.Run("what is shown and what is left reconstruct the sentence", func(t *testing.T) {
		tr := newTracker()
		for i, w := range words {
			tr.AddWord(w)
			got := tr.AccumulatedUserFacingText() + tr.RemainingUserFacingText(false)
			if got != shown {
				t.Fatalf("after word %d, the two halves make %q, want %q", i+1, got, shown)
			}
		}
	})

	t.Run("closing the frame out shows all of it", func(t *testing.T) {
		tr := newTracker()
		tr.AddWord("4111")
		tr.AddWord("WRONG") // closes the frame out
		if !tr.IsComplete() {
			t.Fatal("the frame was not closed out")
		}
		if got := tr.AccumulatedUserFacingText(); got != shown {
			t.Fatalf("shown %q, want the whole sentence: it was closed out", got)
		}
		if got := tr.RemainingUserFacingText(true); got != "" {
			t.Fatalf("left to say = %q, want nothing", got)
		}
	})

	t.Run("a reset takes the caption back to the start", func(t *testing.T) {
		tr := newTracker()
		for _, w := range words {
			tr.AddWord(w)
		}
		tr.Reset()
		if got := tr.AccumulatedUserFacingText(); got != "" {
			t.Fatalf("shown %q after the reset, want nothing", got)
		}
		if got := tr.RemainingUserFacingText(true); got != shown {
			t.Fatalf("left to say = %q, want the whole sentence again", got)
		}
	})
}

// A lone angle bracket is content, not the start of markup: "<3" is an
// emoticon. Everything after it has to survive, and it has to be recognized as
// a word in its own right when the synthesizer reports it as one.
func TestALoneAngleBracketIsContent(t *testing.T) {
	const text = "I love you <3 always"

	t.Run("nothing after it is dropped", func(t *testing.T) {
		tr := NewWordCompletionTracker(text, "", "")
		for w := range strings.FieldsSeq(text) {
			tr.AddWord(w)
		}
		if !tr.IsComplete() {
			t.Fatal("the frame did not complete")
		}
		if got := tr.AccumulatedUserFacingText(); got != text {
			t.Fatalf("shown %q, want the whole sentence", got)
		}
	})

	t.Run("it is a word of its own and does not close the frame out", func(t *testing.T) {
		tr := NewWordCompletionTracker(text, "", "")
		for _, w := range []string{"I", "love", "you"} {
			tr.AddWord(w)
		}
		if !tr.WordBelongsHere("<3") {
			t.Fatal("the emoticon was rejected as not belonging to this frame")
		}
		if tr.AddWord("<3") {
			t.Fatal("the frame completed on the emoticon, with a word still to come")
		}
		if over, ok := tr.OverflowWord(); ok && over != "" {
			t.Fatalf("the emoticon was handed on as %q, want it kept here", over)
		}
		if !tr.AddWord("always") {
			t.Fatal("the frame did not complete on its last word")
		}
		if got := tr.AccumulatedUserFacingText(); got != text {
			t.Fatalf("shown %q, want the whole sentence", got)
		}
	})
}

// A synthesis tag with several attributes has spaces inside it, and a provider
// that splits its report on spaces sends the opening tag as several words. Each
// piece belongs to this frame, and only the one that closes the tag finishes it.
func TestSynthesisTagSplitAcrossWords(t *testing.T) {
	const spoken = `My name is <phoneme alphabet="ipa" ph="ʃəˈvɔːn">Siobhan</phoneme>.`
	words := []string{
		"My", "name", "is",
		"<phoneme", `alphabet="ipa"`, `ph="ʃəˈvɔːn">Siobhan</phoneme>.`,
	}

	t.Run("every piece of the tag belongs here", func(t *testing.T) {
		tr := NewWordCompletionTracker(spoken, "", "")
		for _, w := range words[:3] {
			tr.AddWord(w)
		}
		for i, w := range words[3:] {
			if !tr.WordBelongsHere(w) {
				t.Fatalf("%q is part of the open tag and was rejected", w)
			}
			complete := tr.AddWord(w)
			if last := i == len(words[3:])-1; complete != last {
				t.Fatalf("after %q, complete = %v, want %v", w, complete, last)
			}
		}
	})

	t.Run("the opening piece does not close the frame out", func(t *testing.T) {
		tr := NewWordCompletionTracker(spoken, "", "")
		for _, w := range words[:3] {
			tr.AddWord(w)
		}
		if tr.AddWord("<phoneme") {
			t.Fatal("the frame completed on an unclosed tag, with the name still unspoken")
		}
	})

	t.Run("the pieces map back to the one word they stand for", func(t *testing.T) {
		const written = "My name is Siobhan."
		tr := NewWordCompletionTracker(spoken, "", written)
		for _, w := range words[:len(words)-1] {
			tr.AddWord(w)
		}
		if !tr.AddWord(words[len(words)-1]) {
			t.Fatal("the frame did not complete on the piece closing the tag")
		}
		if raw, _ := tr.RawText(); raw != "Siobhan." {
			t.Fatalf("attributed %q, want the written word %q", raw, "Siobhan.")
		}
	})
}

// A synthesizer may report a symbol as a different one: an arrow comes back as
// a hyphen. The word still belongs where the symbol sits, and mistaking it for
// a word of the next frame would close this one out with most of it unsaid.
func TestASymbolReportedAsAnother(t *testing.T) {
	const sentence = "- Example route: São Paulo → Santiago (Chile) → Auckland (New Zealand)."
	// Both arrows come back as hyphens.
	spokenWords := []string{
		"-", "Example", "route:", "São", "Paulo",
		"-", "Santiago", "(Chile)",
		"-", "Auckland", "(New", "Zealand).",
	}

	t.Run("the substitute belongs where the symbol is", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, "", "")
		for _, w := range []string{"-", "Example", "route:", "São", "Paulo"} {
			tr.AddWord(w)
		}
		if !tr.WordBelongsHere("-") {
			t.Fatal("the substituted symbol was rejected as not belonging here")
		}
	})

	t.Run("it does not close the frame out", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, "", "")
		for _, w := range []string{"-", "Example", "route:", "São", "Paulo"} {
			tr.AddWord(w)
		}
		if tr.AddWord("-") || tr.IsComplete() {
			t.Fatal("the frame was closed out on the substituted symbol, losing the rest " +
				"of the sentence")
		}
	})

	t.Run("the second one is recognized too", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, "", "")
		for _, w := range []string{"-", "Example", "route:", "São", "Paulo", "-", "Santiago", "(Chile)"} {
			tr.AddWord(w)
		}
		if !tr.WordBelongsHere("-") {
			t.Fatal("the second substituted symbol was rejected")
		}
	})

	t.Run("the whole sentence completes, and only on its last word", func(t *testing.T) {
		tr := NewWordCompletionTracker(sentence, "", "")
		for _, w := range spokenWords[:len(spokenWords)-1] {
			if tr.AddWord(w) {
				t.Fatalf("the frame completed early, on %q", w)
			}
		}
		if !tr.AddWord(spokenWords[len(spokenWords)-1]) {
			t.Fatal("the frame did not complete on its last word")
		}
	})
}
