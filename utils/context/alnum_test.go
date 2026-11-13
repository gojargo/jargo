package context

import (
	"testing"

	ttstext "github.com/gojargo/jargo/utils/text"
)

func TestNormalize(t *testing.T) {
	cases := map[string]string{
		"Hello, World!":      "helloworld",
		"$42.50":             "4250",
		"<spell>SQL</spell>": "sql",
		"café":               "cafe",
		"A P I":              "api",
	}
	for in, want := range cases {
		if got := alnumOnly(in); got != want {
			t.Errorf("alnumOnly(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestAdvanceByAlnumsSkipsTagsAndTrailingPunct(t *testing.T) {
	runes := []rune("questions, next")
	// Consume the 9 alnum chars of "questions"; the trailing comma is absorbed
	// but the following space is not.
	pos := advanceByAlnums(runes, 0, 9)
	if got := string(runes[:pos]); got != "questions," {
		t.Fatalf("advanced span = %q, want %q", got, "questions,")
	}
}

func TestAdvanceByAlnumsSkipsLeadingTag(t *testing.T) {
	runes := []rune("<b>hi</b>")
	// The leading tag is skipped without spending budget, and the closing tag
	// written against "hi" goes with it.
	pos := advanceByAlnums(runes, 0, 2) // "hi"
	if got := string(runes[:pos]); got != "<b>hi</b>" {
		t.Fatalf("span = %q, want %q", got, "<b>hi</b>")
	}
}

func TestAdvanceByAlnumsLeavesATagForTheNextWord(t *testing.T) {
	runes := []rune("<b>hi")
	// With nothing consumed, a tag is left for the word it opens.
	if pos := advanceByAlnums(runes, 0, 0); pos != 0 {
		t.Fatalf("pos = %d, want 0", pos)
	}
}

func TestStripTrailingPunctuation(t *testing.T) {
	if got := stripTrailingPunctuation("account.!?"); got != "account" {
		t.Fatalf("got %q, want %q", got, "account")
	}
	if got := stripTrailingPunctuation("plain"); got != "plain" {
		t.Fatalf("got %q, want %q", got, "plain")
	}
}

func TestFoldCaseAndAccentsPreservesLength(t *testing.T) {
	in := "Café-Ω!"
	got := foldForMatching(in)
	if len([]rune(got)) != len([]rune(in)) {
		t.Fatalf("folded %q -> %q changed rune length", in, got)
	}
	if got != "cafe-ω!" {
		t.Fatalf("fold = %q, want %q", got, "cafe-ω!")
	}
}

// alnumOf runs transform over text and returns the alphanumeric sequence of the
// input and of the output.
func alnumOf(transform ttstext.Transform, text string) (before, after string) {
	return alnumOnly(text), alnumOnly(transform(text))
}

func replaceText(t *testing.T, pattern, with string) ttstext.Transform {
	t.Helper()
	tr, err := ttstext.ReplaceText([][2]string{{pattern, with}})
	if err != nil {
		t.Fatalf("ReplaceText: %v", err)
	}
	return tr
}

// The transforms that only reshape the text (Markdown stripping, phone number
// and acronym spacing) leave its alphanumeric sequence as it was, so word
// tracking needs no segment map for them.
func TestAlnumPreservingTransformsKeepTheAlnumSequence(t *testing.T) {
	cases := []struct {
		name      string
		transform ttstext.Transform
		text      string
	}{
		{"StripMarkdown", ttstext.StripMarkdown, "**Hello** and _world_"},
		{"ExpandPhoneNumbers", ttstext.ExpandPhoneNumbers, "123-456-7890"},
		{"NormalizeAcronyms", ttstext.NormalizeAcronyms, "NASA and HTTP"},
	}
	for _, c := range cases {
		if before, after := alnumOf(c.transform, c.text); after != before {
			t.Errorf("%s(%q): alnum %q, want %q", c.name, c.text, after, before)
		}
	}
}

// Spelling a percentage out changes its alphanumeric sequence.
func TestExpandPercentagesChangesTheAlnumSequence(t *testing.T) {
	if before, after := alnumOf(ttstext.ExpandPercentages, "50%"); after == before {
		t.Errorf("ExpandPercentages(%q): alnum %q unchanged, want it to differ", "50%", after)
	}
}

// A replacement that splits one word into several normalizes the same as the
// original, since the alphanumeric sequence drops spaces and case.
func TestAWordSplittingReplacementNormalizesTheSame(t *testing.T) {
	tr := replaceText(t, `\bBODYPUMP\b`, "body pump")
	if before, after := alnumOf(tr, "BODYPUMP"); after != before {
		t.Errorf("alnum after the split = %q, want %q", after, before)
	}
}

// A replacement that only changes the case of a word normalizes the same as
// the original.
func TestACaseOnlyReplacementNormalizesTheSame(t *testing.T) {
	tr := replaceText(t, `\bSQL\b`, "sql")
	if before, after := alnumOf(tr, "SQL"); after != before {
		t.Errorf("alnum after the case change = %q, want %q", after, before)
	}
}

// A respelling ("leisure" to "lezher") changes the letters, so it does not
// normalize the same and is seen as transformed.
func TestARespellingDoesNotNormalizeTheSame(t *testing.T) {
	tr := replaceText(t, `\bleisure\b`, "lezher")
	if before, after := alnumOf(tr, "leisure"); after == before {
		t.Errorf("alnum after the respelling = %q, want it to differ from %q", after, before)
	}
}

// An SSML phoneme tag wraps the word without altering it, so once the tag is
// stripped it normalizes the same as the original.
func TestAnSSMLPhonemeTagNormalizesTheSame(t *testing.T) {
	tr := replaceText(t, `\bSiobhan\b`, `<phoneme alphabet="ipa" ph="ʃəˈvɔːn">Siobhan</phoneme>`)
	if before, after := alnumOf(tr, "Siobhan"); after != before {
		t.Errorf("alnum after the phoneme tag = %q, want %q", after, before)
	}
}
