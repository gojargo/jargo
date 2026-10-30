package context

import "testing"

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
