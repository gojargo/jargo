package text_test

import (
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

// A boundary is only reported once the text holds a complete sentence, and an
// abbreviation is not one. Telling those apart is the whole reason language
// rules are used rather than a scan for a period.
func TestMatchEndOfSentence(t *testing.T) {
	cases := []struct {
		text string
		want int
	}{
		{"", 0},
		{"no period here", 0},
		{"Hello world.", len("Hello world.")},
		{"Hello world. And", len("Hello world.")},
		{"$29. Next", len("$29.")},
		// The title must not end the sentence.
		{"Dr. Smith is here. Next", len("Dr. Smith is here.")},
		{"e.g. this is fine. Then", len("e.g. this is fine.")},
	}
	for _, c := range cases {
		if got := text.MatchEndOfSentence(c.text, ""); got != c.want {
			t.Errorf("MatchEndOfSentence(%q) = %d, want %d", c.text, got, c.want)
		}
	}
}

// A script the English rules do not cover still ends on its own punctuation,
// which needs no disambiguation.
func TestUnambiguousScripts(t *testing.T) {
	cases := []struct {
		text string
		want int
	}{
		{"こんにちは。", len("こんにちは。")},
		{"你好。 再见", len("你好。")},
		{"नमस्ते।", len("नमस्ते।")},
	}
	for _, c := range cases {
		if got := text.MatchEndOfSentence(c.text, ""); got != c.want {
			t.Errorf("MatchEndOfSentence(%q) = %d, want %d", c.text, got, c.want)
		}
	}
}

// Every mark that can end a sentence is recognized, across scripts.
func TestIsSentenceEnding(t *testing.T) {
	for _, r := range []rune{'.', '!', '?', ';', '…', '。', '？', '।', '؟', '።'} {
		if !text.IsSentenceEnding(r) {
			t.Errorf("IsSentenceEnding(%q) = false, want true", r)
		}
	}
	for _, r := range []rune{'a', ',', ':', '-'} {
		if text.IsSentenceEnding(r) {
			t.Errorf("IsSentenceEnding(%q) = true, want false", r)
		}
	}
}
