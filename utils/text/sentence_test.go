package text_test

import (
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

// A boundary is only reported once the text holds a complete sentence, and an
// abbreviation is not one. Telling those apart is the whole reason a trained
// model is used rather than a scan for a period.
func TestPunktMatchEndOfSentence(t *testing.T) {
	tok, err := text.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
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
		if got := tok.MatchEndOfSentence(c.text); got != c.want {
			t.Errorf("MatchEndOfSentence(%q) = %d, want %d", c.text, got, c.want)
		}
	}
}

// A script the model has no training for still ends on its own punctuation,
// which needs no disambiguation.
func TestPunktUnambiguousScripts(t *testing.T) {
	tok, err := text.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		text string
		want int
	}{
		{"こんにちは。", len("こんにちは。")},
		{"你好。 再见", len("你好。")},
		{"नमस्ते।", len("नमस्ते।")},
	}
	for _, c := range cases {
		if got := tok.MatchEndOfSentence(c.text); got != c.want {
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

// A tokenizer for another language is built from the training data the Punkt
// project publishes; data that is not a trained model is refused rather than
// producing a tokenizer that finds no boundaries.
func TestNewPunktRejectsInvalidTraining(t *testing.T) {
	if _, err := text.NewPunkt([]byte("not a trained model")); err == nil {
		t.Fatal("NewPunkt accepted training data that is not a model")
	}
}
