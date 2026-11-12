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

// TestMatchEndOfSentenceKeepsAListMarkerWithItsItem checks a numbered-list
// marker at the start of a line is not a sentence of its own, while a number
// ending a sentence still ends it.
func TestMatchEndOfSentenceKeepsAListMarkerWithItsItem(t *testing.T) {
	for _, c := range []struct {
		text string
		want int
	}{
		{"1. T", 0},
		{"  12. The", 0},
		{"Options:\n1. Take", 0},
		{"1. Take a walk.\n2. Read", 15},
		{"Pick one.\n1. Walk", 9},
		// Numbers that end a sentence still split.
		{"The answer is 1. Next", 16},
		{"1990.\nThat was the year", 5},
		// The marker stays blanked when an unfinished quote triggers a recheck.
		{`1. She said, "Dr. Smith`, 0},
	} {
		if got := text.MatchEndOfSentence(c.text, ""); got != c.want {
			t.Errorf("MatchEndOfSentence(%q) = %d, want %d", c.text, got, c.want)
		}
	}
	// Languages whose rules never start a sentence at a digit still split
	// between items.
	for _, lang := range []string{"de", "da", "fi", "cs", "sk", "ca"} {
		if got := text.MatchEndOfSentence("1. Zug nehmen.\n2. Auto", lang); got != 14 {
			t.Errorf("%s: MatchEndOfSentence = %d, want 14", lang, got)
		}
	}
}

// TestMatchEndOfSentenceSplitsNumberedListsInEveryLanguage checks a list item
// ends where the next one starts, whatever the language and its digits.
func TestMatchEndOfSentenceSplitsNumberedListsInEveryLanguage(t *testing.T) {
	for _, c := range []struct{ lang, first, rest string }{
		{"en", "Options:\n1. Take the train.", "\n2. Drive"},
		{"fr", "Voici les options :\n1. Prendre le train.", "\n2. Prendre"},
		{"de", "Optionen:\n1. Den Zug nehmen.", "\n2. Mit"},
		{"es", "Opciones:\n1. Tomar el tren.", "\n2. Ir"},
		{"it", "Opzioni:\n1. Prendi il treno.", "\n2. Vai"},
		{"pt", "Opções:\n1. Pegar o trem.", "\n2. Ir"},
		{"ru", "Варианты:\n1. Поехать на поезде.", "\n2. Поехать"},
		{"hi", "विकल्प:\n1. ट्रेन लें।", "\n2. कार"},
		{"hi", "विकल्प:\n१. ट्रेन लें।", "\n२. कार"},
		{"ar", "الخيارات:\n1. ركوب القطار.", "\n2. قيادة"},
		{"ar", "الخيارات:\n١. ركوب القطار.", "\n٢. قيادة"},
		{"ja", "選択肢：\n1. 電車に乗る。", "\n2. 車で"},
		{"zh", "选项：\n1. 坐火车。", "\n2. 开车"},
	} {
		if got := text.MatchEndOfSentence(c.first+c.rest, c.lang); got != len(c.first) {
			t.Errorf("%s %q: MatchEndOfSentence = %d, want %d", c.lang, c.first+c.rest, got, len(c.first))
		}
	}
}
