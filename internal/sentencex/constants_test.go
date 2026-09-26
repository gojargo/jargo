package sentencex

import (
	"regexp"
	"slices"
	"testing"
	"unicode"
)

func quoteMatches(text string) []string {
	out := []string{}
	for _, m := range quotesFindAll(text) {
		out = append(out, text[m[0]:m[1]])
	}
	return out
}

func TestIsSentenceTerminatorMatchesTable(t *testing.T) {
	for _, c := range globalSentenceTerminators {
		if !isSentenceTerminator(c) {
			t.Errorf("%q is in the table but isSentenceTerminator returned false", c)
		}
	}
	for _, c := range []rune{'a', 'Z', '0', ' ', ',', ';', ':', 'é', '文', '«'} {
		if isSentenceTerminator(c) {
			t.Errorf("%q wrongly reported as a terminator", c)
		}
	}
}

func TestQuotesGreekBasic(t *testing.T) {
	got := quoteMatches("Ο γιατρός είπε: «Η κατάσταση είναι σταθερή».")
	if want := []string{"«Η κατάσταση είναι σταθερή»"}; !slices.Equal(got, want) {
		t.Errorf("matches = %q, want %q", got, want)
	}
}

func TestQuotesGreekMultiple(t *testing.T) {
	got := quoteMatches("«Καλημέρα» είπε. «Πώς είσαι;»")
	if want := []string{"«Καλημέρα»", "«Πώς είσαι;»"}; !slices.Equal(got, want) {
		t.Errorf("matches = %q, want %q", got, want)
	}
}

func TestQuotesGreekMultiline(t *testing.T) {
	got := quoteMatches("Παράδειγμα:\n«Πρώτη γραμμή\nΔεύτερη γραμμή» τέλος.")
	if want := []string{"«Πρώτη γραμμή\nΔεύτερη γραμμή»"}; !slices.Equal(got, want) {
		t.Errorf("matches = %q, want %q", got, want)
	}
}

func TestQuotesOtherDoubleQuotes(t *testing.T) {
	got := quoteMatches(`He said, "Hello" and left.`)
	if want := []string{`"Hello"`}; !slices.Equal(got, want) {
		t.Errorf("matches = %q, want %q", got, want)
	}
}

func TestQuotesAllQuoteTypes(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		{"Standard double: \"Hello\"", []string{"\"Hello\""}},
		{"Greek: «Γεια σου»", []string{"«Γεια σου»"}},
		{"Curved single: 'Hi'", []string{"'Hi'"}},
		{"German-style: „Hallo“", []string{"„Hallo“"}},
		{"Single angular: ‹Bonjour›", []string{"‹Bonjour›"}},
		{"CJK: 「こんにちは」", []string{"「こんにちは」"}},
		{"Chinese: 《你好》", []string{"《你好》"}},
		{"LaTeX double: ``Hello''", []string{"``Hello''"}},
		{"LaTeX single: `Hello'", []string{"`Hello'"}},
		{"Single backtick: `Hello`", []string{"`Hello`"}},
		{"Double backtick: ``Hello``", []string{"``Hello``"}},
		{"Double apostrophe: ''Hello''", []string{"''Hello''"}},
	}
	for _, tt := range tests {
		if got := quoteMatches(tt.text); !slices.Equal(got, tt.want) {
			t.Errorf("matches(%q) = %q, want %q", tt.text, got, tt.want)
		}
	}
}

func TestQuotesEdgeCases(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		// Empty quotes.
		{"Empty: «»", []string{"«»"}},
		// Nested quotes match the outer ones.
		{"Nested: «Outer 'inner' quotes»", []string{"«Outer 'inner' quotes»"}},
		// No quotes.
		{"No quotes here at all.", []string{}},
	}
	for _, tt := range tests {
		if got := quoteMatches(tt.text); !slices.Equal(got, tt.want) {
			t.Errorf("matches(%q) = %q, want %q", tt.text, got, tt.want)
		}
	}
}

func TestQuotesWithPunctuation(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		{"Question: «Πώς είσαι;»", []string{"«Πώς είσαι;»"}},
		{"Exclamation: «Γεια σου!»", []string{"«Γεια σου!»"}},
		{"Period: «Καλημέρα.»", []string{"«Καλημέρα.»"}},
		{"Complex: «Ελα, πώς είσαι; Καλά!»", []string{"«Ελα, πώς είσαι; Καλά!»"}},
	}
	for _, tt := range tests {
		if got := quoteMatches(tt.text); !slices.Equal(got, tt.want) {
			t.Errorf("matches(%q) = %q, want %q", tt.text, got, tt.want)
		}
	}
}

func TestQuotesWordBoundaryGuards(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		// Contractions and possessives are not quotes.
		{"It wasn't Tom's fault, wasn't it?", []string{}},
		// A space just inside a guarded pair defeats the guards...
		{"He said ' hello ' twice", []string{}},
		// ...unless the pair spans a whole line.
		{"' Hello, world. '\nnext", []string{"' Hello, world. '"}},
		// The guards use Unicode word characters, not only ASCII ones.
		{"é'x'", []string{}},
		{"say 'été' now", []string{"'été'"}},
	}
	for _, tt := range tests {
		if got := quoteMatches(tt.text); !slices.Equal(got, tt.want) {
			t.Errorf("matches(%q) = %q, want %q", tt.text, got, tt.want)
		}
	}
}

func TestPerlWordClassMatchesIsWordChar(t *testing.T) {
	re := regexp.MustCompile(`^[` + perlWordClass + `]$`)
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if r >= 0xD800 && r <= 0xDFFF {
			continue
		}
		if got, want := re.MatchString(string(r)), isWordChar(r); got != want {
			t.Fatalf("U+%04X: class match %v, isWordChar %v", r, got, want)
		}
	}
}
