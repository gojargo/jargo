package language_test

import (
	"testing"

	"github.com/gojargo/jargo/language"
)

func TestCode(t *testing.T) {
	cases := map[language.Language]string{
		language.English:      "en",
		language.EnglishUS:    "en-US",
		language.FrenchCA:     "fr-CA",
		language.ChineseCN:    "zh-CN",
		language.Language(""): "",
	}
	for lang, want := range cases {
		if got := lang.Code(); got != want {
			t.Errorf("Language(%q).Code() = %q, want %q", string(lang), got, want)
		}
	}
}

func TestBaseCode(t *testing.T) {
	cases := []struct {
		lang language.Language
		want string
	}{
		{language.English, "en"},   // already base-only
		{language.EnglishUS, "en"}, // region stripped
		{language.EnglishGB, "en"},
		{language.French, "fr"},
		{language.FrenchCA, "fr"},
		{language.SpanishMX, "es"},
		{language.PortugueseBR, "pt"},
		{language.DutchBE, "nl"},
		{language.ChineseCN, "zh"},
		{language.ChineseTW, "zh"},
		{language.Japanese, "ja"},
		{language.Filipino, "fil"}, // three-letter base, no region
		{language.Language(""), ""},
		{language.Language("fr-CA-variant"), "fr"}, // only the first subtag is the base
	}
	for _, c := range cases {
		if got := c.lang.BaseCode(); got != c.want {
			t.Errorf("Language(%q).BaseCode() = %q, want %q", string(c.lang), got, c.want)
		}
	}
}

// TestBaseCodeMatchesCodeForRegionless checks that a region-less language's
// BaseCode and Code agree.
func TestBaseCodeMatchesCodeForRegionless(t *testing.T) {
	for _, l := range []language.Language{language.German, language.Italian, language.Korean, language.Arabic} {
		if l.BaseCode() != l.Code() {
			t.Errorf("Language(%q): BaseCode()=%q != Code()=%q", string(l), l.BaseCode(), l.Code())
		}
	}
}
