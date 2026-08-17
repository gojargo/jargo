package language_test

import (
	"os"
	"regexp"
	"strings"
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

// TestResolveUsesTheVerifiedCode checks a language the service was verified
// against is sent the code the service was verified with, whatever form that
// takes.
func TestResolveUsesTheVerifiedCode(t *testing.T) {
	codes := map[language.Language]string{
		language.English:   "en",
		language.SpanishES: "es-419",
	}

	if got := language.Resolve(language.English, codes, true); got != "en" {
		t.Errorf("Resolve(English) = %q, want %q", got, "en")
	}
	// The map wins over anything derivable: the service's own code need not
	// resemble the language it stands for.
	if got := language.Resolve(language.SpanishES, codes, true); got != "es-419" {
		t.Errorf("Resolve(SpanishES) = %q, want %q", got, "es-419")
	}
}

// TestResolveFallsBackToTheBaseCode checks a service taking base codes is sent
// one for a language it was not verified against, rather than nothing.
func TestResolveFallsBackToTheBaseCode(t *testing.T) {
	codes := map[language.Language]string{language.English: "en"}

	if got := language.Resolve(language.EnglishGB, codes, true); got != "en" {
		t.Errorf("Resolve(EnglishGB) = %q, want %q", got, "en")
	}
}

// TestResolveFallsBackToTheFullCode checks a service taking full codes is sent
// the language as it stands.
func TestResolveFallsBackToTheFullCode(t *testing.T) {
	codes := map[language.Language]string{language.EnglishUS: "en-US"}

	if got := language.Resolve(language.EnglishGB, codes, false); got != "en-GB" {
		t.Errorf("Resolve(EnglishGB) = %q, want %q", got, "en-GB")
	}
}

// TestResolveWithNoMap checks a service with nothing verified still sends
// something, which is what makes the map optional.
func TestResolveWithNoMap(t *testing.T) {
	if got := language.Resolve(language.FrenchCA, nil, true); got != "fr" {
		t.Errorf("Resolve(FrenchCA) = %q, want %q", got, "fr")
	}
	if got := language.Resolve(language.FrenchCA, nil, false); got != "fr-CA" {
		t.Errorf("Resolve(FrenchCA) = %q, want %q", got, "fr-CA")
	}
}

// TestTheCatalogueIsWhole checks what holds of every language in it: a
// non-empty code whose base is lower-case, and a base that is itself a usable
// language code.
//
// It deliberately does not require a regional language to have a region-less
// sibling. Several do not: Inuktitut is only named by script (iu-Cans, iu-Latn)
// and Xhosa only as xh-ZA, and inventing the bare codes to tidy that up would be
// naming languages no service was verified against.
func TestTheCatalogueIsWhole(t *testing.T) {
	all := language.All()
	if len(all) < 300 {
		t.Fatalf("the catalog names %d languages, want the whole of it", len(all))
	}

	for _, l := range all {
		code := l.Code()
		if code == "" {
			t.Error("the catalog names a language with no code")
			continue
		}
		base := l.BaseCode()
		if base == "" {
			t.Errorf("%q: no base code", code)
		}
		if base != strings.ToLower(base) {
			t.Errorf("%q: the base code %q is not lower-case", code, base)
		}
		if strings.Contains(base, "-") {
			t.Errorf("%q: the base code %q still carries a subtag", code, base)
		}
	}
}

// TestEveryLanguageNamesOneCode checks no two constants carry the same code,
// which would make one of them unreachable through a service's map.
func TestEveryLanguageNamesOneCode(t *testing.T) {
	seen := make(map[string]bool)
	for _, l := range language.All() {
		if seen[l.Code()] {
			t.Errorf("%q is named twice", l.Code())
		}
		seen[l.Code()] = true
	}
}

// TestAllNamesEveryConstant checks the catalog slice and the constants cannot
// drift apart. They are generated together, and this is what says they still
// agree: a constant added without being listed would be invisible to anything
// iterating the languages.
func TestAllNamesEveryConstant(t *testing.T) {
	src, err := os.ReadFile("language.go")
	if err != nil {
		t.Fatal(err)
	}
	declared := regexp.MustCompile(`(?m)^\t([A-Za-z0-9]+)\s+Language = "([^"]+)"`).FindAllStringSubmatch(string(src), -1)
	if len(declared) == 0 {
		t.Fatal("no language constants found; has the file moved?")
	}

	listed := make(map[string]bool, len(language.All()))
	for _, l := range language.All() {
		listed[l.Code()] = true
	}
	for _, d := range declared {
		if !listed[d[2]] {
			t.Errorf("%s (%q) is declared but not in All()", d[1], d[2])
		}
	}
	if len(listed) != len(declared) {
		t.Errorf("All() has %d languages, %d are declared", len(listed), len(declared))
	}
}
