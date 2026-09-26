package deepgram

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/service/tts"
)

func TestFormatPronunciation(t *testing.T) {
	if got, ok := FormatPronunciation("dupilumab", "duːˈpɪljuːmæb"); !ok ||
		got != `\{"word": "dupilumab", "pronounce": "duːpˈɪljuːmæb"\}` {
		t.Fatalf("got %q, %v", got, ok)
	}
	a, _ := FormatPronunciation("Achoo", "/əˈʧu/")
	b, _ := FormatPronunciation("Achoo", "ə'tʃu")
	if a != b {
		t.Fatalf("notation variants format differently: %q and %q", a, b)
	}
	if got, _ := FormatPronunciation(`say "hi"`, "haɪ"); got != `\{"word": "say \"hi\"", "pronounce": "haɪ"\}` {
		t.Fatalf("quotes = %q", got)
	}
}

func TestFormatPronunciationUnusable(t *testing.T) {
	cases := map[string]struct{ word, ipa string }{
		"empty":                    {"Crete", ""},
		"far longer than the word": {"ab", strings.Repeat("ˈkʌɹənt", 3)},
		"over 128 characters":      {"dupilumab", strings.Repeat("duːˈpɪljuːmæb", 12)},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if _, ok := FormatPronunciation(tc.word, tc.ipa); ok {
				t.Fatal("the pronunciation was used")
			}
		})
	}
	if _, ok := FormatPronunciation("a", "əˈbaʊtðæt"); !ok {
		t.Fatal("a short word did not get the length floor")
	}
}

func TestTheServiceFormatsPronunciations(t *testing.T) {
	want, _ := FormatPronunciation("Crete", "kriːt")
	if got, _ := (&synthesizer{}).FormatPronunciation("Crete", "kriːt"); got != want {
		t.Fatalf("the service formats %q, want %q", got, want)
	}
}

// Flux has no pronunciation markup.
func TestFluxTakesNoPronunciations(t *testing.T) {
	if _, ok := any(&fluxSynth{}).(tts.PronunciationFormatter); ok {
		t.Fatal("the Flux service takes pronunciation hints")
	}
}
