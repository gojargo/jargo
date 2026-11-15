package kokoro

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

// Tests for the espeak-ng binding and the phonemization over it. The ones that
// phonemize load the system's libespeak-ng, and skip when it is absent.

// systemEspeak is the system's libespeak-ng, or a skip when there is none.
func systemEspeak(t *testing.T) *espeakLibrary {
	t.Helper()
	lib, err := openEspeak("", "")
	if errors.Is(err, ErrEspeakNotAvailable) {
		t.Skipf("libespeak-ng not available: %v", err)
	}
	if err != nil {
		t.Fatalf("openEspeak: %v", err)
	}
	return lib
}

// backend is an espeak backend for lang with the options given.
func backend(t *testing.T, lang string, opts phonemizerOptions) *espeakBackend {
	t.Helper()
	b, err := newEspeakBackend(systemEspeak(t), lang, opts)
	if err != nil {
		t.Fatalf("newEspeakBackend(%s): %v", lang, err)
	}
	return b
}

// phonemizeLines phonemizes text with b, failing the test on an error.
func phonemizeLines(t *testing.T, b *espeakBackend, text []string, strip bool) []string {
	t.Helper()
	out, err := b.phonemizeLines(text, defaultSeparator, strip)
	if err != nil {
		t.Fatalf("phonemize %q: %v", text, err)
	}
	return out
}

// The voice struct is read at the offsets the library writes it at on a
// 64-bit platform.
func TestEspeakVoiceLayout(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("the layout is checked for 64-bit platforms")
	}
	var v espeakVoiceC
	if unsafe.Sizeof(v) != 40 || unsafe.Offsetof(v.languages) != 8 || unsafe.Offsetof(v.identifier) != 16 {
		t.Errorf("espeak_VOICE layout: size %d, languages at %d, identifier at %d",
			unsafe.Sizeof(v), unsafe.Offsetof(v.languages), unsafe.Offsetof(v.identifier))
	}
}

func TestParseEspeakVersion(t *testing.T) {
	for in, want := range map[string][]int{
		"1.52.0":                {1, 52, 0},
		"1.49.3-dev  2019":      {1, 49, 3},
		"1.51.1 2022-01-01 xyz": {1, 51, 1},
	} {
		got, err := parseEspeakVersion(in)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("parseEspeakVersion(%q) = %v, %v, want %v", in, got, err, want)
		}
	}
	if compareVersions([]int{1, 48, 3}, []int{1, 48, 3}) != 0 || compareVersions([]int{1, 48}, []int{1, 48, 3}) >= 0 ||
		compareVersions([]int{1, 52, 0}, []int{1, 49}) <= 0 {
		t.Error("compareVersions does not order versions as tuples")
	}
}

// A library path that names nothing is reported as a missing library.
func TestAMissingEspeakIsReported(t *testing.T) {
	_, err := openEspeak(t.TempDir()+"/libespeak-ng.so.1", "")
	if !errors.Is(err, ErrEspeakNotAvailable) {
		t.Fatalf("err = %v, want ErrEspeakNotAvailable", err)
	}
}

func TestEspeakDataPathMustHoldData(t *testing.T) {
	if _, err := espeakDataPathArgument(t.TempDir(), []int{1, 52}); !errors.Is(err, errEspeakData) {
		t.Fatalf("err = %v, want errEspeakData", err)
	}
}

func TestEspeakEnglish(t *testing.T) {
	b := backend(t, "en-us", phonemizerOptions{})
	got := phonemizeLines(t, b, []string{"hello world", "goodbye", "third line", "yet another"}, true)
	if want := []string{"həloʊ wɜːld", "ɡʊdbaɪ", "θɜːd laɪn", "jɛt ɐnʌðɚ"}; !reflect.DeepEqual(got, want) {
		t.Errorf("phonemize = %q, want %q", got, want)
	}
}

func TestEspeakStress(t *testing.T) {
	got := phonemizeLines(t, backend(t, "en-us", phonemizerOptions{}), []string{"hello world"}, true)
	if got[0] != "həloʊ wɜːld" {
		t.Errorf("without stress = %q", got[0])
	}
	got = phonemizeLines(t, backend(t, "en-us", phonemizerOptions{withStress: true}), []string{"hello world"}, true)
	if got[0] != "həlˈoʊ wˈɜːld" {
		t.Errorf("with stress = %q", got[0])
	}
}

func TestEspeakPunctuation(t *testing.T) {
	text := []string{"hello, world!"}
	for _, tc := range []struct {
		preserve, strip bool
		want            string
	}{
		{false, true, "həloʊ wɜːld"},
		{true, true, "həloʊ, wɜːld!"},
		{false, false, "həloʊ wɜːld "},
		{true, false, "həloʊ, wɜːld! "},
	} {
		b := backend(t, "en-us", phonemizerOptions{preservePunctuation: tc.preserve})
		if got := phonemizeLines(t, b, text, tc.strip)[0]; got != tc.want {
			t.Errorf("preserve=%v strip=%v: %q, want %q", tc.preserve, tc.strip, got, tc.want)
		}
	}
}

func TestEspeakMarksOnly(t *testing.T) {
	b := backend(t, "en-us", phonemizerOptions{preservePunctuation: true})
	for in, want := range map[string]string{"!'": "! ", "'!": "! ", "!'!": "!! ", "'!'": "! "} {
		if got := phonemizeLines(t, b, []string{in}, false)[0]; got != want {
			t.Errorf("phonemize(%q) = %q, want %q", in, got, want)
		}
	}
	// A line made of marks only neither merges with the next ones nor drops
	// the word separator ending them.
	if got := phonemizeLines(t, b, []string{"a", "!", "?"}, false); !reflect.DeepEqual(got, []string{"eɪ ", "! ", "? "}) {
		t.Errorf("phonemize = %q", got)
	}
	if got := phonemizeLines(t, b, []string{"!", "hello", "a."}, false); !reflect.DeepEqual(got,
		[]string{"! ", "həloʊ ", "eɪ. "}) {
		t.Errorf("phonemize = %q", got)
	}
	if got := phonemizeLines(t, b, []string{`"Hey! "`, `"hey,"`}, false); !reflect.DeepEqual(got,
		[]string{`"heɪ! " `, `"heɪ," `}) {
		t.Errorf("phonemize = %q", got)
	}
}

// The words around quotes and brackets are pronounced as without
// punctuation, in context.
func TestEspeakContext(t *testing.T) {
	for _, tc := range []struct{ lang, text, want string }{
		{"fr-fr", "les (amis)", "lez (ami) "},
		{"fr-fr", "les « amis » de « mes amis »", "lez « ami » də « mez ami » "},
		{"en-us", "an (apple)", "ɐn (æpəl) "},
		{"en-us", `"hello" world, (an) apple.`, `"həloʊ" wɜːld, (ɐn) æpəl. `},
		// espeak joins "do not" in a single word: the marks are placed as when
		// phonemizing each chunk alone.
		{"en-us", "I do (not) know", "aɪ duː (nɑːt) noʊ "},
		{"en-us", "not (a) dog", "nɑːt (eɪ) dɑːɡ "},
	} {
		b := backend(t, tc.lang, phonemizerOptions{preservePunctuation: true})
		if got := phonemizeLines(t, b, []string{tc.text}, false)[0]; got != tc.want {
			t.Errorf("%s %q = %q, want %q", tc.lang, tc.text, got, tc.want)
		}
	}
}

// A switch to another language keeps its flags.
func TestEspeakLanguageSwitchKeepsFlags(t *testing.T) {
	b := backend(t, "fr-fr", phonemizerOptions{preservePunctuation: true})
	got := phonemizeLines(t, b, []string{"le (football) et le weekend", "le football"}, false)
	want := []string{"lə ((en)fʊtbɔːl(fr)) e lə (en)wiːkɛnd(fr) ", "lə (en)fʊtbɔːl(fr) "}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("phonemize = %q, want %q", got, want)
	}
}

func TestEspeakUnsupportedLanguage(t *testing.T) {
	_, err := newEspeakBackend(systemEspeak(t), "xx-nope", phonemizerOptions{})
	if !errors.Is(err, errUnsupportedLanguage) {
		t.Fatalf("err = %v, want errUnsupportedLanguage", err)
	}
	// Language codes are compared ignoring case.
	if _, err := newEspeakBackend(systemEspeak(t), "EN-US", phonemizerOptions{}); err != nil {
		t.Fatalf("EN-US: %v", err)
	}
}

// The tokenizer phonemizes the way the model reads: punctuation and stress
// kept, what the vocabulary does not hold dropped, and the result trimmed.
func TestTokenizerPhonemize(t *testing.T) {
	tok := newTokenizer(systemEspeak(t))
	got, err := tok.phonemize("  Hello, world! ", "en-us")
	if err != nil {
		t.Fatalf("phonemize: %v", err)
	}
	if got != "həlˈoʊ, wˈɜːld!" {
		t.Errorf("phonemize = %q", got)
	}
	tokens, err := tok.tokenize(got, maxPhonemeLength)
	if err != nil || len(tokens) != len([]rune(got)) {
		t.Errorf("tokenize = %v, %v", tokens, err)
	}
	if _, err := tok.tokenize(strings.Repeat("a", maxPhonemeLength+1), maxPhonemeLength); !errors.Is(err, errTooLong) {
		t.Errorf("tokenize past the limit: err = %v", err)
	}
	// Mandarin is phonemized with espeak-ng's own name for it.
	if _, err := tok.phonemize("你好", "cmn"); err != nil {
		t.Errorf("phonemize cmn: %v", err)
	}
}
