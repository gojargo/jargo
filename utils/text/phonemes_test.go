package text_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

func TestNormalizeIPA(t *testing.T) {
	cases := map[string]struct{ in, want string }{
		"strips slashes and syllable breaks": {" /ˈtʃɪ.kən/ ", "ˈtʃɪkən"},
		"strips brackets":                    {"[ˈtʃɪkən]", "ˈtʃɪkən"},
		"unties tʃ written as a ligature":    {"ˈʧɪkən", "ˈtʃɪkən"},
		"unties tʃ written tied":             {"ˈt͡ʃɪkən", "ˈtʃɪkən"},
		"ties other affricates":              {"ˈpiʦa", "ˈpit͡sa"},
		"moves the tie bar above":            {"t͜s", "t͡s"},
		"replaces ASCII look-alikes":         {"'ælɛgrə", "ˈælɛɡrə"},
		"replaces the length colon":          {"bi:", "biː"},
		// r and ɹ, ɾ and t are different sounds in some languages; notation only.
		"keeps sounds":         {"ˈbɛɾə ˈrɑk", "ˈbɛɾə ˈrɑk"},
		"collapses whitespace": {"ˈɡlaɪ   mɛt", "ˈɡlaɪ mɛt"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := text.NormalizeIPA(tc.in); got != tc.want {
				t.Fatalf("NormalizeIPA(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestIPAPhones(t *testing.T) {
	cases := map[string]struct {
		in   string
		want []string
	}{
		"groups multi-character phones":      {"ˈtʃaɪnə", []string{"ˈ", "tʃ", "aɪ", "n", "ə"}},
		"groups a ligature the same":         {"ˈʧaɪnə", []string{"ˈ", "tʃ", "aɪ", "n", "ə"}},
		"leaves an untied cluster apart":     {"kæts", []string{"k", "æ", "t", "s"}},
		"groups a tied cluster":              {"ˈpit͡sa", []string{"ˈ", "p", "i", "ts", "a"}},
		"keeps length and diacritics":        {"biːn̩", []string{"b", "iː", "n̩"}},
		"returns stress where it is written": {"mɛtˈfɔɹmɪn", []string{"m", "ɛ", "t", "ˈ", "f", "ɔ", "ɹ", "m", "ɪ", "n"}},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := text.IPAPhones(tc.in); !slices.Equal(got, tc.want) {
				t.Fatalf("IPAPhones(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestStressBeforeVowels(t *testing.T) {
	got := strings.Join(text.StressBeforeVowels(text.IPAPhones("ˌsɪpɹoʊˈflɑksəsɪn")), "")
	if got != "sˌɪpɹoʊflˈɑksəsɪn" {
		t.Fatalf("got %q, want sˌɪpɹoʊflˈɑksəsɪn", got)
	}
}

// slashes formats a pronunciation as IPA between slashes, a phrase included.
func slashes(_, ipa string) (string, bool) {
	ipa = text.NormalizeIPA(ipa)
	return "/" + ipa + "/", ipa != ""
}

// oneWord formats a pronunciation as IPA between slashes, a single word only.
func oneWord(_, ipa string) (string, bool) {
	ipa = text.NormalizeIPA(ipa)
	return "/" + ipa + "/", ipa != "" && !strings.Contains(ipa, " ")
}

// tag keeps the word, the way an SSML tag does.
func tag(word, ipa string) (string, bool) { return "<" + ipa + ">" + word + "</>", true }

func TestPronunciationTransformReplacesWholeWordsCaseInsensitively(t *testing.T) {
	tr := text.NewPronunciationTransform(map[string]string{"metformin": "mɛtˈfɔɹmɪn"}, tag, "")
	got := tr.Apply("Take METFORMIN twice. Metformin, then water.")
	want := "Take <mɛtˈfɔɹmɪn>METFORMIN</> twice. <mɛtˈfɔɹmɪn>Metformin</>, then water."
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestPronunciationTransformLeavesLongerAndHyphenatedWords(t *testing.T) {
	tr := text.NewPronunciationTransform(map[string]string{"Metformin": "mɛtˈfɔɹmɪn"}, slashes, "")
	in := "Metformins and glyburide-metformin and metformin-ER"
	if got := tr.Apply(in); got != in {
		t.Fatalf("got %q, want the text unchanged", got)
	}
}

func TestPronunciationTransformLongestFirst(t *testing.T) {
	tr := text.NewPronunciationTransform(map[string]string{
		"metformin": "mɛtˈfɔɹmɪn", "glyburide metformin": "ˈɡlaɪbjəˌraɪd mɛtˈfɔɹmɪn",
	}, slashes, "")
	got := tr.Apply("Glyburide metformin, or metformin")
	if want := "/ˈɡlaɪbjəˌraɪd mɛtˈfɔɹmɪn/, or /mɛtˈfɔɹmɪn/"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestPronunciationTransformSpeaksUnusableWordsAsWritten(t *testing.T) {
	tr := text.NewPronunciationTransform(map[string]string{
		"Xarelto": "zəˈɹɛltoʊ", "Glyburide metformin": "ˈɡlaɪ mɛt",
	}, oneWord, "")
	got := tr.Apply("Glyburide metformin or Xarelto")
	if want := "Glyburide metformin or /zəˈɹɛltoʊ/"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestPronunciationTransformWithNothingUsable(t *testing.T) {
	none := func(string, string) (string, bool) { return "", false }
	tr := text.NewPronunciationTransform(map[string]string{"Metformin": "mɛtˈfɔɹmɪn"}, none, "")
	if got := tr.Apply("Metformin"); got != "Metformin" {
		t.Fatalf("got %q, want the text unchanged", got)
	}
}
