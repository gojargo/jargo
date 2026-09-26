package deepgram

import (
	"bytes"
	"encoding/json"
	"strings"
	"unicode/utf8"

	"github.com/gojargo/jargo/utils/text"
)

// Aura-2 rejects IPA that is far longer than the word it replaces, or longer
// than 128 characters. Short words get a floor instead of the ratio.
const (
	maxIPALength    = 128
	maxIPAWordRatio = 10
	minIPALength    = 15
)

// FormatPronunciation renders a pronunciation as a Deepgram inline
// pronunciation object, e.g. \{"word": "dupilumab", "pronounce": "duːpˈɪljuːmæb"\}.
//
// Aura-2 reads an escaped JSON object anywhere in the text and speaks its
// pronounce value in place of its word. The word stays in the text and is what
// Deepgram bills for; the IPA is not billed. English and Spanish only. Stress
// marks go directly before the vowel they stress; Aura-2 warns about any other
// placement and falls back to a best-effort pronunciation. It reports false for
// an empty pronunciation, or IPA longer than Deepgram accepts for the word.
//
// Flux has no pronunciation markup, so the Flux service takes none; see
// https://developers.deepgram.com/docs/tts-voice-controls for which controls
// each model supports.
func FormatPronunciation(word, ipa string) (string, bool) {
	var parts []string
	for w := range strings.FieldsSeq(text.NormalizeIPA(ipa)) {
		parts = append(parts, strings.Join(text.StressBeforeVowels(text.IPAPhones(w)), ""))
	}
	ipa = strings.Join(parts, " ")
	n := utf8.RuneCountInString(ipa)
	if n == 0 || n > maxIPALength || n > max(maxIPAWordRatio*utf8.RuneCountInString(word), minIPALength) {
		return "", false
	}
	return `\{"word": ` + jsonString(word) + `, "pronounce": ` + jsonString(ipa) + `\}`, true
}

// jsonString writes s as a JSON string, leaving non-ASCII characters and HTML
// as they are.
func jsonString(s string) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s); err != nil {
		// A string always encodes; this is unreachable.
		return `""`
	}
	return strings.TrimRight(buf.String(), "\n")
}

// FormatPronunciation implements tts.PronunciationFormatter.
func (s *synthesizer) FormatPronunciation(word, ipa string) (string, bool) {
	return FormatPronunciation(word, ipa)
}
