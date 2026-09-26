package inworld

import (
	"strings"

	"github.com/gojargo/jargo/utils/text"
)

// FormatPronunciation renders a pronunciation as IPA between slashes, as Inworld
// reads it, e.g. /kriːt/.
//
// Inworld speaks the IPA wherever it appears inline, one word per pair of
// slashes, so a pronunciation covering several words cannot be written. Only
// English IPA symbols are read. The word itself is unused: the IPA replaces it.
// It reports false for an empty pronunciation or one spanning several words.
func FormatPronunciation(_, ipa string) (string, bool) {
	ipa = text.NormalizeIPA(ipa)
	if ipa == "" || strings.Contains(ipa, " ") {
		return "", false
	}
	return "/" + ipa + "/", true
}

// FormatPronunciation implements tts.PronunciationFormatter.
func (s *synthesizer) FormatPronunciation(word, ipa string) (string, bool) {
	return FormatPronunciation(word, ipa)
}
