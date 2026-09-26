package elevenlabs

import (
	"strings"

	"github.com/gojargo/jargo/utils/text"
)

// PhonemeModels are the models that read SSML <phoneme> tags. ElevenLabs' docs
// list only eleven_flash_v2; eleven_turbo_v2 reads them too.
//
//nolint:gochecknoglobals // the set of models, fixed
var PhonemeModels = map[string]struct{}{"eleven_flash_v2": {}, "eleven_turbo_v2": {}}

// FormatPronunciation renders a pronunciation as ElevenLabs SSML <phoneme>
// tags, e.g. <phoneme alphabet="ipa" ph="mɛtˈfɔɹmɪn">Metformin</phoneme>.
//
// A phoneme tag holds a single word, so a phrase gets one tag per word and its
// IPA must have as many words; it reports false when they do not line up. Only
// the models in PhonemeModels read phoneme tags.
func FormatPronunciation(word, ipa string) (string, bool) {
	words := strings.Fields(word)
	spoken := strings.Fields(text.NormalizeIPA(ipa))
	if len(words) == 0 || len(spoken) != len(words) {
		return "", false
	}
	tags := make([]string, len(words))
	for i, w := range words {
		tags[i] = `<phoneme alphabet="ipa" ph="` + attrEscaper.Replace(spoken[i]) + `">` +
			textEscaper.Replace(w) + "</phoneme>"
	}
	return strings.Join(tags, " "), true
}

// attrEscaper escapes a value written inside a quoted attribute, and textEscaper
// text written between tags, the way HTML escaping does both.
//
//nolint:gochecknoglobals // stateless escapers, built once
var (
	attrEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&#x27;")
	textEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;")
)

// supportsPhonemes reports whether model reads phoneme tags.
func supportsPhonemes(model string) bool {
	_, ok := PhonemeModels[model]
	return ok
}

// FormatPronunciation implements tts.PronunciationFormatter.
func (s *synthesizer) FormatPronunciation(word, ipa string) (string, bool) {
	return FormatPronunciation(word, ipa)
}

// SupportsPronunciations implements tts.PronunciationSupporter: phoneme tags are
// read only with a model in PhonemeModels.
func (s *synthesizer) SupportsPronunciations() bool { return supportsPhonemes(s.cfg.Model) }

// FormatPronunciation implements tts.PronunciationFormatter.
func (s *realtimeSynthesizer) FormatPronunciation(word, ipa string) (string, bool) {
	return FormatPronunciation(word, ipa)
}

// SupportsPronunciations implements tts.PronunciationSupporter: phoneme tags are
// read only with a model in PhonemeModels, and with EnableSSMLParsing on.
func (s *realtimeSynthesizer) SupportsPronunciations() bool {
	return supportsPhonemes(s.cfg.Model) && s.cfg.EnableSSMLParsing != nil && *s.cfg.EnableSSMLParsing
}
