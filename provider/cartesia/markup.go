package cartesia

import (
	"strconv"
	"strings"

	ttscartesia "github.com/gojargo/jargo/adapter/tts/cartesia"
	"github.com/gojargo/jargo/utils/text"
)

// The markup Cartesia reads inside the text it is given. The helpers below build
// each tag, so the spelling of one lives in a single place.
//
// The preferred way to use them is from a text transform on the TTS service, so
// the markup reaches the synthesizer and nothing else: what the model wrote and
// what the conversation records stay free of it. Text handed over with markup in
// it works too, and the service holds off on a sentence boundary inside a spell
// tag either way.

// Emotion guides the emotional tone of the speech. Cartesia recognizes the set
// below; the recommended voices for it are Leo, Jace, Kyle, Gavin, Maya, Tessa,
// Dana and Marian. It is the emotion its expression adapter performs in an
// expressive conversation.
type Emotion = ttscartesia.CartesiaEmotion

// The emotions Cartesia recognizes.
const (
	EmotionNeutral       = ttscartesia.EmotionNeutral
	EmotionAngry         = ttscartesia.EmotionAngry
	EmotionExcited       = ttscartesia.EmotionExcited
	EmotionContent       = ttscartesia.EmotionContent
	EmotionSad           = ttscartesia.EmotionSad
	EmotionScared        = ttscartesia.EmotionScared
	EmotionHappy         = ttscartesia.EmotionHappy
	EmotionEnthusiastic  = ttscartesia.EmotionEnthusiastic
	EmotionElated        = ttscartesia.EmotionElated
	EmotionEuphoric      = ttscartesia.EmotionEuphoric
	EmotionTriumphant    = ttscartesia.EmotionTriumphant
	EmotionAmazed        = ttscartesia.EmotionAmazed
	EmotionSurprised     = ttscartesia.EmotionSurprised
	EmotionFlirtatious   = ttscartesia.EmotionFlirtatious
	EmotionJokingComedic = ttscartesia.EmotionJokingComedic
	EmotionCurious       = ttscartesia.EmotionCurious
	EmotionPeaceful      = ttscartesia.EmotionPeaceful
	EmotionSerene        = ttscartesia.EmotionSerene
	EmotionCalm          = ttscartesia.EmotionCalm
	EmotionGrateful      = ttscartesia.EmotionGrateful
	EmotionAffectionate  = ttscartesia.EmotionAffectionate
	EmotionTrust         = ttscartesia.EmotionTrust
	EmotionSympathetic   = ttscartesia.EmotionSympathetic
	EmotionAnticipation  = ttscartesia.EmotionAnticipation
	EmotionMysterious    = ttscartesia.EmotionMysterious
	EmotionMad           = ttscartesia.EmotionMad
	EmotionOutraged      = ttscartesia.EmotionOutraged
	EmotionFrustrated    = ttscartesia.EmotionFrustrated
	EmotionAgitated      = ttscartesia.EmotionAgitated
	EmotionThreatened    = ttscartesia.EmotionThreatened
	EmotionDisgusted     = ttscartesia.EmotionDisgusted
	EmotionContempt      = ttscartesia.EmotionContempt
	EmotionEnvious       = ttscartesia.EmotionEnvious
	EmotionSarcastic     = ttscartesia.EmotionSarcastic
	EmotionIronic        = ttscartesia.EmotionIronic
	EmotionDejected      = ttscartesia.EmotionDejected
	EmotionMelancholic   = ttscartesia.EmotionMelancholic
	EmotionDisappointed  = ttscartesia.EmotionDisappointed
	EmotionHurt          = ttscartesia.EmotionHurt
	EmotionGuilty        = ttscartesia.EmotionGuilty
	EmotionBored         = ttscartesia.EmotionBored
	EmotionTired         = ttscartesia.EmotionTired
	EmotionRejected      = ttscartesia.EmotionRejected
	EmotionNostalgic     = ttscartesia.EmotionNostalgic
	EmotionWistful       = ttscartesia.EmotionWistful
	EmotionApologetic    = ttscartesia.EmotionApologetic
	EmotionHesitant      = ttscartesia.EmotionHesitant
	EmotionInsecure      = ttscartesia.EmotionInsecure
	EmotionConfused      = ttscartesia.EmotionConfused
	EmotionResigned      = ttscartesia.EmotionResigned
	EmotionAnxious       = ttscartesia.EmotionAnxious
	EmotionPanicked      = ttscartesia.EmotionPanicked
	EmotionAlarmed       = ttscartesia.EmotionAlarmed
	EmotionProud         = ttscartesia.EmotionProud
	EmotionConfident     = ttscartesia.EmotionConfident
	EmotionDistant       = ttscartesia.EmotionDistant
	EmotionSkeptical     = ttscartesia.EmotionSkeptical
	EmotionContemplative = ttscartesia.EmotionContemplative
	EmotionDetermined    = ttscartesia.EmotionDetermined
)

// Spell wraps text so Cartesia reads it out character by character, which is how
// a reference number or a code is said aloud.
func Spell(text string) string { return "<spell>" + text + "</spell>" }

// EmotionTag shifts the emotional tone from where it appears onward.
func EmotionTag(e Emotion) string { return `<emotion value="` + string(e) + `" />` }

// PauseTag inserts a silence of the given length.
func PauseTag(seconds float64) string {
	return `<break time="` + formatTagNumber(seconds) + `s" />`
}

// VolumeTag multiplies the volume from where it appears onward (0.5 to 2.0).
func VolumeTag(volume float64) string {
	return `<volume ratio="` + formatTagNumber(volume) + `" />`
}

// SpeedTag multiplies the speaking rate from where it appears onward (0.6 to 1.5).
func SpeedTag(speed float64) string {
	return `<speed ratio="` + formatTagNumber(speed) + `" />`
}

// formatTagNumber writes a number the way the markup spells one, keeping the
// decimal point on a whole value so "1" goes out as "1.0".
func formatTagNumber(v float64) string {
	s := strconv.FormatFloat(v, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}

// FormatPronunciation renders a pronunciation as Cartesia inline phonemes.
//
// Cartesia reads <<…>> blocks of |-separated IPA phones, with stress marks
// directly before the vowel they stress. Each word of the IPA becomes its own
// block, e.g. <<m|ɛ|t|f|ˈ|ɔ|ɹ|m|ɪ|n>>. The word itself is unused: the block
// replaces it. It reports false for an empty pronunciation.
func FormatPronunciation(_, ipa string) (string, bool) {
	var blocks []string
	for w := range strings.FieldsSeq(text.NormalizeIPA(ipa)) {
		blocks = append(blocks, "<<"+strings.Join(text.StressBeforeVowels(text.IPAPhones(w)), "|")+">>")
	}
	if len(blocks) == 0 {
		return "", false
	}
	return strings.Join(blocks, " "), true
}

// FormatPronunciation implements tts.PronunciationFormatter.
func (s *synthesizer) FormatPronunciation(word, ipa string) (string, bool) {
	return FormatPronunciation(word, ipa)
}

// FormatPronunciation implements tts.PronunciationFormatter.
func (s *httpSynthesizer) FormatPronunciation(word, ipa string) (string, bool) {
	return FormatPronunciation(word, ipa)
}
