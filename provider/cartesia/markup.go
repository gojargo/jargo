package cartesia

import (
	"strconv"
	"strings"
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
// Dana and Marian.
type Emotion string

// The emotions Cartesia recognizes.
const (
	// The primary emotions.
	EmotionNeutral Emotion = "neutral"
	EmotionAngry   Emotion = "angry"
	EmotionExcited Emotion = "excited"
	EmotionContent Emotion = "content"
	EmotionSad     Emotion = "sad"
	EmotionScared  Emotion = "scared"
	// The rest of the set.
	EmotionHappy         Emotion = "happy"
	EmotionEnthusiastic  Emotion = "enthusiastic"
	EmotionElated        Emotion = "elated"
	EmotionEuphoric      Emotion = "euphoric"
	EmotionTriumphant    Emotion = "triumphant"
	EmotionAmazed        Emotion = "amazed"
	EmotionSurprised     Emotion = "surprised"
	EmotionFlirtatious   Emotion = "flirtatious"
	EmotionJokingComedic Emotion = "joking/comedic"
	EmotionCurious       Emotion = "curious"
	EmotionPeaceful      Emotion = "peaceful"
	EmotionSerene        Emotion = "serene"
	EmotionCalm          Emotion = "calm"
	EmotionGrateful      Emotion = "grateful"
	EmotionAffectionate  Emotion = "affectionate"
	EmotionTrust         Emotion = "trust"
	EmotionSympathetic   Emotion = "sympathetic"
	EmotionAnticipation  Emotion = "anticipation"
	EmotionMysterious    Emotion = "mysterious"
	EmotionMad           Emotion = "mad"
	EmotionOutraged      Emotion = "outraged"
	EmotionFrustrated    Emotion = "frustrated"
	EmotionAgitated      Emotion = "agitated"
	EmotionThreatened    Emotion = "threatened"
	EmotionDisgusted     Emotion = "disgusted"
	EmotionContempt      Emotion = "contempt"
	EmotionEnvious       Emotion = "envious"
	EmotionSarcastic     Emotion = "sarcastic"
	EmotionIronic        Emotion = "ironic"
	EmotionDejected      Emotion = "dejected"
	EmotionMelancholic   Emotion = "melancholic"
	EmotionDisappointed  Emotion = "disappointed"
	EmotionHurt          Emotion = "hurt"
	EmotionGuilty        Emotion = "guilty"
	EmotionBored         Emotion = "bored"
	EmotionTired         Emotion = "tired"
	EmotionRejected      Emotion = "rejected"
	EmotionNostalgic     Emotion = "nostalgic"
	EmotionWistful       Emotion = "wistful"
	EmotionApologetic    Emotion = "apologetic"
	EmotionHesitant      Emotion = "hesitant"
	EmotionInsecure      Emotion = "insecure"
	EmotionConfused      Emotion = "confused"
	EmotionResigned      Emotion = "resigned"
	EmotionAnxious       Emotion = "anxious"
	EmotionPanicked      Emotion = "panicked"
	EmotionAlarmed       Emotion = "alarmed"
	EmotionProud         Emotion = "proud"
	EmotionConfident     Emotion = "confident"
	EmotionDistant       Emotion = "distant"
	EmotionSkeptical     Emotion = "skeptical"
	EmotionContemplative Emotion = "contemplative"
	EmotionDetermined    Emotion = "determined"
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
