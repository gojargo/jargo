// This file holds Cartesia's expression adapter, which formats expression
// tags as Cartesia's inline markup for an expressive conversation (alpha).

package cartesia

import (
	"slices"
	"strconv"

	"github.com/gojargo/jargo/expressive"
)

// Names the vocabulary uses more than once.
const (
	laughter = "laughter"
)

// Emotion is an emotion Cartesia recognizes, which guides the emotional
// tone of the speech. The recommended voices for it are Leo, Jace, Kyle, Gavin,
// Maya, Tessa, Dana and Marian.
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

// Speeds is the speed each speed delivery asks for. Cartesia takes
// speed as guidance, so the steps are large.
func Speeds() map[string]float64 {
	return map[string]float64{expressive.SlowDelivery: 0.7, expressive.FastDelivery: 1.3}
}

// emotions is every emotion Cartesia recognizes, in order.
func emotions() []Emotion {
	return []Emotion{
		EmotionNeutral,
		EmotionAngry,
		EmotionExcited,
		EmotionContent,
		EmotionSad,
		EmotionScared,
		EmotionHappy,
		EmotionEnthusiastic,
		EmotionElated,
		EmotionEuphoric,
		EmotionTriumphant,
		EmotionAmazed,
		EmotionSurprised,
		EmotionFlirtatious,
		EmotionJokingComedic,
		EmotionCurious,
		EmotionPeaceful,
		EmotionSerene,
		EmotionCalm,
		EmotionGrateful,
		EmotionAffectionate,
		EmotionTrust,
		EmotionSympathetic,
		EmotionAnticipation,
		EmotionMysterious,
		EmotionMad,
		EmotionOutraged,
		EmotionFrustrated,
		EmotionAgitated,
		EmotionThreatened,
		EmotionDisgusted,
		EmotionContempt,
		EmotionEnvious,
		EmotionSarcastic,
		EmotionIronic,
		EmotionDejected,
		EmotionMelancholic,
		EmotionDisappointed,
		EmotionHurt,
		EmotionGuilty,
		EmotionBored,
		EmotionTired,
		EmotionRejected,
		EmotionNostalgic,
		EmotionWistful,
		EmotionApologetic,
		EmotionHesitant,
		EmotionInsecure,
		EmotionConfused,
		EmotionResigned,
		EmotionAnxious,
		EmotionPanicked,
		EmotionAlarmed,
		EmotionProud,
		EmotionConfident,
		EmotionDistant,
		EmotionSkeptical,
		EmotionContemplative,
		EmotionDetermined,
	}
}

// ExpressionAdapter formats expression tags as Cartesia's inline
// markup.
//
// Its deliveries are Cartesia's emotions, formatted as <emotion> tags; Cartesia
// supports them in English only. Cartesia changes emotion reliably only between
// generations, so a delivery can change only between sentences, and a sentence
// written in several emotions is spoken in the first. Slow and fast set each
// request's speed (see [Speeds]). The one sound is [laughter], and
// pauses are <break> tags. Cartesia has no emphasis.
type ExpressionAdapter struct {
	expressive.BaseExpressionAdapter
}

// Capabilities implements expressive.ExpressionAdapter: Cartesia's emotions,
// laughter and pauses.
func (ExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	var delivery []expressive.ExpressionTerm
	for _, e := range emotions() {
		if e != EmotionNeutral && e != EmotionJokingComedic {
			delivery = append(delivery, expressive.ExpressionTerm{Name: string(e)})
		}
	}
	return &expressive.ExpressionCapabilities{
		Sounds:        []expressive.ExpressionTerm{{Name: laughter}},
		Delivery:      append(delivery, expressive.SpeedDelivery()...),
		PauseRangeMS:  &[2]int{100, 5000},
		DeliveryScope: expressive.DeliveryScopeSentence,
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliveryCalm:         {"calm", "peaceful", "serene"},
			expressive.DeliveryWarm:         {"content", "happy"},
			expressive.DeliveryExcited:      {"excited", "enthusiastic"},
			expressive.DeliverySad:          {"sad", "disappointed"},
			expressive.SoundLaugh:           {laughter},
			expressive.SoundChuckle:         {laughter},
			expressive.DeliveryTrusting:     {"trust"},
			expressive.DeliveryAnticipating: {"anticipation"},
			expressive.DeliveryContemptuous: {"contempt"},
		},
	}
}

// FormatText implements expressive.ExpressionAdapter: it prefixes an emotion
// tag, set apart from the words, where the emotion changes.
func (ExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
	emotion, has := emotionOf(delivery)
	previous, hadPrevious := emotionOf(previousDelivery)
	if emotion == previous && has == hadPrevious {
		return text
	}
	if !has {
		emotion = string(EmotionNeutral)
	}
	return `<emotion value="` + emotion + `"/> ` + text
}

// RequestSettings implements expressive.ExpressionAdapter: the speed a request
// in this delivery asks for, merged into its generation_config.
func (ExpressionAdapter) RequestSettings(delivery []string) map[string]any {
	speeds := Speeds()
	for _, name := range delivery {
		if speed, ok := speeds[name]; ok {
			return map[string]any{"speed": speed}
		}
	}
	return nil
}

// FormatSound implements expressive.ExpressionAdapter: laughter, set apart so
// Cartesia times it as a word of its own.
func (ExpressionAdapter) FormatSound(name string) string { return " [" + name + "] " }

// FormatPause implements expressive.ExpressionAdapter: a break, set apart from
// the words.
func (ExpressionAdapter) FormatPause(durationMS int) string {
	return ` <break time="` + strconv.Itoa(durationMS) + `ms"/> `
}

// emotionOf is the first name in the delivery that is an emotion, not a speed,
// and false when there is none.
func emotionOf(delivery []string) (string, bool) {
	speeds := Speeds()
	i := slices.IndexFunc(delivery, func(name string) bool { _, speed := speeds[name]; return !speed })
	if i < 0 {
		return "", false
	}
	return delivery[i], true
}

var _ expressive.ExpressionAdapter = ExpressionAdapter{}
