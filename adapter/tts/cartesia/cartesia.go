// Package cartesia is Cartesia's expression adapter, which formats expression
// tags as Cartesia's inline markup for an expressive conversation (alpha).
package cartesia

import (
	"slices"
	"strconv"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive"
)

// Names the vocabulary uses more than once.
const (
	laughter = "laughter"
)

// CartesiaEmotion is an emotion Cartesia recognizes, which guides the emotional
// tone of the speech. The recommended voices for it are Leo, Jace, Kyle, Gavin,
// Maya, Tessa, Dana and Marian.
type CartesiaEmotion string

// The emotions Cartesia recognizes.
const (
	// The primary emotions.
	EmotionNeutral CartesiaEmotion = "neutral"
	EmotionAngry   CartesiaEmotion = "angry"
	EmotionExcited CartesiaEmotion = "excited"
	EmotionContent CartesiaEmotion = "content"
	EmotionSad     CartesiaEmotion = "sad"
	EmotionScared  CartesiaEmotion = "scared"
	// The rest of the set.
	EmotionHappy         CartesiaEmotion = "happy"
	EmotionEnthusiastic  CartesiaEmotion = "enthusiastic"
	EmotionElated        CartesiaEmotion = "elated"
	EmotionEuphoric      CartesiaEmotion = "euphoric"
	EmotionTriumphant    CartesiaEmotion = "triumphant"
	EmotionAmazed        CartesiaEmotion = "amazed"
	EmotionSurprised     CartesiaEmotion = "surprised"
	EmotionFlirtatious   CartesiaEmotion = "flirtatious"
	EmotionJokingComedic CartesiaEmotion = "joking/comedic"
	EmotionCurious       CartesiaEmotion = "curious"
	EmotionPeaceful      CartesiaEmotion = "peaceful"
	EmotionSerene        CartesiaEmotion = "serene"
	EmotionCalm          CartesiaEmotion = "calm"
	EmotionGrateful      CartesiaEmotion = "grateful"
	EmotionAffectionate  CartesiaEmotion = "affectionate"
	EmotionTrust         CartesiaEmotion = "trust"
	EmotionSympathetic   CartesiaEmotion = "sympathetic"
	EmotionAnticipation  CartesiaEmotion = "anticipation"
	EmotionMysterious    CartesiaEmotion = "mysterious"
	EmotionMad           CartesiaEmotion = "mad"
	EmotionOutraged      CartesiaEmotion = "outraged"
	EmotionFrustrated    CartesiaEmotion = "frustrated"
	EmotionAgitated      CartesiaEmotion = "agitated"
	EmotionThreatened    CartesiaEmotion = "threatened"
	EmotionDisgusted     CartesiaEmotion = "disgusted"
	EmotionContempt      CartesiaEmotion = "contempt"
	EmotionEnvious       CartesiaEmotion = "envious"
	EmotionSarcastic     CartesiaEmotion = "sarcastic"
	EmotionIronic        CartesiaEmotion = "ironic"
	EmotionDejected      CartesiaEmotion = "dejected"
	EmotionMelancholic   CartesiaEmotion = "melancholic"
	EmotionDisappointed  CartesiaEmotion = "disappointed"
	EmotionHurt          CartesiaEmotion = "hurt"
	EmotionGuilty        CartesiaEmotion = "guilty"
	EmotionBored         CartesiaEmotion = "bored"
	EmotionTired         CartesiaEmotion = "tired"
	EmotionRejected      CartesiaEmotion = "rejected"
	EmotionNostalgic     CartesiaEmotion = "nostalgic"
	EmotionWistful       CartesiaEmotion = "wistful"
	EmotionApologetic    CartesiaEmotion = "apologetic"
	EmotionHesitant      CartesiaEmotion = "hesitant"
	EmotionInsecure      CartesiaEmotion = "insecure"
	EmotionConfused      CartesiaEmotion = "confused"
	EmotionResigned      CartesiaEmotion = "resigned"
	EmotionAnxious       CartesiaEmotion = "anxious"
	EmotionPanicked      CartesiaEmotion = "panicked"
	EmotionAlarmed       CartesiaEmotion = "alarmed"
	EmotionProud         CartesiaEmotion = "proud"
	EmotionConfident     CartesiaEmotion = "confident"
	EmotionDistant       CartesiaEmotion = "distant"
	EmotionSkeptical     CartesiaEmotion = "skeptical"
	EmotionContemplative CartesiaEmotion = "contemplative"
	EmotionDetermined    CartesiaEmotion = "determined"
)

// CartesiaSpeeds is the speed each speed delivery asks for. Cartesia takes
// speed as guidance, so the steps are large.
func CartesiaSpeeds() map[string]float64 {
	return map[string]float64{expressive.SlowDelivery: 0.7, expressive.FastDelivery: 1.3}
}

// emotions is every emotion Cartesia recognizes, in order.
func emotions() []CartesiaEmotion {
	return []CartesiaEmotion{
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

// CartesiaExpressionAdapter formats expression tags as Cartesia's inline
// markup.
//
// Its deliveries are Cartesia's emotions, formatted as <emotion> tags; Cartesia
// supports them in English only. Cartesia changes emotion reliably only between
// generations, so a delivery can change only between sentences, and a sentence
// written in several emotions is spoken in the first. Slow and fast set each
// request's speed (see [CartesiaSpeeds]). The one sound is [laughter], and
// pauses are <break> tags. Cartesia has no emphasis.
type CartesiaExpressionAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

// Capabilities implements ttsadapter.ExpressionAdapter: Cartesia's emotions,
// laughter and pauses.
func (CartesiaExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
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

// FormatText implements ttsadapter.ExpressionAdapter: it prefixes an emotion
// tag, set apart from the words, where the emotion changes.
func (CartesiaExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
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

// RequestSettings implements ttsadapter.ExpressionAdapter: the speed a request
// in this delivery asks for, merged into its generation_config.
func (CartesiaExpressionAdapter) RequestSettings(delivery []string) map[string]any {
	speeds := CartesiaSpeeds()
	for _, name := range delivery {
		if speed, ok := speeds[name]; ok {
			return map[string]any{"speed": speed}
		}
	}
	return nil
}

// FormatSound implements ttsadapter.ExpressionAdapter: laughter, set apart so
// Cartesia times it as a word of its own.
func (CartesiaExpressionAdapter) FormatSound(name string) string { return " [" + name + "] " }

// FormatPause implements ttsadapter.ExpressionAdapter: a break, set apart from
// the words.
func (CartesiaExpressionAdapter) FormatPause(durationMS int) string {
	return ` <break time="` + strconv.Itoa(durationMS) + `ms"/> `
}

// emotionOf is the first name in the delivery that is an emotion, not a speed,
// and false when there is none.
func emotionOf(delivery []string) (string, bool) {
	speeds := CartesiaSpeeds()
	i := slices.IndexFunc(delivery, func(name string) bool { _, speed := speeds[name]; return !speed })
	if i < 0 {
		return "", false
	}
	return delivery[i], true
}

var _ ttsadapter.ExpressionAdapter = CartesiaExpressionAdapter{}
