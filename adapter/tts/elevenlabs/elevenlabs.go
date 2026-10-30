// Package elevenlabs is ElevenLabs' expression adapter, which formats
// expression tags as Eleven v3 and v4 audio tags for an expressive conversation
// (alpha).
package elevenlabs

import (
	"slices"
	"strings"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive"
)

// Names the vocabulary uses more than once.
const (
	soundEffect = "sound effect"
	warmly      = "warmly"
	sheepishly  = "sheepishly"
)

// speedTags is the audio tag each speed delivery is written as.
func speedTags() map[string]string {
	return map[string]string{expressive.SlowDelivery: "slowly", expressive.FastDelivery: "quickly"}
}

// audioTagSounds and audioTagDelivery are the audio tags in ElevenLabs'
// prompting guide for Eleven v3 and v4, and a short breath, a light one beside
// the guide's deep one, sorted into sounds made at one point and deliveries
// that last. Their guide doesn't draw that line. Multi-word deliveries, such as
// accents, aren't listed; written in a tag, they're passed on as free text.
func audioTagSounds() []expressive.ExpressionTerm {
	return []expressive.ExpressionTerm{
		{Name: "laughs"},
		{Name: "laughs harder"},
		{Name: "starts laughing"},
		{Name: "laughing"},
		{Name: "wheezing", Description: "a wheezing laugh"},
		{Name: "giggles"},
		{Name: "giggling"},
		{Name: "chuckles"},
		{Name: "snorts"},
		{Name: "sighs"},
		{Name: "sighing"},
		{Name: "exhales"},
		{Name: "exhales sharply"},
		{Name: "inhales deeply"},
		{Name: "short breath"},
		{Name: "clears throat"},
		{Name: "gulps"},
		{Name: "swallows"},
		{Name: "woo", Description: "an excited whoop"},
		{Name: "applause", Description: soundEffect},
		{Name: "clapping", Description: soundEffect},
		{Name: "gunshot", Description: soundEffect},
		{Name: "explosion", Description: soundEffect},
		{Name: "fart", Description: soundEffect},
	}
}

func audioTagDelivery() []expressive.ExpressionTerm {
	return []expressive.ExpressionTerm{
		{Name: "happy"},
		{Name: "excited"},
		{Name: "delighted"},
		{Name: "amazed"},
		{Name: "impressed"},
		{Name: "surprised"},
		{Name: "curious"},
		{Name: "questioning"},
		{Name: "thoughtful"},
		{Name: warmly},
		{Name: "sympathetic"},
		{Name: "reassuring"},
		{Name: "professional"},
		{Name: "cautiously"},
		{Name: "nervously"},
		{Name: sheepishly},
		{Name: "sad"},
		{Name: "crying", Description: "speaking while crying"},
		{Name: "desperately"},
		{Name: "panicking"},
		{Name: "frustrated"},
		{Name: "annoyed"},
		{Name: "angry"},
		{Name: "appalled"},
		{Name: "dismissive"},
		{Name: "sarcastic"},
		{Name: "deadpan"},
		{Name: "mischievously"},
		{Name: "dramatically"},
		{Name: "cute"},
		{Name: "whispers", Description: "speaking in a whisper"},
		{Name: "muttering", Description: "speaking under one's breath"},
		{Name: "singing"},
	}
}

// ElevenLabsAudioTagExpressionAdapter formats expression tags as Eleven v3 and
// v4 audio tags.
//
// Delivery and sounds are both audio tags, such as [excited] or [laughs],
// written in free text: the terms listed are the ones in ElevenLabs' prompting
// guide, and others described in a few words work too. Emphasis is written in
// capitals. A delivery tag is written where the delivery changes; Eleven v3 and
// v4 have no tag for the voice's baseline, so a return to it writes none. Slow
// and fast are written as [slowly] and [quickly]. They don't support <break>,
// so a pause is written as a short or long pause cue, which is approximate.
type ElevenLabsAudioTagExpressionAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

// Capabilities implements ttsadapter.ExpressionAdapter: free-text audio tags,
// emphasis and approximate pauses.
func (ElevenLabsAudioTagExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:                 audioTagSounds(),
		Delivery:               append(audioTagDelivery(), expressive.SpeedDelivery()...),
		AllowsFreeTextSounds:   true,
		AllowsFreeTextDelivery: true,
		PauseRangeMS:           &[2]int{100, 3000},
		Emphasis:               true,
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliveryCalm:       {"calm", "reassuring"},
			expressive.DeliveryApologetic: {"apologetic", sheepishly},
			expressive.DeliveryWarm:       {warmly},
			expressive.DeliveryExcited:    {"excited", "delighted"},
			expressive.DeliveryGrateful:   {"grateful", warmly},
			expressive.DeliveryConfident:  {"confident", "professional"},
			expressive.SoundLaugh:         {"laughs", "laughing"},
			expressive.SoundChuckle:       {"chuckles"},
			expressive.SoundSigh:          {"sighs", "sighing"},
			expressive.SoundBreath:        {"quick breath"},
			// A policy that leaves out a kind's own name gets its next best.
			expressive.SoundLightBreath:    {"quick breath", "short breath"},
			expressive.SoundDeepBreath:     {"deep breath", "inhales deeply"},
			expressive.SoundExhale:         {"exhales softly", "exhales"},
			expressive.DeliveryCautious:    {"cautiously"},
			expressive.DeliveryNervous:     {"nervously"},
			expressive.DeliverySheepish:    {sheepishly},
			expressive.DeliveryDesperate:   {"desperately"},
			expressive.DeliveryPanicked:    {"panicking"},
			expressive.DeliveryMischievous: {"mischievously"},
			expressive.DeliveryDramatic:    {"dramatically"},
			expressive.DeliveryWhispering:  {"whispers"},
			expressive.SoundLaughHarder:    {"laughs harder"},
			expressive.SoundStartLaughing:  {"starts laughing"},
			expressive.SoundWheeze:         {"wheezing"},
			expressive.SoundGiggle:         {"giggles", "giggling"},
			expressive.SoundSnort:          {"snorts"},
			expressive.SoundSharpExhale:    {"exhales sharply"},
			expressive.SoundClearThroat:    {"clears throat"},
			expressive.SoundGulp:           {"gulps"},
			expressive.SoundSwallow:        {"swallows"},
		},
	}
}

// FormatText implements ttsadapter.ExpressionAdapter: it prefixes audio tags
// where the delivery changes, and capitalizes emphasis.
func (ElevenLabsAudioTagExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
	if slices.Contains(delivery, expressive.EmphasisDelivery) {
		text = strings.ToUpper(text)
	}
	tags := ttsadapter.DeliveryTags(delivery, speedTags())
	previous := ttsadapter.DeliveryTags(previousDelivery, speedTags())
	if len(tags) > 0 && !slices.Equal(tags, previous) {
		var b strings.Builder
		for _, tag := range tags {
			b.WriteString("[" + tag + "] ")
		}
		text = b.String() + text
	}
	return text
}

// FormatSound implements ttsadapter.ExpressionAdapter: an audio tag, set apart
// so its alignment times it as a word of its own.
func (ElevenLabsAudioTagExpressionAdapter) FormatSound(name string) string {
	return " [" + name + "] "
}

// FormatPause implements ttsadapter.ExpressionAdapter: a short pause under a
// second, otherwise a long one, set apart like a sound.
func (ElevenLabsAudioTagExpressionAdapter) FormatPause(durationMS int) string {
	if durationMS < 1000 {
		return " [short pause] "
	}
	return " [long pause] "
}

var _ ttsadapter.ExpressionAdapter = ElevenLabsAudioTagExpressionAdapter{}
