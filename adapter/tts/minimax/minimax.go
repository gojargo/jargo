// Package minimax is MiniMax's expression adapter, which performs expression
// tags with MiniMax speech-2.8 models for an expressive conversation (alpha).
package minimax

import (
	"strconv"
	"strings"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive"
)

// Names the vocabulary uses more than once.
const (
	happy = "happy"
)

// MiniMaxSpeeds is the speed each speed delivery asks for.
func MiniMaxSpeeds() map[string]float64 {
	return map[string]float64{expressive.SlowDelivery: 0.75, expressive.FastDelivery: 1.25}
}

// MiniMaxExpressionAdapter performs expression tags with MiniMax speech-2.8
// models.
//
// Its deliveries are MiniMax's emotions and speeds, sent with each request, so
// a delivery can change only between sentences, and a sentence written in
// several emotions is spoken in the first. Sounds are its interjections, such
// as (laughs), and pauses are <#s#> markers. Its vocabulary is fixed: an
// interjection outside it is read aloud.
type MiniMaxExpressionAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

// Capabilities implements ttsadapter.ExpressionAdapter: MiniMax's emotions,
// interjections and pauses.
func (MiniMaxExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	var sounds, delivery []expressive.ExpressionTerm
	for name := range strings.FieldsSeq("laughs chuckle coughs clear-throat groans breath pant inhale exhale " +
		"gasps sniffs sighs snorts burps lip-smacking humming hissing emm sneezes") {
		sounds = append(sounds, expressive.ExpressionTerm{Name: name})
	}
	for _, name := range []string{happy, "sad", "angry", "fearful", "disgusted", "surprised", "calm"} {
		delivery = append(delivery, expressive.ExpressionTerm{Name: name})
	}
	return &expressive.ExpressionCapabilities{
		Sounds:        sounds,
		Delivery:      append(delivery, expressive.SpeedDelivery()...),
		PauseRangeMS:  &[2]int{10, 99990},
		DeliveryScope: expressive.DeliveryScopeSentence,
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliveryWarm:     {happy},
			expressive.DeliveryExcited:  {happy},
			expressive.SoundLaugh:       {"laughs"},
			expressive.SoundSigh:        {"sighs"},
			expressive.SoundBreath:      {"breath", "inhale"},
			expressive.SoundLightBreath: {"breath"},
			expressive.SoundDeepBreath:  {"inhale"},
			expressive.SoundCough:       {"coughs"},
			expressive.SoundClearThroat: {"clear-throat"},
			expressive.SoundGroan:       {"groans"},
			expressive.SoundGasp:        {"gasps"},
			expressive.SoundSniff:       {"sniffs"},
			expressive.SoundSnort:       {"snorts"},
			expressive.SoundBurp:        {"burps"},
			expressive.SoundLipSmack:    {"lip-smacking"},
			expressive.SoundHum:         {"humming"},
			expressive.SoundHiss:        {"hissing"},
			expressive.SoundUm:          {"emm"},
			expressive.SoundSneeze:      {"sneezes"},
		},
	}
}

// RequestSettings implements ttsadapter.ExpressionAdapter: the emotion and
// speed a request in this delivery asks for, merged into its voice_setting.
func (MiniMaxExpressionAdapter) RequestSettings(delivery []string) map[string]any {
	speeds := MiniMaxSpeeds()
	settings := map[string]any{}
	for _, name := range delivery {
		if _, speed := speeds[name]; !speed {
			settings["emotion"] = name
			break
		}
	}
	for _, name := range delivery {
		if speed, ok := speeds[name]; ok {
			settings["speed"] = speed
			break
		}
	}
	return settings
}

// FormatSound implements ttsadapter.ExpressionAdapter: an interjection.
func (MiniMaxExpressionAdapter) FormatSound(name string) string { return "(" + name + ")" }

// FormatPause implements ttsadapter.ExpressionAdapter: a pause marker, in
// seconds.
func (MiniMaxExpressionAdapter) FormatPause(durationMS int) string {
	return "<#" + strconv.FormatFloat(float64(durationMS)/1000, 'g', 6, 64) + "#>"
}

var _ ttsadapter.ExpressionAdapter = MiniMaxExpressionAdapter{}
