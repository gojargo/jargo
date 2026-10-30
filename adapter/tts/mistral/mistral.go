// Package mistral is Mistral's expression adapter, which formats expression
// tags as Voxtral 3 TTS tags for an expressive conversation (alpha).
package mistral

import (
	"strings"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive"
)

// normalPace is the pace a return from slow or fast writes.
const normalPace = "moderate"

func paces() map[string]string {
	return map[string]string{expressive.SlowDelivery: "slow", expressive.FastDelivery: "fast"}
}

func terms(names ...string) []expressive.ExpressionTerm {
	out := make([]expressive.ExpressionTerm, 0, len(names))
	for _, name := range names {
		out = append(out, expressive.ExpressionTerm{Name: name})
	}
	return out
}

// MistralExpressionAdapter formats expression tags as Voxtral 3 TTS tags.
//
// A delivery is an emotion tag, such as [emotion=warm], from Mistral's fixed
// list, and slow and fast are pace tags, [pace=slow] and [pace=fast]. A tag
// lasts until the next tag of its kind or the end of the request, so a sentence
// states its delivery from its start. Voxtral 3 has no emotion for the voice's
// baseline, so a return to it within a sentence writes none, while a return to
// the normal pace writes [pace=moderate]. A request speaks one emotion at a
// time, so a delivery of several is spoken in the first. Sounds are tags such
// as [laugh]. Voxtral 3 has no pauses or emphasis.
type MistralExpressionAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

// Capabilities implements ttsadapter.ExpressionAdapter: Voxtral 3's emotions,
// paces and sounds.
func (MistralExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	emotions := terms(strings.Fields("amused angry anxious apologetic appreciative assertive bored calm " +
		"confident confused curious determined disappointed disgusted excited frustrated grateful happy " +
		"hopeful playful proud reflective resigned sad sarcastic skeptical somber surprised tense " +
		"tired urgent warm wistful")...)
	return &expressive.ExpressionCapabilities{
		Sounds: terms("um", "uh", "laugh", "laugh hard", "chuckle", "breath", "sigh", "gasp", "sniff",
			"cough", "throat clearing", "yawn"),
		Delivery: append(emotions, expressive.SpeedDelivery()...),
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliverySympathetic:   {"warm"},
			expressive.DeliveryGrateful:      {"grateful", "appreciative"},
			expressive.DeliveryConfident:     {"confident", "assertive"},
			expressive.DeliveryThoughtful:    {"reflective"},
			expressive.DeliveryContemplative: {"reflective"},
			expressive.DeliveryNervous:       {"tense"},
			expressive.DeliveryWorried:       {"anxious"},
			expressive.SoundUm:               {"um", "uh"},
			expressive.SoundLaughHarder:      {"laugh hard"},
			expressive.SoundLightBreath:      {"breath"},
			expressive.SoundClearThroat:      {"throat clearing"},
		},
	}
}

// FormatText implements ttsadapter.ExpressionAdapter: it prefixes an emotion
// tag and a pace tag where each changes.
func (MistralExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
	emotion, hasEmotion := emotionOf(delivery)
	previousEmotion, hadEmotion := emotionOf(previousDelivery)
	pace, previousPace := paceOf(delivery), paceOf(previousDelivery)
	var tags string
	if hasEmotion && (!hadEmotion || emotion != previousEmotion) {
		tags += "[emotion=" + emotion + "]"
	}
	if pace != previousPace {
		if pace == "" {
			pace = normalPace
		}
		tags += "[pace=" + pace + "]"
	}
	if tags == "" {
		return text
	}
	return tags + " " + text
}

// FormatSound implements ttsadapter.ExpressionAdapter: a tag.
func (MistralExpressionAdapter) FormatSound(name string) string { return "[" + name + "]" }

// emotionOf is the first name in the delivery that is an emotion, not a pace,
// and false when there is none.
func emotionOf(delivery []string) (string, bool) {
	p := paces()
	for _, name := range delivery {
		if _, pace := p[name]; !pace {
			return name, true
		}
	}
	return "", false
}

// paceOf is the pace of the delivery's first speed, or "" for the normal pace.
func paceOf(delivery []string) string {
	p := paces()
	for _, name := range delivery {
		if pace, ok := p[name]; ok {
			return pace
		}
	}
	return ""
}

var _ ttsadapter.ExpressionAdapter = MistralExpressionAdapter{}
