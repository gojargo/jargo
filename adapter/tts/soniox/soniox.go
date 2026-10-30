// Package soniox is Soniox's expression adapter, which formats expression tags
// as Soniox TTS audio tags for an expressive conversation (alpha).
package soniox

import (
	"slices"
	"strings"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive"
)

// neutralTag returns to the voice's baseline.
const neutralTag = "[neutral]"

func speedTags() map[string]string {
	return map[string]string{expressive.SlowDelivery: "slowly", expressive.FastDelivery: "quickly"}
}

func terms(names ...string) []expressive.ExpressionTerm {
	out := make([]expressive.ExpressionTerm, 0, len(names))
	for _, name := range names {
		out = append(out, expressive.ExpressionTerm{Name: name})
	}
	return out
}

// SonioxExpressionAdapter formats expression tags as Soniox TTS audio tags.
//
// Delivery and sounds are both tags, such as [excited] or [laughs], from
// Soniox's emotion and tone reference; Soniox may read a tag outside it aloud.
// A tag lasts past the sentence it is sent in, so a return to the voice's
// baseline within a sentence writes [neutral], and so does the end of a
// sentence spoken in a delivery. Slow and fast are written as [slowly] and
// [quickly], emphasis as *word*, and a pause as a pause or a long pause, which
// is approximate.
type SonioxExpressionAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

// Capabilities implements ttsadapter.ExpressionAdapter: Soniox's tags, emphasis
// and approximate pauses.
func (SonioxExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	delivery := terms(strings.Fields("happy sad angry excited nervous fearful surprised annoyed relieved " +
		"disappointed curious delighted calm warm stern serious playful sarcastic flirty deadpan " +
		"sincerely reassuringly dramatically mockingly whispering softly loudly shouting " +
		"muttering hesitantly")...)
	return &expressive.ExpressionCapabilities{
		Sounds: terms("laughs", "chuckles", "giggles", "sobs", "sniffles", "whimpers", "sighs",
			"exhales", "gasps", "grunts", "groans", "clears throat", "coughs", "yawns"),
		Delivery:     append(delivery, expressive.SpeedDelivery()...),
		PauseRangeMS: &[2]int{100, 2000},
		Emphasis:     true,
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliverySympathetic: {"warm"},
			expressive.DeliveryApologetic:  {"sincerely"},
			expressive.DeliveryGrateful:    {"warm"},
			expressive.DeliveryFlirtatious: {"flirty"},
			expressive.DeliverySincere:     {"sincerely"},
			expressive.DeliveryReassuring:  {"reassuringly"},
			expressive.DeliveryDramatic:    {"dramatically"},
			expressive.DeliveryMocking:     {"mockingly"},
			expressive.DeliverySoft:        {"softly"},
			expressive.DeliveryLoud:        {"loudly"},
			expressive.DeliveryHesitant:    {"hesitantly"},
			expressive.SoundLaugh:          {"laughs"},
			expressive.SoundChuckle:        {"chuckles"},
			expressive.SoundGiggle:         {"giggles"},
			expressive.SoundSob:            {"sobs"},
			expressive.SoundSniffle:        {"sniffles"},
			expressive.SoundWhimper:        {"whimpers"},
			expressive.SoundSigh:           {"sighs"},
			expressive.SoundBreath:         {"exhales"},
			expressive.SoundExhale:         {"exhales"},
			expressive.SoundGasp:           {"gasps"},
			expressive.SoundGrunt:          {"grunts"},
			expressive.SoundGroan:          {"groans"},
			expressive.SoundClearThroat:    {"clears throat"},
			expressive.SoundCough:          {"coughs"},
			expressive.SoundYawn:           {"yawns"},
		},
	}
}

// FormatText implements ttsadapter.ExpressionAdapter: it prefixes tags where
// the delivery changes, and marks emphasis.
func (SonioxExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
	if slices.Contains(delivery, expressive.EmphasisDelivery) {
		text = "*" + text + "*"
	}
	tags := ttsadapter.DeliveryTags(delivery, speedTags())
	previous := ttsadapter.DeliveryTags(previousDelivery, speedTags())
	if slices.Equal(tags, previous) {
		return text
	}
	if len(tags) > 0 {
		var b strings.Builder
		for _, tag := range tags {
			b.WriteString("[" + tag + "] ")
		}
		return b.String() + text
	}
	return neutralTag + " " + text
}

// FormatEnd implements ttsadapter.ExpressionAdapter: it sets the delivery back
// after a sentence spoken in one.
func (SonioxExpressionAdapter) FormatEnd(delivery []string) string {
	if len(ttsadapter.DeliveryTags(delivery, speedTags())) > 0 {
		return " " + neutralTag
	}
	return ""
}

// FormatSound implements ttsadapter.ExpressionAdapter: a tag, set apart so its
// timestamps time it as a word of its own.
func (SonioxExpressionAdapter) FormatSound(name string) string { return " [" + name + "] " }

// FormatPause implements ttsadapter.ExpressionAdapter: a pause under a second,
// otherwise a long pause, set apart like a sound.
func (SonioxExpressionAdapter) FormatPause(durationMS int) string {
	if durationMS < 1000 {
		return " [pause] "
	}
	return " [long pause] "
}

var _ ttsadapter.ExpressionAdapter = SonioxExpressionAdapter{}
