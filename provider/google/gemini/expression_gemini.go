// This file holds Gemini's expression adapter, which performs expression tags
// with Gemini 3.8 TTS models for an expressive conversation (alpha).

package gemini

import (
	"slices"
	"strings"

	"github.com/gojargo/jargo/expressive"
)

func speedStyles() map[string]string {
	return map[string]string{expressive.SlowDelivery: "speaking slowly", expressive.FastDelivery: "speaking quickly"}
}

// TTSExpressionAdapter performs expression tags with Gemini 3.8 TTS
// models.
//
// Gemini 3.8 reads its text as a verbatim transcript and takes delivery as a
// free-text style with each request, so it performs any delivery described in
// a few words, such as "calm, speaking slowly", and a delivery can change only
// between sentences. Sounds and pauses are its inline audio tags, such as
// <laugh> and <short pause>, and emphasis is written in capitals.
type TTSExpressionAdapter struct {
	expressive.BaseExpressionAdapter
}

// Capabilities implements expressive.ExpressionAdapter: free-text delivery,
// Gemini's audio tags, emphasis and approximate pauses.
func (TTSExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	names := []string{
		"laugh", "chuckle", "giggle", "cackle", "sigh", "breath", "heavy breath", "exhales", "gasp",
		"pant", "cough", "throat-clearing", "sneeze", "yawn", "snort", "groan", "grunt", "moan",
		"growl", "hiss", "sob", "cry", "whimper", "shriek",
	}
	sounds := make([]expressive.ExpressionTerm, 0, len(names))
	for _, name := range names {
		sounds = append(sounds, expressive.ExpressionTerm{Name: name})
	}
	return &expressive.ExpressionCapabilities{
		Sounds:                 sounds,
		Delivery:               expressive.SpeedDelivery(),
		AllowsFreeTextDelivery: true,
		PauseRangeMS:           &[2]int{100, 2000},
		Emphasis:               true,
		DeliveryScope:          expressive.DeliveryScopeSentence,
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.SoundLightBreath: {"breath"},
			expressive.SoundDeepBreath:  {"heavy breath"},
			expressive.SoundExhale:      {"exhales"},
			expressive.SoundClearThroat: {"throat-clearing"},
		},
	}
}

// FormatText implements expressive.ExpressionAdapter: it capitalizes emphasis;
// the delivery goes in the request's style.
func (TTSExpressionAdapter) FormatText(text string, delivery, _ []string) string {
	if slices.Contains(delivery, expressive.EmphasisDelivery) {
		return strings.ToUpper(text)
	}
	return text
}

// RequestSettings implements expressive.ExpressionAdapter: the style a request
// in this delivery asks for, sent as its speech metadata.
func (TTSExpressionAdapter) RequestSettings(delivery []string) map[string]any {
	styles := expressive.DeliveryTags(delivery, speedStyles())
	if len(styles) == 0 {
		return nil
	}
	return map[string]any{"style": strings.Join(styles, ", ")}
}

// FormatSound implements expressive.ExpressionAdapter: an audio tag.
func (TTSExpressionAdapter) FormatSound(name string) string { return " <" + name + "> " }

// FormatPause implements expressive.ExpressionAdapter: a short pause under a
// second, otherwise a long pause.
func (TTSExpressionAdapter) FormatPause(durationMS int) string {
	if durationMS < 1000 {
		return " <short pause> "
	}
	return " <long pause> "
}

var _ expressive.ExpressionAdapter = TTSExpressionAdapter{}
