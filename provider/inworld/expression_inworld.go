// This file holds Inworld's expression adapter, which formats expression
// tags as Inworld TTS-2 audio markup for an expressive conversation (alpha).

package inworld

import (
	"slices"
	"strconv"
	"strings"

	"github.com/gojargo/jargo/expressive"
)

// Names the vocabulary uses more than once.
const (
	breathe = "breathe"
)

// resetTag returns an instruction to the voice's baseline.
const resetTag = "[reset]"

// sounds is Inworld's non-verbal tags.
func sounds() []expressive.ExpressionTerm {
	names := []string{
		breathe, "sigh", "gasp", "pant", "huff", "grunt", "groan", "moan", "laugh", "chuckle",
		"giggle", "cackle", "snort", "scoff", "cry", "sob", "whimper", "sniffle", "sniff", "shriek",
		"clear throat", "cough", "sneeze", "hiccup", "yawn", "burp", "snore", "swallow", "gulp",
		"tongue click", "lip smack", "shush", "whistle", "growl",
	}
	terms := make([]expressive.ExpressionTerm, 0, len(names))
	for _, name := range names {
		terms = append(terms, expressive.ExpressionTerm{Name: name})
	}
	return terms
}

func soundVocabulary() map[expressive.UniversalName][]string {
	return map[expressive.UniversalName][]string{
		expressive.SoundBreath:      {breathe},
		expressive.SoundLightBreath: {breathe},
	}
}

// FlashExpressionAdapter formats expression tags as Inworld TTS-2 Flash
// audio markup. Inworld TTS-2 Flash performs non-verbal tags, such as [laugh],
// and <break> pauses, and ignores instructions, so it has no delivery.
type FlashExpressionAdapter struct {
	expressive.BaseExpressionAdapter
}

// Capabilities implements expressive.ExpressionAdapter: Inworld's non-verbal
// sounds and pauses.
func (FlashExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:            sounds(),
		PauseRangeMS:      &[2]int{100, 10000},
		ServiceVocabulary: soundVocabulary(),
	}
}

// FormatSound implements expressive.ExpressionAdapter: a non-verbal tag, set
// apart so its alignment times it as a word of its own.
func (FlashExpressionAdapter) FormatSound(name string) string { return " [" + name + "] " }

// FormatPause implements expressive.ExpressionAdapter: a break, set apart from
// the words.
func (FlashExpressionAdapter) FormatPause(durationMS int) string {
	return ` <break time="` + strconv.Itoa(durationMS) + `ms"/> `
}

// ExpressionAdapter formats expression tags as Inworld TTS-2 audio
// markup.
//
// A delivery is an instruction tag, which Inworld TTS-2 reads as free text, so
// it performs any delivery described in a few words: <<calm>> is written
// [calm], and several deliveries at once, such as a slow span in a calm
// sentence, [calm and slow]. An instruction lasts past the sentence it is sent
// in, so a return to the voice's baseline within a sentence writes [reset], and
// so does the end of a sentence spoken in a delivery. Emphasis is written in
// capitals. Sounds are its non-verbal tags, and pauses are <break> tags.
type ExpressionAdapter struct {
	FlashExpressionAdapter
}

// Capabilities implements expressive.ExpressionAdapter: free-text delivery,
// Inworld's non-verbal sounds, emphasis and pauses.
func (ExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:                 sounds(),
		Delivery:               expressive.SpeedDelivery(),
		AllowsFreeTextDelivery: true,
		PauseRangeMS:           &[2]int{100, 10000},
		Emphasis:               true,
		ServiceVocabulary:      soundVocabulary(),
	}
}

// FormatText implements expressive.ExpressionAdapter: it prefixes an
// instruction where the delivery changes, and capitalizes emphasis.
func (ExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
	if slices.Contains(delivery, expressive.EmphasisDelivery) {
		text = strings.ToUpper(text)
	}
	tags := expressive.DeliveryTags(delivery, nil)
	previous := expressive.DeliveryTags(previousDelivery, nil)
	if slices.Equal(tags, previous) {
		return text
	}
	if len(tags) > 0 {
		return "[" + strings.Join(tags, " and ") + "] " + text
	}
	return resetTag + " " + text
}

// FormatEnd implements expressive.ExpressionAdapter: it sets the delivery back
// after a sentence spoken in one.
func (ExpressionAdapter) FormatEnd(delivery []string) string {
	if len(expressive.DeliveryTags(delivery, nil)) > 0 {
		return " " + resetTag
	}
	return ""
}

var (
	_ expressive.ExpressionAdapter = FlashExpressionAdapter{}
	_ expressive.ExpressionAdapter = ExpressionAdapter{}
)
