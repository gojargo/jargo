// Package inworld is Inworld's expression adapter, which formats expression
// tags as Inworld TTS-2 audio markup for an expressive conversation (alpha).
package inworld

import (
	"slices"
	"strconv"
	"strings"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
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

// InworldFlashExpressionAdapter formats expression tags as Inworld TTS-2 Flash
// audio markup. Inworld TTS-2 Flash performs non-verbal tags, such as [laugh],
// and <break> pauses, and ignores instructions, so it has no delivery.
type InworldFlashExpressionAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

// Capabilities implements ttsadapter.ExpressionAdapter: Inworld's non-verbal
// sounds and pauses.
func (InworldFlashExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:            sounds(),
		PauseRangeMS:      &[2]int{100, 10000},
		ServiceVocabulary: soundVocabulary(),
	}
}

// FormatSound implements ttsadapter.ExpressionAdapter: a non-verbal tag, set
// apart so its alignment times it as a word of its own.
func (InworldFlashExpressionAdapter) FormatSound(name string) string { return " [" + name + "] " }

// FormatPause implements ttsadapter.ExpressionAdapter: a break, set apart from
// the words.
func (InworldFlashExpressionAdapter) FormatPause(durationMS int) string {
	return ` <break time="` + strconv.Itoa(durationMS) + `ms"/> `
}

// InworldExpressionAdapter formats expression tags as Inworld TTS-2 audio
// markup.
//
// A delivery is an instruction tag, which Inworld TTS-2 reads as free text, so
// it performs any delivery described in a few words: <<calm>> is written
// [calm], and several deliveries at once, such as a slow span in a calm
// sentence, [calm and slow]. An instruction lasts past the sentence it is sent
// in, so a return to the voice's baseline within a sentence writes [reset], and
// so does the end of a sentence spoken in a delivery. Emphasis is written in
// capitals. Sounds are its non-verbal tags, and pauses are <break> tags.
type InworldExpressionAdapter struct {
	InworldFlashExpressionAdapter
}

// Capabilities implements ttsadapter.ExpressionAdapter: free-text delivery,
// Inworld's non-verbal sounds, emphasis and pauses.
func (InworldExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:                 sounds(),
		Delivery:               expressive.SpeedDelivery(),
		AllowsFreeTextDelivery: true,
		PauseRangeMS:           &[2]int{100, 10000},
		Emphasis:               true,
		ServiceVocabulary:      soundVocabulary(),
	}
}

// FormatText implements ttsadapter.ExpressionAdapter: it prefixes an
// instruction where the delivery changes, and capitalizes emphasis.
func (InworldExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
	if slices.Contains(delivery, expressive.EmphasisDelivery) {
		text = strings.ToUpper(text)
	}
	tags := ttsadapter.DeliveryTags(delivery, nil)
	previous := ttsadapter.DeliveryTags(previousDelivery, nil)
	if slices.Equal(tags, previous) {
		return text
	}
	if len(tags) > 0 {
		return "[" + strings.Join(tags, " and ") + "] " + text
	}
	return resetTag + " " + text
}

// FormatEnd implements ttsadapter.ExpressionAdapter: it sets the delivery back
// after a sentence spoken in one.
func (InworldExpressionAdapter) FormatEnd(delivery []string) string {
	if len(ttsadapter.DeliveryTags(delivery, nil)) > 0 {
		return " " + resetTag
	}
	return ""
}

var (
	_ ttsadapter.ExpressionAdapter = InworldFlashExpressionAdapter{}
	_ ttsadapter.ExpressionAdapter = InworldExpressionAdapter{}
)
