// Package fish is Fish Audio's expression adapter, which formats expression
// tags as Fish Audio S2 bracket tags for an expressive conversation (alpha).
package fish

import (
	"slices"
	"strings"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive"
)

// Names the vocabulary uses more than once.
const (
	soundEffect = "sound effect"
)

// normalPaceTag ends a speed delivery.
const normalPaceTag = "[normal pace]"

// speedTags is the tag each speed delivery is written as.
func speedTags() map[string]string {
	return map[string]string{expressive.SlowDelivery: "speaking slowly", expressive.FastDelivery: "speaking quickly"}
}

// s2Sounds and s2Delivery are Fish Audio's S2 emotion reference, sorted into
// sounds made at one point and deliveries that last.
func s2Sounds() []expressive.ExpressionTerm {
	return []expressive.ExpressionTerm{
		{Name: "laughing"},
		{Name: "chuckling"},
		{Name: "sobbing"},
		{Name: "crying loudly"},
		{Name: "sighing"},
		{Name: "groaning"},
		{Name: "panting"},
		{Name: "gasping"},
		{Name: "yawning"},
		{Name: "snoring"},
		{Name: "clear throat"},
		{Name: "audience laughing", Description: soundEffect},
		{Name: "background laughter", Description: soundEffect},
		{Name: "crowd laughing", Description: soundEffect},
	}
}

func s2Delivery() []expressive.ExpressionTerm {
	names := "happy sad angry excited calm nervous confident surprised satisfied delighted " +
		"scared worried upset frustrated depressed empathetic embarrassed disgusted moved " +
		"proud relaxed grateful curious sarcastic disdainful unhappy anxious hysterical " +
		"indifferent uncertain doubtful confused disappointed regretful guilty ashamed " +
		"jealous envious hopeful optimistic pessimistic nostalgic lonely bored contemptuous " +
		"sympathetic compassionate determined resigned shouting screaming whispering"
	var terms []expressive.ExpressionTerm
	for name := range strings.FieldsSeq(names) {
		terms = append(terms, expressive.ExpressionTerm{Name: name})
	}
	return terms
}

// FishAudioS2ExpressionAdapter formats expression tags as Fish Audio S2
// bracket tags.
//
// Delivery and sounds are both tags, such as [excited] or [laughing], written
// in free text: the terms listed are Fish Audio's S2 emotion reference, and
// others described in a few words work too. A tag applies within the input it
// is sent in, so each sentence states its delivery from its start. There is no
// tag for the voice's baseline, so a return to it writes none. Slow and fast
// are written as [speaking slowly] and [speaking quickly], and since a pace
// lasts until another tag changes it, a [normal pace] tag ends them, within a
// sentence or at the end of one. Emphasis is an [emphasis] tag before the
// stressed words, and a pause is a break or a long break, which is
// approximate.
type FishAudioS2ExpressionAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

// Capabilities implements ttsadapter.ExpressionAdapter: free-text tags,
// emphasis and approximate pauses.
func (FishAudioS2ExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:                 s2Sounds(),
		Delivery:               append(s2Delivery(), expressive.SpeedDelivery()...),
		AllowsFreeTextSounds:   true,
		AllowsFreeTextDelivery: true,
		PauseRangeMS:           &[2]int{100, 1500},
		Emphasis:               true,
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliveryCalm:        {"calm", "relaxed"},
			expressive.DeliverySympathetic: {"sympathetic", "empathetic", "compassionate"},
			expressive.DeliveryApologetic:  {"regretful"},
			expressive.DeliveryWarm:        {"happy", "relaxed"},
			expressive.DeliveryExcited:     {"excited", "delighted"},
			expressive.SoundLaugh:          {"laughing"},
			expressive.SoundChuckle:        {"chuckling"},
			expressive.SoundSigh:           {"sighing"},
			expressive.SoundBreath:         {"inhaling"},
			expressive.SoundLightBreath:    {"inhaling"},
			expressive.SoundExhale:         {"exhaling"},
			expressive.SoundSob:            {"sobbing", "crying loudly"},
			expressive.SoundGroan:          {"groaning"},
			expressive.SoundPant:           {"panting"},
			expressive.SoundGasp:           {"gasping"},
			expressive.SoundYawn:           {"yawning"},
			expressive.SoundSnore:          {"snoring"},
		},
	}
}

// FormatText implements ttsadapter.ExpressionAdapter: it prefixes tags where
// the delivery changes, and emphasis before stressed words.
func (FishAudioS2ExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
	tags := ttsadapter.DeliveryTags(delivery, speedTags())
	previous := ttsadapter.DeliveryTags(previousDelivery, speedTags())
	var prefix string
	if len(tags) > 0 && !slices.Equal(tags, previous) {
		for _, tag := range tags {
			prefix += "[" + tag + "]"
		}
	}
	if hasSpeed(previousDelivery) && !hasSpeed(delivery) {
		prefix = normalPaceTag + prefix
	}
	if slices.Contains(delivery, expressive.EmphasisDelivery) {
		prefix += "[emphasis]"
	}
	if prefix == "" {
		return text
	}
	return prefix + " " + text
}

// FormatEnd implements ttsadapter.ExpressionAdapter: it sets the pace back
// after a request that ends in a speed delivery.
func (FishAudioS2ExpressionAdapter) FormatEnd(delivery []string) string {
	if hasSpeed(delivery) {
		return " " + normalPaceTag
	}
	return ""
}

// FormatSound implements ttsadapter.ExpressionAdapter: a tag.
func (FishAudioS2ExpressionAdapter) FormatSound(name string) string { return "[" + name + "]" }

// FormatPause implements ttsadapter.ExpressionAdapter: a break under a second,
// otherwise a long break.
func (FishAudioS2ExpressionAdapter) FormatPause(durationMS int) string {
	if durationMS < 1000 {
		return "[break]"
	}
	return "[long-break]"
}

func hasSpeed(delivery []string) bool {
	tags := speedTags()
	return slices.ContainsFunc(delivery, func(name string) bool { _, ok := tags[name]; return ok })
}

var _ ttsadapter.ExpressionAdapter = FishAudioS2ExpressionAdapter{}
