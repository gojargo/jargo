// This file holds Azure's expression adapter, which formats expression tags as
// Azure HD voice style markers for an expressive conversation (alpha).

package speech

import (
	"strings"

	"github.com/gojargo/jargo/expressive"
)

// neutralMarker returns to the voice's default style.
const neutralMarker = "[Neutral]"

func terms(names ...string) []expressive.ExpressionTerm {
	out := make([]expressive.ExpressionTerm, 0, len(names))
	for _, name := range names {
		out = append(out, expressive.ExpressionTerm{Name: name})
	}
	return out
}

// HDExpressionAdapter formats expression tags as Azure HD voice style
// markers.
//
// A delivery is a style marker in the text, such as [excited], from Azure's
// fixed list for HD voices, and slow and fast are its [slow] and [fast] styles.
// A voice speaks one style at a time, so a delivery of several is spoken in the
// one added last, such as the slow of a slow span in a calm sentence. A marker
// lasts until the next one or the end of its sentence, and [Neutral] returns to
// the voice's default style. Sounds are its paralinguistic markers, such as
// [laughter]. Styles are performed on English text only. HD voices have no
// emphasis, and their pauses are SSML, which isn't written into the text.
type HDExpressionAdapter struct {
	expressive.BaseExpressionAdapter
}

// Capabilities implements expressive.ExpressionAdapter: Azure's HD voice styles
// and paralinguistic sounds.
func (HDExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities {
	styles := terms(strings.Fields("amazed amused angry annoyed anxious appreciative calm cautious concerned " +
		"confident confused curious defeated determined disappointed disgusted doubtful ecstatic " +
		"encouraging excited fearful frustrated happy hesitant hurt impressed joking " +
		"optimistic panicked pleading proud quiet reassuring reflective relieved remorseful " +
		"resigned sad sarcastic serious shocked shouting shy skeptical surprised sympathetic " +
		"terrified upset urgent whispering")...)
	return &expressive.ExpressionCapabilities{
		Sounds:   terms("laughter", "coughing", "throat_clearing", "breathing", "sighing", "yawning"),
		Delivery: append(styles, expressive.SpeedDelivery()...),
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliveryApologetic:    {"remorseful"},
			expressive.DeliveryRegretful:     {"remorseful"},
			expressive.DeliveryWarm:          {"happy"},
			expressive.DeliveryGrateful:      {"appreciative"},
			expressive.DeliveryWorried:       {"concerned"},
			expressive.DeliveryDejected:      {"defeated"},
			expressive.DeliveryElated:        {"ecstatic"},
			expressive.DeliveryEuphoric:      {"ecstatic"},
			expressive.DeliveryPlayful:       {"joking"},
			expressive.DeliveryDesperate:     {"pleading"},
			expressive.DeliverySoft:          {"quiet"},
			expressive.DeliveryThoughtful:    {"reflective"},
			expressive.DeliveryContemplative: {"reflective"},
			expressive.DeliveryAlarmed:       {"shocked"},
			expressive.DeliverySheepish:      {"shy"},
			expressive.DeliveryScared:        {"terrified"},
			expressive.SoundLaugh:            {"laughter"},
			expressive.SoundChuckle:          {"laughter"},
			expressive.SoundCough:            {"coughing"},
			expressive.SoundClearThroat:      {"throat_clearing"},
			expressive.SoundBreath:           {"breathing"},
			expressive.SoundLightBreath:      {"breathing"},
			expressive.SoundSigh:             {"sighing"},
			expressive.SoundYawn:             {"yawning"},
		},
	}
}

// FormatText implements expressive.ExpressionAdapter: it prefixes a style
// marker where the style changes.
func (HDExpressionAdapter) FormatText(text string, delivery, previousDelivery []string) string {
	style, previous := styleOf(delivery), styleOf(previousDelivery)
	if style == previous {
		return text
	}
	if style == "" {
		return neutralMarker + " " + text
	}
	return "[" + style + "] " + text
}

// FormatSound implements expressive.ExpressionAdapter: a paralinguistic marker,
// set apart from the words.
func (HDExpressionAdapter) FormatSound(name string) string { return " [" + name + "] " }

// styleOf is the style a voice speaks a delivery in: its name added last, or ""
// for none.
func styleOf(delivery []string) string {
	tags := expressive.DeliveryTags(delivery, nil)
	if len(tags) == 0 {
		return ""
	}
	return tags[len(tags)-1]
}

var _ expressive.ExpressionAdapter = HDExpressionAdapter{}
