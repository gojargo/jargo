package tagger

import (
	"cmp"
	"fmt"
	"slices"
	"strings"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/processor/tone/internal/shared"
)

// The tagger labels with expressive.CoreDelivery and expressive.CoreSounds,
// universal names, and draws a breath as one of BreathKinds. A call's label
// space holds those its TTS service performs and its policy offers, each with
// the name to write: its universal name, or a developer's own, which the TTS
// service then performs by its own name.

// BreathKinds is the kinds the tagger draws a breath as, each a universal
// sound.
func BreathKinds() []expressive.ExpressionSound {
	return []expressive.ExpressionSound{
		expressive.SoundLightBreath, expressive.SoundDeepBreath, expressive.SoundExhale,
	}
}

// OtherService is the service the model reads for one it was not trained with.
const OtherService = "other"

// trainedServices is the services the model was trained with, by the names it
// reads for them.
//
//nolint:gochecknoglobals // a read-only table
var trainedServices = []string{"cartesia", "elevenlabs", "fish", "minimax"}

// LabelName is a label with the name to write it as.
type LabelName struct {
	Label string
	Name  string
}

// Names is labels with the names to write them as, in order.
type Names []LabelName

// Name is the name to write label as, and false when the names hold no label.
func (n Names) Name(label string) (string, bool) {
	for _, l := range n {
		if l.Label == label {
			return l.Name, true
		}
	}
	return "", false
}

// Has reports whether the names hold label.
func (n Names) Has(label string) bool {
	_, ok := n.Name(label)
	return ok
}

// Labels is the labels, in order.
func (n Names) Labels() []string {
	out := make([]string, len(n))
	for i, l := range n {
		out[i] = l.Label
	}
	return out
}

// LabelSpace is the labels a call can use, each with the name to write it as.
type LabelSpace struct {
	// Deliveries is each delivery label with the name to write, in
	// expressive.CoreDelivery order.
	Deliveries Names
	// Sounds is each sound label with the name to write, in
	// expressive.CoreSounds order.
	Sounds Names
	// Emphasis is whether the service stresses ^^words^^.
	Emphasis bool
	// TTS is the service, by the name the model was trained to read for it.
	// Empty is OtherService.
	TTS string
	// Breaths is each kind of breath with the name to write, in BreathKinds
	// order, where a breath is offered. The model chooses only a breath, and
	// the tagger draws its kind.
	Breaths Names
}

// ResolveLabelSpace is the labels the service performs and the policy offers.
//
// A label is written as the first of the developer's names for it, then its
// own, that the service performs and the policy offers. A breath, and its
// kinds, are offered only with expressive.GuidanceBreathe in the policy. A
// breath is offered where the service performs only some of its kinds, written
// as the first of them for a model that draws no kinds.
//
// serviceName is the service's name, such as "ElevenLabsTTS#0", which says
// which of the services the model was trained with it is. A service given a
// name of its own, such as "tts", is read as one the model was not trained
// with. vocabulary is the names to try first for a label, before its own, by
// the label.
func ResolveLabelSpace(
	capabilities *expressive.ExpressionCapabilities,
	policy expressive.ExpressionPolicy,
	serviceName string,
	vocabulary map[string][]string,
) LabelSpace {
	breathes := policy.Breathes()
	var sounds []expressive.ExpressionSound
	for _, s := range expressive.CoreSounds() {
		if s != expressive.SoundBreath || breathes {
			sounds = append(sounds, s)
		}
	}
	resolved := resolve(sounds, capabilities.SoundName, vocabulary, policy.Sounds)
	// Each kind of breath is a sound of its own.
	var breaths Names
	if breathes {
		breaths = resolve(BreathKinds(), capabilities.SoundName, vocabulary, policy.Sounds)
	}
	// A breath is last in CoreSounds, so adding it keeps the order.
	if len(breaths) > 0 && !resolved.Has(string(expressive.SoundBreath)) {
		resolved = append(resolved, LabelName{string(expressive.SoundBreath), breaths[0].Name})
	}
	return LabelSpace{
		Deliveries: resolve(expressive.CoreDelivery(), capabilities.DeliveryName, vocabulary, policy.Delivery),
		Sounds:     resolved,
		Emphasis:   capabilities.Emphasis,
		TTS:        trainedService(serviceName),
		Breaths:    breaths,
	}
}

// SoundName is the name to write for a sound or a kind of breath, or empty if
// the space holds neither.
func (s LabelSpace) SoundName(label string) string {
	if name, _ := s.Sounds.Name(label); name != "" {
		return name
	}
	name, _ := s.Breaths.Name(label)
	return name
}

// Span is a delivery for some words, <<delivery: words>>, by code-point offsets
// in the sentence.
type Span struct {
	// Start is the offset of the first word's first character.
	Start int
	// End is the offset just past the last word.
	End int
	// Delivery is the words' delivery, a label from expressive.CoreDelivery.
	Delivery string
}

// SentenceLabels is what the tagger chose for one sentence, in its labels;
// usually nothing.
type SentenceLabels struct {
	// Delivery is the sentence's delivery, or empty for the voice's baseline.
	Delivery string
	// Spans is words in a delivery of their own, mainly slow details.
	Spans []Span
	// Emphasis is words to stress, as [start, end) code-point offsets.
	Emphasis [][2]int
	// SoundBefore is a sound before the sentence, or empty.
	SoundBefore string
	// SoundAfter is a sound after the sentence, or empty.
	SoundAfter string
}

// replyRenderer writes one reply's sentences as tags, a delivery only where it
// changes.
type replyRenderer struct {
	space LabelSpace
	// delivery is the delivery the reply is in, nil for the voice's baseline.
	delivery *string
}

func newReplyRenderer(space LabelSpace) *replyRenderer { return &replyRenderer{space: space} }

// render writes the next sentence of the reply with its labels as tags.
//
// Labels the space does not hold, or that would overlap or cut a word, are
// dropped, so the tags always speak exactly the sentence's words.
func (r *replyRenderer) render(sentence string, labels SentenceLabels) string {
	text := []rune(shared.Strip(sentence))
	if len(text) == 0 {
		return ""
	}
	kept := keep(text, labels, r.space)
	names := r.space.Deliveries

	// At one offset a close (order 0) comes before an open (order 1).
	type insert struct {
		at, order int
		tag       string
	}
	var inserts []insert
	for _, span := range kept.Spans {
		name, _ := names.Name(span.Delivery)
		inserts = append(inserts, insert{span.Start, 1, "<<" + name + ": "}, insert{span.End, 0, ">>"})
	}
	for _, e := range kept.Emphasis {
		inserts = append(inserts, insert{e[0], 1, "^^"}, insert{e[1], 0, "^^"})
	}
	// Last first, so each offset still points into the sentence as written.
	slices.SortStableFunc(inserts, func(a, b insert) int {
		return cmp.Or(cmp.Compare(b.at, a.at), cmp.Compare(b.order, a.order))
	})
	for _, in := range inserts {
		text = slices.Concat(text[:in.at], []rune(in.tag), text[in.at:])
	}

	out := string(text)
	if kept.SoundBefore != "" {
		out = fmt.Sprintf("[[%s]] %s", r.space.SoundName(kept.SoundBefore), out)
	}
	if kept.SoundAfter != "" {
		out = fmt.Sprintf("%s [[%s]]", out, r.space.SoundName(kept.SoundAfter))
	}
	var delivery *string
	if kept.Delivery != "" {
		name, _ := names.Name(kept.Delivery)
		delivery = &name
	}
	if !sameDelivery(delivery, r.delivery) {
		written := expressive.NeutralDelivery
		if delivery != nil && *delivery != "" {
			written = *delivery
		}
		out = fmt.Sprintf("<<%s>> %s", written, out)
		r.delivery = delivery
	}
	return out
}

func sameDelivery(a, b *string) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// trainedService is the service the model reads for serviceName: the first it
// was trained with whose name is in it, lower-cased, or OtherService.
func trainedService(serviceName string) string {
	name := string(lower(serviceName))
	for _, service := range trainedServices {
		if strings.Contains(name, service) {
			return service
		}
	}
	return OtherService
}

// resolve is each label's first name the service performs and the policy
// offers. The developer's names for a label are tried best first, then the
// label itself. A label with no such name is left out. A nil offered offers
// every name.
func resolve[T ~string](
	labels []T, performedName func(string) (string, bool), vocabulary map[string][]string, offered []T,
) Names {
	var resolved Names
	for _, label := range labels {
		for _, name := range append(slices.Clone(vocabulary[string(label)]), string(label)) {
			if _, ok := performedName(name); ok && (offered == nil || slices.Contains(offered, T(name))) {
				resolved = append(resolved, LabelName{string(label), name})
				break
			}
		}
	}
	return resolved
}

// keep is the labels the space holds, on whole words and not overlapping.
func keep(sentence []rune, labels SentenceLabels, space LabelSpace) SentenceLabels {
	var taken [][2]int
	var spans []Span
	sorted := slices.Clone(labels.Spans)
	slices.SortStableFunc(sorted, func(a, b Span) int { return cmp.Compare(a.Start, b.Start) })
	for _, span := range sorted {
		if space.Deliveries.Has(span.Delivery) && words(sentence, span.Start, span.End) &&
			!overlaps(taken, span.Start, span.End) {
			taken = append(taken, [2]int{span.Start, span.End})
			spans = append(spans, span)
		}
	}
	var emphasis [][2]int
	if space.Emphasis {
		sorted := slices.Clone(labels.Emphasis)
		slices.SortStableFunc(sorted, func(a, b [2]int) int {
			return cmp.Or(cmp.Compare(a[0], b[0]), cmp.Compare(a[1], b[1]))
		})
		for _, e := range sorted {
			if words(sentence, e[0], e[1]) && !overlaps(taken, e[0], e[1]) {
				taken = append(taken, e)
				emphasis = append(emphasis, e)
			}
		}
	}
	kept := SentenceLabels{Spans: spans, Emphasis: emphasis}
	if space.Deliveries.Has(labels.Delivery) {
		kept.Delivery = labels.Delivery
	}
	if space.SoundName(labels.SoundBefore) != "" {
		kept.SoundBefore = labels.SoundBefore
	}
	if space.SoundName(labels.SoundAfter) != "" {
		kept.SoundAfter = labels.SoundAfter
	}
	return kept
}

// words reports whether [start, end) of text is whole words, with no space at
// either end.
func words(text []rune, start, end int) bool {
	return start >= 0 && start < end && end <= len(text) &&
		!shared.IsSpace(text[start]) && !shared.IsSpace(text[end-1]) &&
		wordBoundary(text, start) && wordBoundary(text, end)
}

func wordBoundary(text []rune, at int) bool {
	return at == 0 || at == len(text) || !shared.IsAlnum(text[at-1]) || !shared.IsAlnum(text[at])
}

func overlaps(taken [][2]int, start, end int) bool {
	for _, t := range taken {
		if start < t[1] && t[0] < end {
			return true
		}
	}
	return false
}
