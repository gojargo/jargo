package tagger

import (
	"cmp"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/processor/tone/internal/shared"
)

// What the tagger model reads: the call so far, as the text it was trained on.
//
// For each sentence the model reads a text pair. The first text is the end of
// the call. The second holds what the TTS service performs, the policy's
// presets, how the user sounded, the tags of the bot's previous reply, state
// hints from the whole call, the reply so far and the sentence, last. State
// hints carry what the end of the call leaves out.
//
// Training wrote the same text, so it must not change for a released model.

// presetNames is the presets the model reads, by the names it was trained on.
//
//nolint:gochecknoglobals // a read-only table
var presetNames = map[string]string{
	string(expressive.GuidanceMatchStyle):     "match style",
	string(expressive.GuidanceSlowForDetails): "slow for details",
	string(expressive.GuidanceBreathe):        "breathe",
}

// emotionToTone folds the reader's emotions into the tone signal the model was
// trained on, by adding their confidences.
//
//nolint:gochecknoglobals // a read-only table
var emotionToTone = map[string]string{
	"neutral":      toneNeutral,
	"happy":        tonePositive,
	"amused":       tonePositive,
	"surprised":    toneNeutral,
	"sad":          toneSad,
	"afraid":       toneSad,
	"angry":        toneFrustrated,
	"disgusted":    toneFrustrated,
	"contemptuous": toneFrustrated,
}

// The tones the model was trained on.
const (
	toneNeutral    = "neutral"
	tonePositive   = "positive"
	toneSad        = "sad"
	toneFrustrated = "frustrated"
)

const (
	// roleUser is the role of the user's messages.
	roleUser = "user"
	// none is the word the model reads and writes for nothing.
	none = "none"
	// breath is a breath, the sound the model chooses with a head of its own.
	breath = string(expressive.SoundBreath)
)

const (
	clearBand  = 0.7
	likelyBand = 0.5
)

// Message is a turn of the call, as plain words.
type Message struct {
	// Role is "user" or "assistant".
	Role string
	// Text is the words.
	Text string
	// Heard is how the user sounded, folded into the model's tones, or nil.
	Heard []expressive.PerceptionSignal
	// Speculation is whether it is the turn an early reply answers, not yet
	// confirmed.
	Speculation bool
}

// TurnContext is what the model knows when a reply starts.
type TurnContext struct {
	// Messages is the call so far.
	Messages []Message
	// Space is the labels the call can use.
	Space LabelSpace
	// Policy is the pipeline's expression policy.
	Policy expressive.ExpressionPolicy
	// Previous is the labels of the bot's previous reply, or nil before the
	// first. A previous reply with no sentences is empty, not nil.
	Previous []SentenceLabels
}

// Heard is how the user sounded in the turn this reply answers.
func (t TurnContext) Heard() []expressive.PerceptionSignal {
	for _, m := range slices.Backward(t.Messages) {
		if m.Role == roleUser {
			return m.Heard
		}
	}
	return nil
}

// layout is what a model reads, as its config records it was trained.
type layout struct {
	// ContextTurns is the messages of the call.
	ContextTurns int `json:"context_turns"`
	// Earlier is the reply's earlier sentences.
	Earlier int `json:"earlier"`
	// CustomGuidance is whether the policy's own lines are read too.
	CustomGuidance bool `json:"custom_guidance"`
	// State is whether state hints are read.
	State bool `json:"state"`
	// MaxTokens is the token budget.
	MaxTokens int `json:"max_tokens"`
	// CutCallFirst is whether, over budget, the call's oldest tokens go first.
	CutCallFirst bool `json:"cut_call_first"`
	// PreviousTags is whether the labels of the bot's previous reply are read.
	PreviousTags bool `json:"previous_tags"`
}

func defaultLayout() layout {
	return layout{ContextTurns: 1, Earlier: 1, State: true, MaxTokens: 160}
}

// parseLayout reads a config's layout over the defaults, refusing a setting it
// does not know.
func parseLayout(raw json.RawMessage) (layout, error) {
	l := defaultLayout()
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&l); err != nil {
		return layout{}, fmt.Errorf("layout: %w", err)
	}
	return l, nil
}

// fold folds an emotion signal into a tone signal, leaving the others as they
// are.
func fold(heard []expressive.PerceptionSignal) []expressive.PerceptionSignal {
	if len(heard) == 0 {
		return heard
	}
	folded := make([]expressive.PerceptionSignal, 0, len(heard))
	for _, signal := range heard {
		if signal.Name != "emotion" || len(signal.Labels) == 0 {
			folded = append(folded, signal)
			continue
		}
		var tones []expressive.PerceptionLabel
		for _, l := range signal.Labels {
			tone, ok := emotionToTone[l.Label]
			if !ok {
				continue
			}
			i := slices.IndexFunc(tones, func(t expressive.PerceptionLabel) bool { return t.Label == tone })
			if i < 0 {
				tones = append(tones, expressive.PerceptionLabel{Label: tone})
				i = len(tones) - 1
			}
			tones[i].Confidence += l.Confidence
		}
		for i := range tones {
			tones[i].Confidence = shared.Round(min(tones[i].Confidence, 1.0), 2)
		}
		slices.SortStableFunc(tones, func(a, b expressive.PerceptionLabel) int {
			return cmp.Compare(b.Confidence, a.Confidence)
		})
		folded = append(folded, expressive.PerceptionSignal{Name: "tone", Labels: tones[:min(2, len(tones))]})
	}
	return folded
}

// encode writes one sentence's text pair, and where the sentence starts in the
// second, in code points.
func encode(turn TurnContext, earlier []string, sentence string, l layout) (call, text string, start int) {
	var lines []string
	if l.ContextTurns != 0 {
		for _, m := range turn.Messages[last(len(turn.Messages), l.ContextTurns):] {
			if m.Role == roleUser {
				lines = append(lines, fmt.Sprintf("caller (%s): %s", describe(m.Heard), m.Text))
			} else {
				lines = append(lines, "agent: "+m.Text)
			}
		}
	}

	var presets, custom []string
	for _, g := range turn.Policy.Guidance {
		if g == nil {
			continue
		}
		if name, ok := presetNames[g.String()]; ok {
			presets = append(presets, name)
		} else if l.CustomGuidance {
			custom = append(custom, g.String())
		}
	}
	space := turn.Space
	tts := space.TTS
	if tts == "" {
		tts = OtherService
	}
	policy := or(strings.Join(presets, ", "), none)
	if len(custom) > 0 {
		policy += "; " + strings.Join(custom, " ")
	}
	stress := "no"
	if space.Emphasis {
		stress = "yes"
	}
	header := []string{
		"tts " + tts,
		"can " + or(strings.Join(space.Deliveries.Labels(), " "), "nothing"),
		"sounds " + or(strings.Join(space.Sounds.Labels(), " "), none),
		"stress " + stress,
		"policy " + policy,
		"caller sounded " + describe(turn.Heard()),
	}
	if l.PreviousTags {
		header = append(header, "last reply "+lastReply(turn.Previous))
	}
	if l.State {
		header = append(header, "state "+or(strings.Join(stateHints(turn.Messages, sentence), ", "), none))
	}
	var before string
	if l.Earlier != 0 {
		before = strings.Join(earlier[last(len(earlier), l.Earlier):], " ")
	}
	prefix := strings.Join(header, " | ") + " | reply so far: " + or(before, "-") + " | sentence: "
	return strings.Join(lines, "\n"), prefix + sentence, utf8.RuneCountInString(prefix)
}

// last is where Python's slice [-n:] of a sequence of length size starts.
func last(size, n int) int {
	start := -n
	if start < 0 {
		start += size
	}
	return min(max(start, 0), size)
}

// or is s, or otherwise when s is empty.
func or(s, otherwise string) string {
	if s == "" {
		return otherwise
	}
	return s
}

// stateHints is facts from the whole call that the model's short window
// misses.
func stateHints(messages []Message, sentence string) []string {
	var agent, user []string
	for _, m := range messages {
		if m.Role == roleUser {
			user = append(user, m.Text)
		} else {
			agent = append(agent, m.Text)
		}
	}
	var hints []string
	if slices.ContainsFunc(agent, apologizes) {
		hints = append(hints, "apologized before")
	}
	if slices.ContainsFunc(user, asksToSlowOrRepeat) {
		hints = append(hints, "caller asked to slow down or repeat")
	}
	text := []rune(sentence)
	var found []string
	for _, d := range details(text) {
		found = append(found, normalize(string(text[d[0]:d[1]])))
	}
	if len(found) > 0 {
		said := normalize(strings.Join(agent, " "))
		var lastTurn string
		if len(user) > 0 {
			lastTurn = normalize(user[len(user)-1])
		}
		in := func(text string) func(string) bool {
			return func(d string) bool { return strings.Contains(text, d) }
		}
		switch {
		case slices.ContainsFunc(found, in(lastTurn)):
			hints = append(hints, "caller just read this detail back")
		case slices.ContainsFunc(found, in(said)):
			hints = append(hints, "detail given before")
		default:
			hints = append(hints, "new detail")
		}
	}
	return hints
}

// lastReply is the deliveries and sounds the previous reply used, in order:
// "none" before the first.
func lastReply(labels []SentenceLabels) string {
	if labels == nil {
		return none
	}
	var used []string
	for _, s := range labels {
		used = append(used, s.SoundBefore, s.Delivery)
		for _, span := range s.Spans {
			used = append(used, span.Delivery)
		}
		used = append(used, s.SoundAfter)
	}
	var unique []string
	for _, u := range used {
		if u != "" && !slices.Contains(unique, u) {
			unique = append(unique, u)
		}
	}
	return or(strings.Join(unique, " "), "plain")
}

// describe is a reading in bands: "tone sad clear, neutral guess; sarcasm low".
func describe(heard []expressive.PerceptionSignal) string {
	if len(heard) == 0 {
		return "no reading"
	}
	var parts []string
	for _, signal := range heard {
		switch {
		case len(signal.Labels) > 0:
			bands := make([]string, len(signal.Labels))
			for i, l := range signal.Labels {
				bands[i] = l.Label + " " + band(l.Confidence)
			}
			parts = append(parts, signal.Name+" "+strings.Join(bands, ", "))
		case signal.Value != nil:
			parts = append(parts, signal.Name+" "+level(*signal.Value))
		}
	}
	return strings.Join(parts, "; ")
}

func band(confidence float64) string {
	switch {
	case confidence >= clearBand:
		return "clear"
	case confidence >= likelyBand:
		return "likely"
	default:
		return "guess"
	}
}

func level(value float64) string {
	switch {
	case value >= clearBand:
		return "high"
	case value >= likelyBand:
		return "some"
	default:
		return "low"
	}
}
