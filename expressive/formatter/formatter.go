// Package formatter reads the expression tags an LLM writes, and formats them
// for a TTS service.
//
// Part of expressive conversation (alpha). A TTS service with expressive
// conversation on is given its text a unit at a time: a sentence, or a token
// when it streams tokens. [ExpressionFormatter] handles each unit in three
// steps:
//
//  1. [ExpressionFormatter.Parse] splits the unit into segments: the words to
//     speak, with the delivery they are spoken in, and the sounds and pauses
//     between them.
//  2. [ShownText] gives the words without the tags, which clients are shown.
//  3. [ExpressionFormatter.Format] builds the request the service is sent, in
//     its provider's markup, and says what each stretch of the unit performs,
//     for the spoken package to report as the words are spoken.
//
// With ElevenLabs' audio tags, one sentence goes:
//
//	"<<warm>> It's [[laugh]] ^^back^^ up."    the unit, as the LLM wrote it
//	"It's back up."                           shown to clients
//	"[warmly] It's [laughs] BACK up."         sent to ElevenLabs
//
// The tags are:
//
//   - <<name>> sets the delivery the words after it are spoken in, until the
//     next delivery tag replaces it or the response ends: <<warm>>. A name may
//     be several words, as in <<soft tone>>. <<neutral>> returns to the voice's
//     baseline. Delivery tags aren't segments of their own; the delivery
//     travels with the [Text] it applies to.
//   - <<name: words>> is a span: it speaks just those words in its delivery,
//     added to the delivery around them, then goes back to it, as in
//     <<slow: A B 7>>. <<neutral: words>> speaks them in the baseline. A span
//     holds words and emphasis, and doesn't nest.
//   - ^^words^^ stresses words. It is a span of emphasis, written short, and
//     can stand within another span.
//   - [[name]] is a [Sound], a sound at one point ([[laugh]]), or a [Pause],
//     [[pause N]] for N milliseconds.
//
// The brackets decide a tag's kind, except that a universal name, or one the
// TTS service declares, keeps its kind in the other kind's brackets: [[warm]]
// still sets a delivery. LLMs make that slip, so it is logged as a warning.
//
// The formatter reads whole units: an aggregator such as
// text.SkipTagsAggregator, given [ExpressionTags], keeps each tag in one unit as
// text streams. A tag left unfinished at the end of a unit is removed, never
// spoken.
//
// The service's expression adapter says what it can perform and formats each
// piece in its provider's markup. A universal name, such as "warm" or "laugh",
// is sent as the name the service performs it by
// (expressive.ExpressionCapabilities.DeliveryName), such as ElevenLabs'
// "warmly"; everything else about the unit keeps the names as written. What the
// service can't perform is left out of the request, and kept in the unit's text
// for the context.
//
// Alpha: the tag syntax and these types are likely to change between releases.
package formatter

import (
	"log/slog"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/text"
)

// ExpressionTags is the delimiters of the tags, for an aggregator such as
// text.SkipTagsAggregator to keep each tag in one unit.
func ExpressionTags() []text.StartEndTags {
	return []text.StartEndTags{{Start: "<<", End: ">>"}, {Start: "[[", End: "]]"}, {Start: "^^", End: "^^"}}
}

// A word character, as a tag names its delivery or sound: a letter, a digit or
// an underscore, in any script.
const word = `\pL\pN_`

//nolint:gochecknoglobals // compiled once, read-only
var (
	// tagRe reads a tag. A tag holds words, digits and a little punctuation;
	// anything else is text. A span names its delivery before its colon, then
	// the words it delivers.
	tagRe = regexp.MustCompile(
		`<<([` + word + ` '\-/]+?):[ \t]*([^<>\n]+?)>>` +
			`|<<([` + word + ` '\-:/]*)>>|\[\[([` + word + ` '\-:/]*)\]\]|\^\^([^\n]+?)\^\^`)
	emphasisRe = regexp.MustCompile(`\^\^([^\n]+?)\^\^`)
	// unfinishedRe is what is left of a tag the unit ended in the middle of.
	unfinishedRe = regexp.MustCompile(
		`(?:<<|\[\[)(?:[` + word + `'\-:/][` + word + ` '\-:/]*)?\n?\z|<<[` + word + ` '\-/]+:[^<>\n]*\n?\z`)
	pauseRe = regexp.MustCompile(`^pause ([0-9]+)$`)
	// spacesRe is spaces and tabs left side by side where a tag stood between
	// two words.
	spacesRe = regexp.MustCompile(`[ \t]{2,}`)
	// markupRe is the delivery and sound tags EXPRESSION_TAG_PATTERN reads. Its
	// first alternative is a whole span, of which only the opening is passed
	// over; see TagPattern.
	markupRe = regexp.MustCompile(
		`<<[` + word + ` '\-/]+?:[ \t]*([^<>\n]+?)>>|<<[^<>\n]*>>|>>|\[\[[^\[\]\n]*\]\]`)
)

// unclosedEmphasis is an emphasis that never closed.
const unclosedEmphasis = "^^"

// TagPattern finds the delivery and sound tags in the LLM's text, which word
// tracking passes over so their names aren't matched against spoken words: a
// span's opening and its closing ">>", not the words between them, then whole
// delivery and sound tags. Emphasis has no letters in its delimiters, so it
// needs none.
//
// It reads as a regular expression would whose span opening looks ahead for
// the ">>" closing it, which Go's regular expressions can't write.
type TagPattern struct{}

// ExpressionTagPattern finds the delivery and sound tags in the LLM's text.
// See [TagPattern].
//
//nolint:gochecknoglobals // stateless, read-only
var ExpressionTagPattern = TagPattern{}

// FindAllStringIndex reports where each tag is in s, as regexp.Regexp does: at
// most n of them, or all when n is negative.
func (TagPattern) FindAllStringIndex(s string, n int) [][]int {
	var out [][]int
	for pos := 0; pos <= len(s) && (n < 0 || len(out) < n); {
		loc := markupRe.FindStringSubmatchIndex(s[pos:])
		if loc == nil {
			break
		}
		start, end := pos+loc[0], pos+loc[1]
		if loc[2] >= 0 {
			// A span: its opening, up to the words it delivers, which are read
			// for tags of their own, then its closing.
			end = pos + loc[2]
		}
		out = append(out, []int{start, end})
		pos = end
	}
	return out
}

// ReplaceAllString returns s with each tag replaced by repl.
func (p TagPattern) ReplaceAllString(s, repl string) string {
	var b strings.Builder
	last := 0
	for _, loc := range p.FindAllStringIndex(s, -1) {
		b.WriteString(s[last:loc[0]])
		b.WriteString(repl)
		last = loc[1]
	}
	b.WriteString(s[last:])
	return b.String()
}

// Segment is a piece of a unit: a [Text], a [Sound] or a [Pause].
type Segment interface {
	// Bounds is where the segment was written in the unit: its start and end
	// offsets, tags included.
	Bounds() (start, end int)
	isSegment()
}

// Text is words to speak, and the delivery to speak them in.
//
// Start and End are byte offsets into the unit as the LLM wrote it, tags
// included. In "<<calm>> Hi.", the words are Text{" Hi.", ["calm"], 8, 12}.
type Text struct {
	// Text is the words, with the spacing around them.
	Text string
	// Delivery is the delivery the words are spoken in, empty for the voice's
	// baseline.
	Delivery []string
	// Start is where the words start in the unit.
	Start int
	// End is where they end.
	End int
	// Span is the delivery a span such as <<slow: A B 7>> or ^^B^^ adds for
	// just these words, empty for words outside one.
	Span []string
}

// Bounds implements [Segment].
func (t Text) Bounds() (start, end int) { return t.Start, t.End }

func (Text) isSegment() {}

// Sound is a sound made at one point, such as a laugh. In "Hi [[laugh]]", the
// sound is Sound{"laugh", 3, 12}.
type Sound struct {
	// Name is the sound, as written between the brackets.
	Name string
	// Start is where its tag starts in the unit.
	Start int
	// End is where it ends.
	End int
}

// Bounds implements [Segment].
func (s Sound) Bounds() (start, end int) { return s.Start, s.End }

func (Sound) isSegment() {}

// Tag is the sound written as a tag.
func (s Sound) Tag() string { return "[[" + s.Name + "]]" }

// PerformedBy reports whether a service with these capabilities makes the
// sound.
func (s Sound) PerformedBy(c *expressive.ExpressionCapabilities) bool {
	if c == nil {
		return false
	}
	_, ok := c.SoundName(s.Name)
	return ok
}

// Reported is the sound as reported to clients.
func (s Sound) Reported(*expressive.ExpressionCapabilities) ReportedSound {
	return ReportedSound{TextType: frames.TextTypeExpressionSound, Text: s.Name}
}

// Pause is a silence at one point, written [[pause N]] for N milliseconds.
type Pause struct {
	// DurationMS is how long, in milliseconds.
	DurationMS int
	// Start is where its tag starts in the unit.
	Start int
	// End is where it ends.
	End int
}

// Bounds implements [Segment].
func (p Pause) Bounds() (start, end int) { return p.Start, p.End }

func (Pause) isSegment() {}

// Tag is the pause written as a tag.
func (p Pause) Tag() string { return "[[pause " + strconv.Itoa(p.DurationMS) + "]]" }

// PerformedBy reports whether a service with these capabilities pauses.
func (p Pause) PerformedBy(c *expressive.ExpressionCapabilities) bool {
	return c != nil && c.PauseRangeMS != nil
}

// DurationWithin is how long the pause is, in milliseconds, within the
// service's range.
func (p Pause) DurationWithin(c *expressive.ExpressionCapabilities) int {
	return min(max(p.DurationMS, c.PauseRangeMS[0]), c.PauseRangeMS[1])
}

// Reported is the pause as reported to clients: its milliseconds.
func (p Pause) Reported(c *expressive.ExpressionCapabilities) ReportedSound {
	return ReportedSound{TextType: frames.TextTypeExpressionPause, Text: strconv.Itoa(p.DurationWithin(c))}
}

// performable is a sound or a pause: a tag that stands at one point.
type performable interface {
	Segment
	Tag() string
	PerformedBy(c *expressive.ExpressionCapabilities) bool
	Reported(c *expressive.ExpressionCapabilities) ReportedSound
}

// ReportedSound is a sound or pause as reported to clients: its text type,
// frames.TextTypeExpressionSound or frames.TextTypeExpressionPause, and its
// text, the sound's name or the pause's milliseconds.
type ReportedSound struct {
	TextType frames.TextType
	Text     string
}

// ExpressionSpan is what a stretch of a unit performs: words in their delivery,
// or a sound.
//
// [ExpressionFormatter.Format] returns one for each segment the service
// performs, with what it actually performs: a delivery or emphasis the service
// can't perform is left out of Delivery, and a sound it can't perform has no
// span. Offsets are into the unit as the LLM wrote it, so the spoken package
// can find which spans the text of each word the service reports covers.
//
// For "<<calm>> Hi [[laugh]]" on a service that performs both:
//
//	ExpressionSpan{Start: 8, End: 12, Delivery: []string{"calm"}}
//	ExpressionSpan{Start: 12, End: 21, Sound: &ReportedSound{TextTypeExpressionSound, "laugh"}}
type ExpressionSpan struct {
	// Start is where the stretch starts in the unit.
	Start int
	// End is where it ends.
	End int
	// Delivery is the delivery its words are spoken in, empty for the voice's
	// baseline, or nil for a sound.
	Delivery []string
	// Sound is the sound or pause, as reported to clients, or nil for words.
	Sound *ReportedSound
}

// FormattedRequest is a unit's request, as [ExpressionFormatter.Format] builds
// it.
type FormattedRequest struct {
	// Text is the text to send to the service, empty when nothing is left to
	// say.
	Text string
	// Spans has a span for each stretch of the unit the service performs.
	Spans []ExpressionSpan
	// Settings is what the adapter asks to send beside the text, from its
	// RequestSettings.
	Settings map[string]any
}

// ExpressionFormatter reads a TTS service's expression tags and formats its
// requests.
//
// A TTS service has one formatter, which follows a response from unit to unit.
// A delivery set in one sentence carries on into the next, until the next
// delivery tag or [ExpressionFormatter.Reset], which the service calls when the
// response ends or is interrupted. Call [ExpressionFormatter.Parse] on each unit
// in order, then [ExpressionFormatter.Format] to build its request. What the
// service can't perform is left out of the request, with one warning per tag.
//
//	f := formatter.NewExpressionFormatter(func() ttsadapter.ExpressionAdapter { return adapter }, "tts")
//
//	segments := f.Parse("<<warm>> It's [[laugh]] ^^back^^ up.", false)
//	// [Text{" It's ", ["warm"], 8, 14},
//	//  Sound{"laugh", 14, 23},
//	//  Text{" ", ["warm"], 23, 24},
//	//  Text{"back", ["warm", "emphasis"], 26, 30},
//	//  Text{" up.", ["warm"], 32, 36}]
//
//	formatter.ShownText(segments, frames.TextTypeSentence)
//	// "It's back up."
//
//	request := f.Format(segments, frames.TextTypeSentence, transform, trim)
//	// With ElevenLabs' audio tags, request.Text is "[warmly] It's [laughs] BACK up."
//
//	f.Parse("Bye.", false)
//	// [Text{"Bye.", ["warm"], 0, 4}]: still warm
//
// It is not safe for concurrent use: the service calls it from its frame
// goroutine.
type ExpressionFormatter struct {
	adapter func() ttsadapter.ExpressionAdapter
	owner   string
	// activeDelivery is the delivery the next words of the response are
	// spoken in.
	activeDelivery []string
	// formattedDelivery is the delivery of the words last formatted, so markup
	// marks only changes, or nil before any.
	formattedDelivery []string
	unperformed       map[string]struct{}
}

// NewExpressionFormatter builds a formatter. adapter returns the service's
// adapter, which may depend on its settings, such as its model; owner names
// the service in warnings.
func NewExpressionFormatter(adapter func() ttsadapter.ExpressionAdapter, owner string) *ExpressionFormatter {
	return &ExpressionFormatter{adapter: adapter, owner: owner, unperformed: map[string]struct{}{}}
}

// Capabilities is the expression the service can perform now: its adapter's
// capabilities, or nil when it performs none.
func (f *ExpressionFormatter) Capabilities() *expressive.ExpressionCapabilities {
	return f.adapter().Capabilities()
}

// Parse splits a unit of text, with each of its tags whole, into segments, in
// order. Spacing alone between two tags is a [Text] of its own, so the request
// keeps it.
//
// Words take the delivery in effect where they are written, which may have been
// set by an earlier unit of the response:
//
//	f.Parse("<<calm>> All set.", false) // [Text{" All set.", ["calm"], 8, 17}]
//	f.Parse(" Bye.", false)             // [Text{" Bye.", ["calm"], 0, 5}]
//
// Text that stands on its own, such as a TTSSpeakFrame's spoken in the middle
// of a response, is parsed with whole set: it starts in the voice's baseline,
// and the response's delivery carries on after it:
//
//	f.Parse("<<excited>> One moment.", true) // [Text{" One moment.", ["excited"], 11, 23}]
//	f.Parse(" Back.", false)                 // [Text{" Back.", ["calm"], 0, 6}]
func (f *ExpressionFormatter) Parse(unit string, whole bool) []Segment {
	responseDelivery := f.activeDelivery
	if whole {
		f.activeDelivery = nil
	}
	// The names the service declares, and the universal names it performs,
	// keep their kind in either brackets.
	declared := f.Capabilities()
	if declared == nil {
		declared = &expressive.ExpressionCapabilities{}
	}
	sounds := map[string]bool{}
	for _, t := range declared.Sounds {
		sounds[t.Name] = true
	}
	for _, name := range expressive.Sounds() {
		if _, ok := declared.SoundName(string(name)); ok {
			sounds[string(name)] = true
		}
	}
	delivery := map[string]bool{expressive.NeutralDelivery: true}
	for _, t := range declared.Delivery {
		delivery[t.Name] = true
	}
	for _, name := range expressive.Deliveries() {
		if _, ok := declared.DeliveryName(string(name)); ok {
			delivery[string(name)] = true
		}
	}
	var segments []Segment
	position := 0
	for _, m := range tagRe.FindAllStringSubmatchIndex(unit, -1) {
		segments = f.addText(segments, unit, position, m[0])
		switch {
		case m[4] >= 0:
			span := []string{deliveryName(unit[m[2]:m[3]])}
			segments = f.addSpan(segments, span, unit[m[4]:m[5]], m[4], nil, false)
		case m[10] >= 0:
			segments = f.addSpan(segments, []string{expressive.EmphasisDelivery}, unit[m[10]:m[11]], m[10], nil, false)
		default:
			segments = f.addTag(segments, unit, m, sounds, delivery)
		}
		position = m[1]
	}
	segments = f.addText(segments, unit, position, len(unit))
	if whole {
		f.activeDelivery = responseDelivery
	}
	return segments
}

// RestateDelivery has the next request state its delivery from its start.
//
// Markup marks only where the delivery changes, so a request continuing the
// one before it doesn't repeat them. Text spoken in an audio context of its
// own, such as a TTSSpeakFrame's, follows no earlier request there, nor does
// the response's next request follow it, so the service calls this before and
// after it.
func (f *ExpressionFormatter) RestateDelivery() {
	f.formattedDelivery = nil
}

// Reset returns to the baseline, as at the end of a response or an
// interruption.
func (f *ExpressionFormatter) Reset() {
	f.activeDelivery = nil
	f.formattedDelivery = nil
}

// ShownText is the words of a unit, without its tags, spaced as written, with
// one space where a tag stood between two. It is the unit's text as clients
// are shown it:
//
//	segments := f.Parse("<<warm>> It's [[laugh]] ^^back^^ up.", false)
//	formatter.ShownText(segments, frames.TextTypeSentence) // "It's back up."
//
// A tag is removed as a word, with the space before it when punctuation
// follows it: "Hi [[laugh]], there." is shown as "Hi, there.", and so is
// "Hi <<calm>>, there.". A sentence's words are trimmed at both ends; a
// streamed token keeps its spacing.
func ShownText(segments []Segment, textType frames.TextType) string {
	var b strings.Builder
	for _, s := range moveClosingPunctuation(segments) {
		if t, ok := s.(Text); ok {
			b.WriteString(t.Text)
		}
	}
	words := spacesRe.ReplaceAllString(b.String(), " ")
	if textType == frames.TextTypeToken {
		return words
	}
	return strings.TrimSpace(words)
}

// Transform rewrites a run of words for the service, and reports false when
// nothing is left of it.
type Transform func(words string, textType frames.TextType) (string, bool)

// Trim trims a unit's first words as the service trims a unit's text, and
// reports false when nothing is left of them.
type Trim func(words string) (string, bool)

// Format builds the request for a unit: its text, its spans, and the settings
// the adapter asks for the delivery its first words are spoken in.
//
// Each run of words goes through transform first, so text transforms such as
// markdown stripping never see markup, and is then formatted by the adapter in
// the delivery the service can perform. Sounds and pauses are formatted at
// their places, except that punctuation written after one stays with the words
// before it, where the service reports it: "Hi [[laugh]], there." asks
// ElevenLabs for "Hi, [laughs] there.". A sentence's request is trimmed at
// both ends; a streamed token keeps its spacing.
//
// Cartesia names warm and laugh its own way, has no emphasis, and changes
// emotion only between sentences:
//
//	segments := f.Parse("<<warm>> It's [[laugh]] ^^back^^ up.", false)
//	request := f.Format(segments, frames.TextTypeSentence, transform, trim)
//	// text:  `<emotion value="content"/> It's [laughter] back up.`
//	// spans: {8, 14, ["warm"]}, {14, 23, laugh}, {23, 24, ["warm"]},
//	//        {26, 30, ["warm"]} (no emphasis), {32, 36, ["warm"]}
func (f *ExpressionFormatter) Format(
	segments []Segment, textType frames.TextType, transform Transform, trim Trim,
) FormattedRequest {
	r := &request{
		f: f, capabilities: f.Capabilities(), token: textType == frames.TextTypeToken,
		textType: textType, transform: transform, trim: trim,
	}
	if !r.token {
		// A sentence's request states its delivery from its start, since not
		// every service keeps delivery from one request to the next.
		f.formattedDelivery = nil
	}
	if r.capabilities != nil && r.capabilities.DeliveryScope == expressive.DeliveryScopeSentence {
		r.sentenceDelivery, r.sentenceScope = f.sentenceDelivery(segments, r.capabilities)
	}
	for _, segment := range moveClosingPunctuation(segments) {
		if t, isText := segment.(Text); isText {
			r.addWords(t)
		} else if p, ok := segment.(performable); ok {
			r.addSound(p)
		}
	}
	if r.hasLast && !r.token {
		r.parts = append(r.parts, f.adapter().FormatEnd(serviceDelivery(r.lastDelivery, r.capabilities)))
	}
	text := spacesRe.ReplaceAllString(strings.Join(r.parts, ""), " ")
	// A streamed token with no words of its own goes on in the delivery before
	// it.
	settingsDelivery := f.formattedDelivery
	if r.hasFirst {
		settingsDelivery = serviceDelivery(r.firstDelivery, r.capabilities)
	}
	if settingsDelivery == nil {
		settingsDelivery = []string{}
	}
	if !r.token {
		text = strings.TrimSpace(text)
	}
	return FormattedRequest{
		Text:     text,
		Spans:    r.spans,
		Settings: f.adapter().RequestSettings(settingsDelivery),
	}
}

// request is a unit's request as Format builds it, segment by segment.
type request struct {
	f            *ExpressionFormatter
	capabilities *expressive.ExpressionCapabilities
	token        bool
	textType     frames.TextType
	transform    Transform
	trim         Trim
	// sentenceDelivery is the one delivery the request takes, when the service
	// changes delivery only between requests (sentenceScope).
	sentenceDelivery []string
	sentenceScope    bool

	parts         []string
	spans         []ExpressionSpan
	firstDelivery []string
	lastDelivery  []string
	hasFirst      bool
	hasLast       bool
	// trimmed is whether the unit's first words have been trimmed.
	trimmed bool
}

// addSound formats a sound or pause at its place, or leaves it out when the
// service can't perform it.
func (r *request) addSound(p performable) {
	if r.capabilities == nil || !p.PerformedBy(r.capabilities) {
		r.f.warnUnperformed(p.Tag())
		return
	}
	sound := p.Reported(r.capabilities)
	start, end := p.Bounds()
	r.spans = append(r.spans, ExpressionSpan{Start: start, End: end, Sound: &sound})
	r.parts = append(r.parts, r.f.formatSound(p, r.capabilities))
}

// addWords formats a run of words in the delivery the service performs.
func (r *request) addWords(t Text) {
	delivery := r.f.performedDelivery(t.Delivery, r.capabilities)
	if r.sentenceScope {
		// Stress stays on its own words, which a service stresses in its text.
		stressed := slices.Contains(delivery, expressive.EmphasisDelivery)
		delivery = append([]string{}, r.sentenceDelivery...)
		if stressed {
			delivery = append(delivery, expressive.EmphasisDelivery)
		}
	}
	r.spans = append(r.spans, ExpressionSpan{Start: t.Start, End: t.End, Delivery: delivery})
	if strings.TrimSpace(t.Text) == "" {
		// Spacing before the unit's first words is trimmed as they are.
		spacing, ok := t.Text, true
		if !r.trimmed {
			spacing, ok = r.trim(t.Text)
		}
		if ok && spacing != "" {
			r.parts = append(r.parts, spacing)
		}
		return
	}
	if !r.hasFirst {
		r.firstDelivery, r.hasFirst = delivery, true
	}
	words, ok := t.Text, true
	if !r.trimmed {
		// The unit's first words are trimmed as its text would be, and a
		// sentence's even after a sound or pause written before them.
		if !r.token {
			words = strings.TrimLeftFunc(words, unicode.IsSpace)
		}
		words, ok = r.trim(words)
		r.trimmed = true
	}
	if ok {
		words, ok = r.transform(words, r.textType)
	}
	if ok {
		r.parts = append(r.parts, r.f.format(words, serviceDelivery(delivery, r.capabilities)))
		r.lastDelivery, r.hasLast = delivery, true
	}
}

func (f *ExpressionFormatter) addText(segments []Segment, unit string, start, end int) []Segment {
	words := unit[start:end]
	if end == len(unit) {
		if loc := unfinishedRe.FindStringIndex(words); loc != nil {
			slog.Warn("removing an unfinished expression tag", "tag", words[loc[0]:])
			words = words[:loc[0]]
			end = start + len(words)
		}
	}
	words = strings.ReplaceAll(words, unclosedEmphasis, "")
	if words != "" {
		segments = append(segments, Text{Text: words, Delivery: clone(f.activeDelivery), Start: start, End: end})
	}
	return segments
}

// sentenceDelivery is the one delivery a request takes on a service that
// changes delivery between requests, and false when the request has no words.
//
// It is the delivery its words start in, and any span's, which such a service
// can perform only over the whole request: "Your code is <<slow: A B 7>>." is
// spoken slowly. Emphasis is the exception, as stress over a whole sentence
// stresses nothing: it stays on its own words. Clients are told the delivery
// that was requested.
func (f *ExpressionFormatter) sentenceDelivery(
	segments []Segment, capabilities *expressive.ExpressionCapabilities,
) ([]string, bool) {
	var words []Text
	for _, s := range segments {
		if t, ok := s.(Text); ok && strings.TrimSpace(t.Text) != "" {
			words = append(words, t)
		}
	}
	if len(words) == 0 {
		return nil, false
	}
	var delivery []string
	for _, name := range words[0].Delivery {
		if name != expressive.EmphasisDelivery {
			delivery = append(delivery, name)
		}
	}
	for _, segment := range words {
		for _, name := range segment.Span {
			if name != expressive.EmphasisDelivery && !slices.Contains(delivery, name) {
				delivery = append(delivery, name)
			}
		}
	}
	return f.performedDelivery(delivery, capabilities), true
}

// addSpan adds a span's words, in the delivery around it and its own, or in the
// baseline for neutral. around is the delivery around the span, the active
// delivery unless hasAround.
//
// ^^B^^ is a span of emphasis, written short, and can stand within another
// span: <<slow: A ^^B^^ 7>> stresses B on top of slow.
func (f *ExpressionFormatter) addSpan(
	segments []Segment, span []string, words string, start int, around []string, hasAround bool,
) []Segment {
	if !hasAround {
		around = f.activeDelivery
	}
	var delivery []string
	if len(span) == 1 && span[0] == expressive.NeutralDelivery {
		span = nil
	} else {
		delivery = clone(around)
		for _, name := range span {
			if !slices.Contains(around, name) {
				delivery = append(delivery, name)
			}
		}
	}
	position := 0
	for _, m := range emphasisRe.FindAllStringSubmatchIndex(words, -1) {
		if m[0] > position {
			segments = append(segments, Text{
				Text: words[position:m[0]], Delivery: clone(delivery),
				Start: start + position, End: start + m[0], Span: clone(span),
			})
		}
		stressed := append(clone(span), expressive.EmphasisDelivery)
		segments = f.addSpan(segments, stressed, words[m[2]:m[3]], start+m[2], delivery, true)
		position = m[1]
	}
	if position < len(words) {
		segments = append(segments, Text{
			Text: words[position:], Delivery: clone(delivery),
			Start: start + position, End: start + len(words), Span: clone(span),
		})
	}
	return segments
}

func (f *ExpressionFormatter) addTag(
	segments []Segment, unit string, m []int, sounds, delivery map[string]bool,
) []Segment {
	isDeliveryTag := m[6] >= 0
	var content string
	switch {
	case isDeliveryTag:
		content = unit[m[6]:m[7]]
	case m[8] >= 0:
		content = unit[m[8]:m[9]]
	}
	content = strings.TrimSpace(content)
	written := unit[m[0]:m[1]]
	if !isDeliveryTag {
		if pause := pauseRe.FindStringSubmatch(content); pause != nil {
			if ms, err := strconv.Atoi(pause[1]); err == nil {
				return append(segments, Pause{DurationMS: ms, Start: m[0], End: m[1]})
			}
		}
		if delivery[content] && !sounds[content] {
			slog.Warn("reading a delivery written in sound brackets as a delivery", "tag", written)
			f.setDelivery(content)
			return segments
		}
		return append(segments, Sound{Name: content, Start: m[0], End: m[1]})
	}
	if sounds[content] && !delivery[content] {
		slog.Warn("reading a sound written in delivery brackets as a sound", "tag", written)
		return append(segments, Sound{Name: content, Start: m[0], End: m[1]})
	}
	f.setDelivery(content)
	return segments
}

func (f *ExpressionFormatter) setDelivery(content string) {
	name := deliveryName(content)
	if name == "" || name == expressive.NeutralDelivery {
		f.activeDelivery = nil
		return
	}
	f.activeDelivery = []string{name}
}

// format formats words to send to the TTS in their delivery, by the service's
// names.
func (f *ExpressionFormatter) format(words string, delivery []string) string {
	// Leading spacing stays outside the markup, between the words.
	trimmed := strings.TrimLeftFunc(words, unicode.IsSpace)
	formatted := f.adapter().FormatText(trimmed, delivery, f.formattedDelivery)
	f.formattedDelivery = delivery
	return words[:len(words)-len(trimmed)] + formatted
}

// performedDelivery is the delivery the service can perform, warning once
// about each it can't. It is never nil, so an empty delivery reads as the
// baseline rather than as none.
func (f *ExpressionFormatter) performedDelivery(
	delivery []string, capabilities *expressive.ExpressionCapabilities,
) []string {
	performed := []string{}
	for _, name := range delivery {
		var supported bool
		switch {
		case capabilities == nil:
		case name == expressive.EmphasisDelivery:
			supported = capabilities.Emphasis
		default:
			_, supported = capabilities.DeliveryName(name)
		}
		if supported {
			performed = append(performed, name)
			continue
		}
		if name == expressive.EmphasisDelivery {
			f.warnUnperformed("^^emphasis^^")
		} else {
			f.warnUnperformed("<<" + name + ">>")
		}
	}
	return performed
}

// formatSound formats a sound or pause the service can perform.
func (f *ExpressionFormatter) formatSound(p performable, capabilities *expressive.ExpressionCapabilities) string {
	switch s := p.(type) {
	case Pause:
		return f.adapter().FormatPause(s.DurationWithin(capabilities))
	case Sound:
		name, ok := capabilities.SoundName(s.Name)
		if !ok {
			name = s.Name
		}
		return f.adapter().FormatSound(name)
	}
	return ""
}

func (f *ExpressionFormatter) warnUnperformed(tag string) {
	if _, warned := f.unperformed[tag]; warned {
		return
	}
	f.unperformed[tag] = struct{}{}
	slog.Warn("the service can't perform an expression tag, leaving it out of its speech",
		"service", f.owner, "tag", tag)
}

// StripExpressionTags returns text written with expression tags as its words
// alone, spaced as clients are shown them, as by [ShownText]:
//
//	formatter.StripExpressionTags("<<warm>> It's [[laugh]] ^^back^^ up.") // "It's back up."
func StripExpressionTags(unit string) string {
	parser := NewExpressionFormatter(
		func() ttsadapter.ExpressionAdapter { return ttsadapter.BaseExpressionAdapter{} },
		"StripExpressionTags")
	return ShownText(parser.Parse(unit, true), frames.TextTypeSentence)
}

// deliveryName is the delivery a tag names, its words single-spaced.
func deliveryName(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// serviceDelivery is a performed delivery by the names the service performs it
// by.
func serviceDelivery(delivery []string, capabilities *expressive.ExpressionCapabilities) []string {
	if capabilities == nil {
		return delivery
	}
	out := make([]string, 0, len(delivery))
	for _, name := range delivery {
		if name != expressive.EmphasisDelivery {
			if named, ok := capabilities.DeliveryName(name); ok {
				name = named
			}
		}
		out = append(out, name)
	}
	return out
}

// closingPunctuationLen is how many bytes of punctuation open s that close the
// words before a tag written ahead of it, as in "Hi [[laugh]], there.": what
// ends a sentence, and what ends a clause.
func closingPunctuationLen(s string) int {
	n := 0
	for n < len(s) {
		r, size := utf8.DecodeRuneInString(s[n:])
		if !text.IsSentenceEnding(r) && !strings.ContainsRune(",:)，、：）،", r) {
			break
		}
		n += size
	}
	return n
}

// moveClosingPunctuation moves punctuation written after a tag onto the words
// before it.
//
// A tag is removed as a word, so in "Hi [[laugh]], there." the comma closes
// "Hi": the words become "Hi," and " there.", with the sound between them,
// where a service reports the comma. Words follow a tag when a sound or pause
// stands between them, or the gap a delivery tag left.
func moveClosingPunctuation(segments []Segment) []Segment {
	moved := make([]Segment, 0, len(segments))
	wordsAt := -1
	tagSinceWords := false
	end := -1
	for _, segment := range segments {
		t, ok := segment.(Text)
		if !ok {
			tagSinceWords = true
			moved = append(moved, segment)
			continue
		}
		if end >= 0 && t.Start > end {
			tagSinceWords = true
		}
		end = t.End
		if strings.TrimSpace(t.Text) != "" {
			if n := closingPunctuationLen(t.Text); n > 0 && wordsAt >= 0 && tagSinceWords {
				before, _ := moved[wordsAt].(Text)
				before.Text = strings.TrimRightFunc(before.Text, unicode.IsSpace) + t.Text[:n]
				moved[wordsAt] = before
				t.Text = t.Text[n:]
			}
			tagSinceWords = false
			if strings.TrimSpace(t.Text) != "" {
				wordsAt = len(moved)
			}
		}
		moved = append(moved, t)
	}
	return moved
}

func clone(s []string) []string {
	if s == nil {
		return nil
	}
	return append([]string{}, s...)
}
