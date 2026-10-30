package formatter_test

import (
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttscartesia "github.com/gojargo/jargo/adapter/tts/cartesia"
	ttselevenlabs "github.com/gojargo/jargo/adapter/tts/elevenlabs"
	ttsminimax "github.com/gojargo/jargo/adapter/tts/minimax"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/expressive/formatter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/text"
)

// declaredAdapter declares sounds and delivery, and formats delivery as
// <delivery> where it changes.
type declaredAdapter struct {
	ttsadapter.BaseExpressionAdapter
	sounds   []string
	delivery []string
	scope    expressive.DeliveryScope
	closing  bool
}

func (a *declaredAdapter) Capabilities() *expressive.ExpressionCapabilities {
	c := &expressive.ExpressionCapabilities{
		PauseRangeMS:  &[2]int{100, 1000},
		Emphasis:      true,
		DeliveryScope: a.scope,
	}
	for _, name := range a.sounds {
		c.Sounds = append(c.Sounds, expressive.ExpressionTerm{Name: name})
	}
	for _, name := range a.delivery {
		c.Delivery = append(c.Delivery, expressive.ExpressionTerm{Name: name})
	}
	return c
}

func (a *declaredAdapter) FormatText(words string, delivery, previous []string) string {
	if previous == nil {
		previous = []string{}
	}
	if !slices.Equal(delivery, previous) && len(delivery) > 0 {
		return "<" + strings.Join(delivery, ",") + "> " + words
	}
	return words
}

func (a *declaredAdapter) FormatSound(name string) string { return " <" + name + "> " }

// FormatEnd closes each request with the delivery its last words are spoken
// in, for the closing adapter.
func (a *declaredAdapter) FormatEnd(delivery []string) string {
	if !a.closing {
		return ""
	}
	return " </" + strings.Join(delivery, ",") + ">"
}

func newFormatter(a *declaredAdapter) *formatter.ExpressionFormatter {
	return formatter.NewExpressionFormatter(func() ttsadapter.ExpressionAdapter { return a }, "tts")
}

// shape is a segment compared without its position.
type shape struct {
	kind     string
	text     string
	delivery string
}

func words(t, delivery string) shape { return shape{"text", t, delivery} }

func shapeOf(s formatter.Segment) shape {
	switch seg := s.(type) {
	case formatter.Text:
		return words(seg.Text, strings.Join(seg.Delivery, ","))
	case formatter.Sound:
		return shape{kind: "sound", text: seg.Name}
	case formatter.Pause:
		return shape{kind: "pause", text: strconv.Itoa(seg.DurationMS)}
	}
	return shape{}
}

func shapes(segments []formatter.Segment) []shape {
	out := make([]shape, 0, len(segments))
	for _, s := range segments {
		out = append(out, shapeOf(s))
	}
	return out
}

func parse(t *testing.T, unit string, a *declaredAdapter) []shape {
	t.Helper()
	if a == nil {
		a = &declaredAdapter{}
	}
	return shapes(newFormatter(a).Parse(unit, false))
}

func checkShapes(t *testing.T, got, want []shape) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("segments\n got: %v\nwant: %v", got, want)
	}
}

const reply = "<<apologetic>> I'm so sorry about the outage. <<enunciate>> Your ticket is 2, 3, 8, 7. " +
	"[[pause 600]] <<cheerful>> The good news is it's ^^back up^^ now. [[laugh]] Finally!"

func TestAReplySplitsIntoWordsWithTheirDeliveryAndSounds(t *testing.T) {
	checkShapes(t, parse(t, reply, nil), []shape{
		words(" I'm so sorry about the outage. ", "apologetic"),
		words(" Your ticket is 2, 3, 8, 7. ", "enunciate"),
		{kind: "pause", text: "600"},
		words(" ", "enunciate"),
		words(" The good news is it's ", "cheerful"),
		words("back up", "cheerful,emphasis"),
		words(" now. ", "cheerful"),
		{kind: "sound", text: "laugh"},
		words(" Finally!", "cheerful"),
	})
}

func TestEachSegmentSaysWhereItWasWritten(t *testing.T) {
	for _, segment := range newFormatter(&declaredAdapter{}).Parse(reply, false) {
		start, end := segment.Bounds()
		written := reply[start:end]
		if w, ok := segment.(formatter.Text); ok {
			if written != w.Text {
				t.Fatalf("%q written as %q", w.Text, written)
			}
		} else if !strings.HasPrefix(written, "[[") || !strings.HasSuffix(written, "]]") {
			t.Fatalf("tag written as %q", written)
		}
	}
}

func TestTextWithoutTagsIsInTheBaseline(t *testing.T) {
	checkShapes(t, parse(t, "Hello there. How are you?", nil), []shape{words("Hello there. How are you?", "")})
}

func TestADeliveryNameMayBeSeveralWords(t *testing.T) {
	checkShapes(t, parse(t, "<<soft  tone>>Hi", nil), []shape{words("Hi", "soft tone")})
}

func TestADeliveryLastsUntilTheNextDeliveryTagAcrossUnits(t *testing.T) {
	p := newFormatter(&declaredAdapter{})

	checkShapes(t, shapes(p.Parse("<<calm>>One. ", false)), []shape{words("One. ", "calm")})
	checkShapes(t, shapes(p.Parse("Two. <<warm>>Three.", false)), []shape{
		words("Two. ", "calm"),
		words("Three.", "warm"),
	})
}

func TestNeutralReturnsToTheBaseline(t *testing.T) {
	checkShapes(t, parse(t, "<<calm>>One. <<neutral>>Two.", nil), []shape{words("One. ", "calm"), words("Two.", "")})
}

func TestTheDeliveryEndsWithTheResponse(t *testing.T) {
	p := newFormatter(&declaredAdapter{})
	p.Parse("<<calm>>One.", false)
	p.Reset()

	checkShapes(t, shapes(p.Parse("Two", false)), []shape{words("Two", "")})
}

func TestTextThatStandsOnItsOwnLeavesTheResponsesDelivery(t *testing.T) {
	p := newFormatter(&declaredAdapter{})
	p.Parse("<<calm>>One. ", false)

	checkShapes(t, shapes(p.Parse("Hello.", true)), []shape{words("Hello.", "")})
	checkShapes(t, shapes(p.Parse("Two.", false)), []shape{words("Two.", "calm")})
}

func TestADeliveryTagEndingTheTextSaysNothing(t *testing.T) {
	checkShapes(t, parse(t, "Bye! <<warm>>", nil), []shape{words("Bye! ", "")})
}

func TestAPauseIsADurationInMilliseconds(t *testing.T) {
	checkShapes(t, parse(t, "[[pause 1500]]", nil), []shape{{kind: "pause", text: "1500"}})
}

func TestAnythingElseBetweenTheBracketsIsPassedOnAsWritten(t *testing.T) {
	for _, tt := range []struct {
		text string
		want shape
	}{
		{"[[pause]]x", shape{kind: "sound", text: "pause"}},
		{"[[pause 800ms]]x", shape{kind: "sound", text: "pause 800ms"}},
		{"[[clear throat]]x", shape{kind: "sound", text: "clear throat"}},
		{"<<pause 500>>x", words("x", "pause 500")},
		{"<</cheerful>>x", words("x", "/cheerful")},
	} {
		if got := parse(t, tt.text, nil)[0]; got != tt.want {
			t.Errorf("%q: first segment %v, want %v", tt.text, got, tt.want)
		}
	}
}

func TestADeclaredDeliveryInSoundBracketsSetsTheDelivery(t *testing.T) {
	checkShapes(t, parse(t, "[[cheerful]] Hi", &declaredAdapter{delivery: []string{"cheerful"}}),
		[]shape{words(" Hi", "cheerful")})
}

func TestADeclaredSoundInDeliveryBracketsStaysASound(t *testing.T) {
	checkShapes(t, parse(t, "<<laugh>> Hi", &declaredAdapter{sounds: []string{"laugh"}}),
		[]shape{{kind: "sound", text: "laugh"}, words(" Hi", "")})
}

func TestTextThatOnlyLooksLikeATagIsSpoken(t *testing.T) {
	for _, unit := range []string{
		"He said [sic].",
		"Hi [cheerful] there.",
		"a < b",
		"a <b> c",
		"Shift it left: a << b.",
		"Brackets [[like this.]] stay text.",
	} {
		checkShapes(t, parse(t, unit, &declaredAdapter{delivery: []string{"cheerful"}}), []shape{words(unit, "")})
	}
}

func TestASpanAddsItsDeliveryToJustItsWords(t *testing.T) {
	checkShapes(t, parse(t, "<<calm>> Your code is <<slow: A B 7>>. Bye.", nil), []shape{
		words(" Your code is ", "calm"),
		words("A B 7", "calm,slow"),
		words(". Bye.", "calm"),
	})
}

func TestANeutralSpanSpeaksItsWordsInTheBaseline(t *testing.T) {
	checkShapes(t, parse(t, "<<calm>> Say <<neutral: hi there>>, okay.", nil), []shape{
		words(" Say ", "calm"),
		words("hi there", ""),
		words(", okay.", "calm"),
	})
}

func TestEmphasisInASpanIsAddedToItsDelivery(t *testing.T) {
	checkShapes(t, parse(t, "It's <<slow: A ^^B^^ 7>>.", nil), []shape{
		words("It's ", ""),
		words("A ", "slow"),
		words("B", "slow,emphasis"),
		words(" 7", "slow"),
		words(".", ""),
	})
}

func TestEmphasisIsASpanWrittenShort(t *testing.T) {
	checkShapes(t, parse(t, "Now ^^that's^^ it.", nil), parse(t, "Now <<emphasis: that's>> it.", nil))
}

func TestEachSpanSegmentSaysWhereItWasWritten(t *testing.T) {
	unit := "<<calm>> Your code is <<slow: A ^^B^^ 7>>, okay."
	for _, segment := range newFormatter(&declaredAdapter{}).Parse(unit, false) {
		w, _ := segment.(formatter.Text)
		if unit[w.Start:w.End] != w.Text {
			t.Fatalf("%q written as %q", w.Text, unit[w.Start:w.End])
		}
	}
}

func TestASpanLeftUnfinishedIsRemoved(t *testing.T) {
	checkShapes(t, parse(t, "Your code is <<slow: A B", nil), []shape{words("Your code is ", "")})
}

func TestWordTrackingPassesOverASpansMarkupButNotItsWords(t *testing.T) {
	masked := formatter.ExpressionTagPattern.ReplaceAllString(
		"Code <<slow: A B>>, ok. <<calm>> [[laugh]] Bye.", "#")

	if masked != "Code #A B#, ok. # # Bye." {
		t.Fatalf("masked = %q", masked)
	}
}

func TestATagLeftUnfinishedIsRemoved(t *testing.T) {
	checkShapes(t, parse(t, "Thanks for calling. <<chee", nil), []shape{words("Thanks for calling. ", "")})
	checkShapes(t, parse(t, "Thanks for calling. [[", nil), []shape{words("Thanks for calling. ", "")})
}

func TestEmphasisIsAddedToTheActiveDelivery(t *testing.T) {
	checkShapes(t, parse(t, "<<calm>>It's ^^fixed^^ now.", nil), []shape{
		words("It's ", "calm"),
		words("fixed", "calm,emphasis"),
		words(" now.", "calm"),
	})
}

func TestEmphasis(t *testing.T) {
	for _, tt := range []struct{ text, stressed string }{
		{"It's ^^back up^^.", "back up"},
		{"The code is ^^8492^^.", "8492"},
		{"^^Wow^^, nice.", "Wow"},
	} {
		if !slices.Contains(parse(t, tt.text, nil), words(tt.stressed, "emphasis")) {
			t.Errorf("%q: %q not stressed", tt.text, tt.stressed)
		}
	}
}

func TestSingleCaretsAndAsterisksAreText(t *testing.T) {
	for _, unit := range []string{
		"Two to the tenth is 2^10.",
		"a^b^c",
		"Thanks ^_^ bye",
		"It's ^fixed^ and *done*.",
	} {
		checkShapes(t, parse(t, unit, nil), []shape{words(unit, "")})
	}
}

func TestEmphasisThatNeverClosesIsSpokenPlainly(t *testing.T) {
	checkShapes(t, parse(t, "It's ^^back up", nil), []shape{words("It's back up", "")})
}

func TestAnAggregatorKeepsEachTagInOneUnit(t *testing.T) {
	for _, textType := range []frames.TextType{frames.TextTypeToken, frames.TextTypeSentence} {
		aggregator := text.NewSkipTagsAggregator(textType, "", formatter.ExpressionTags())
		var units []string
		for _, chunk := range []string{"<<chee", "rful>> It's [[la", "ugh]] ^^back", " up^", "^ now."} {
			for _, a := range aggregator.Aggregate(chunk) {
				units = append(units, a.Text)
			}
		}
		if flushed, ok := aggregator.Flush(); ok {
			units = append(units, flushed.Text)
		}

		p := newFormatter(&declaredAdapter{})
		var segments []shape
		for _, unit := range units {
			segments = append(segments, shapes(p.Parse(unit, false))...)
		}
		if !slices.Contains(segments, shape{kind: "sound", text: "laugh"}) {
			t.Errorf("%s: no laugh in %v", textType, segments)
		}
		if !slices.Contains(segments, words("back up", "cheerful,emphasis")) {
			t.Errorf("%s: no stressed words in %v", textType, segments)
		}
	}
}

func TestNamesDeclaredAfterTheFormatterIsMadeAreKnown(t *testing.T) {
	a := &declaredAdapter{}
	p := newFormatter(a)
	a.sounds = []string{"laugh"}

	// A declared sound keeps its kind in delivery brackets.
	segments := p.Parse("<<laugh>> Hi.", false)

	if s, ok := segments[0].(formatter.Sound); !ok || s.Name != "laugh" {
		t.Fatalf("first segment = %#v", segments[0])
	}
	if _, ok := segments[1].(formatter.Text); !ok {
		t.Fatalf("second segment = %#v", segments[1])
	}
}

func TestAUnitsWordsAreShownWithoutItsTags(t *testing.T) {
	segments := newFormatter(&declaredAdapter{}).Parse("<<calm>> It's [[laugh]] fine. [[pause 400]]", false)

	if got := formatter.ShownText(segments, frames.TextTypeSentence); got != "It's fine." {
		t.Fatalf("sentence = %q", got)
	}
	if got := formatter.ShownText(segments, frames.TextTypeToken); got != " It's fine. " {
		t.Fatalf("token = %q", got)
	}
}

func same(words string, _ frames.TextType) (string, bool) { return words, true }

func asIs(words string) (string, bool) { return words, true }

func sound(textType frames.TextType, name string) *formatter.ReportedSound {
	return &formatter.ReportedSound{TextType: textType, Text: name}
}

func TestARequestFormatsWhatTheServicePerformsAndSaysWhere(t *testing.T) {
	p := newFormatter(&declaredAdapter{sounds: []string{"laugh"}, delivery: []string{"calm"}})
	unit := "<<calm>> Hi. [[laugh]] [[pause 50]]"

	formatted := p.Format(p.Parse(unit, false), frames.TextTypeSentence, same, asIs)

	if formatted.Text != "<calm> Hi. <laugh>" {
		t.Fatalf("request = %q", formatted.Text)
	}
	want := []formatter.ExpressionSpan{
		{Start: 8, End: 13, Delivery: []string{"calm"}},
		{Start: 13, End: 22, Sound: sound(frames.TextTypeExpressionSound, "laugh")},
		{Start: 22, End: 23, Delivery: []string{"calm"}},
		{Start: 23, End: 35, Sound: sound(frames.TextTypeExpressionPause, "100")},
	}
	if !reflect.DeepEqual(formatted.Spans, want) {
		t.Fatalf("spans = %+v", formatted.Spans)
	}
}

func TestPunctuationAfterATagClosesTheWordsBeforeIt(t *testing.T) {
	for _, tt := range []struct{ text, shown, sent string }{
		{"Hi [[laugh]], there.", "Hi, there.", "Hi, <laugh> there."},
		{"It's ^^back^^ [[laugh]]!", "It's back!", "It's <emphasis> back! <laugh>"},
		{"Hi[[laugh]], there.", "Hi, there.", "Hi, <laugh> there."},
		{"Well [[laugh]]... okay.", "Well... okay.", "Well... <laugh> okay."},
		{"Really [[laugh]]？ Yes.", "Really？ Yes.", "Really？ <laugh> Yes."},
		{"مرحبا [[laugh]]؟", "مرحبا؟", "مرحبا؟ <laugh>"},
		// A sound the service can't perform is removed the same way.
		{"Hi [[sigh]], there.", "Hi, there.", "Hi, there."},
		// Spacing the LLM wrote before the punctuation is kept.
		{"Hi [[laugh]] , there.", "Hi , there.", "Hi <laugh> , there."},
		{"It's [[laugh]] back.", "It's back.", "It's <laugh> back."},
		// So does punctuation after a span or a delivery tag.
		{"It's <<slow: A B 7>>, okay.", "It's A B 7, okay.", "It's <slow> A B 7, okay."},
		{"Hi <<calm>>, there.", "Hi, there.", "Hi, <calm> there."},
	} {
		p := newFormatter(&declaredAdapter{sounds: []string{"laugh"}, delivery: []string{"calm", "slow"}})
		segments := p.Parse(tt.text, false)

		if got := formatter.ShownText(segments, frames.TextTypeSentence); got != tt.shown {
			t.Errorf("%q shown as %q, want %q", tt.text, got, tt.shown)
		}
		if got := p.Format(segments, frames.TextTypeSentence, same, asIs).Text; got != tt.sent {
			t.Errorf("%q sent as %q, want %q", tt.text, got, tt.sent)
		}
	}
}

func TestASentenceRequestIsClosedByTheAdapter(t *testing.T) {
	for _, tt := range []struct {
		textType frames.TextType
		endsWith string
	}{{frames.TextTypeSentence, "</calm>"}, {frames.TextTypeToken, " up."}} {
		p := newFormatter(&declaredAdapter{delivery: []string{"calm"}, closing: true})

		request := p.Format(p.Parse("<<calm>> It's back up.", false), tt.textType, same, asIs).Text

		// A streamed token is followed by more of the same request, so isn't
		// closed.
		if !strings.HasSuffix(request, tt.endsWith) {
			t.Errorf("%s request %q does not end with %q", tt.textType, request, tt.endsWith)
		}
	}
}

func TestARequestLeavesOutWhatTheServiceCantPerform(t *testing.T) {
	p := newFormatter(&declaredAdapter{})
	segments := p.Parse("<<calm>> Hi. [[laugh]]", true)

	formatted := p.Format(segments, frames.TextTypeSentence, same, asIs)

	if formatted.Text != "Hi." {
		t.Fatalf("request = %q", formatted.Text)
	}
	want := []formatter.ExpressionSpan{{Start: 8, End: 13, Delivery: []string{}}}
	if !reflect.DeepEqual(formatted.Spans, want) {
		t.Fatalf("spans = %+v", formatted.Spans)
	}
}

func TestASpanSetsTheDeliveryOfItsWholeRequestOnASentenceService(t *testing.T) {
	p := newFormatter(&declaredAdapter{delivery: []string{"calm", "slow"}, scope: expressive.DeliveryScopeSentence})
	segments := p.Parse("<<calm>> Your code is <<slow: A B 7>>, okay.", false)

	formatted := p.Format(segments, frames.TextTypeSentence, same, asIs)

	if formatted.Text != "<calm,slow> Your code is A B 7, okay." {
		t.Fatalf("request = %q", formatted.Text)
	}
	for _, span := range formatted.Spans {
		if !slices.Equal(span.Delivery, []string{"calm", "slow"}) {
			t.Fatalf("span delivery = %v", span.Delivery)
		}
	}
}

func spanDeliveries(spans []formatter.ExpressionSpan) [][]string {
	out := make([][]string, 0, len(spans))
	for _, s := range spans {
		out = append(out, s.Delivery)
	}
	return out
}

func TestEmphasisStaysOnItsWordsOnASentenceService(t *testing.T) {
	p := newFormatter(&declaredAdapter{delivery: []string{"calm"}, scope: expressive.DeliveryScopeSentence})
	segments := p.Parse("<<calm>> It's ^^back^^ now.", false)

	formatted := p.Format(segments, frames.TextTypeSentence, same, asIs)

	if formatted.Text != "<calm> It's <calm,emphasis> back <calm> now." {
		t.Fatalf("request = %q", formatted.Text)
	}
	want := [][]string{{"calm"}, {"calm", "emphasis"}, {"calm"}}
	if got := spanDeliveries(formatted.Spans); !reflect.DeepEqual(got, want) {
		t.Fatalf("deliveries = %v", got)
	}
}

func TestStressedFirstWordsDontStressAWholeSentenceOnASentenceService(t *testing.T) {
	p := newFormatter(&declaredAdapter{delivery: []string{"excited"}, scope: expressive.DeliveryScopeSentence})
	segments := p.Parse("<<excited>> ^^Really^^ good news!", false)

	formatted := p.Format(segments, frames.TextTypeSentence, same, asIs)

	if formatted.Text != "<excited,emphasis> Really <excited> good news!" {
		t.Fatalf("request = %q", formatted.Text)
	}
	want := [][]string{{"excited"}, {"excited", "emphasis"}, {"excited"}}
	if got := spanDeliveries(formatted.Spans); !reflect.DeepEqual(got, want) {
		t.Fatalf("deliveries = %v", got)
	}
}

func TestTagsAreStrippedToTheWordsClientsAreShown(t *testing.T) {
	if got := formatter.StripExpressionTags("<<cheerful>> It's [[laughs]] ^^back^^ up."); got != "It's back up." {
		t.Fatalf("stripped = %q", got)
	}
	if got := formatter.StripExpressionTags("Hi [[laughs]], <<calm: there>>."); got != "Hi, there." {
		t.Fatalf("stripped = %q", got)
	}
}

func formattedWith(a ttsadapter.ExpressionAdapter, unit string) formatter.FormattedRequest {
	p := formatter.NewExpressionFormatter(func() ttsadapter.ExpressionAdapter { return a }, "tts")
	return p.Format(p.Parse(unit, false), frames.TextTypeSentence, same, asIs)
}

func TestAUniversalDeliveryIsSentByTheServicesName(t *testing.T) {
	formatted := formattedWith(ttselevenlabs.ElevenLabsAudioTagExpressionAdapter{}, "<<warm>> Hi there.")

	if formatted.Text != "[warmly] Hi there." {
		t.Fatalf("request = %q", formatted.Text)
	}
	want := []formatter.ExpressionSpan{{Start: 8, End: 18, Delivery: []string{"warm"}}}
	if !reflect.DeepEqual(formatted.Spans, want) {
		t.Fatalf("spans = %+v", formatted.Spans)
	}
}

func TestAUniversalSoundIsSentByTheServicesName(t *testing.T) {
	formatted := formattedWith(ttscartesia.CartesiaExpressionAdapter{}, "Ha [[laugh]] okay.")

	if !strings.Contains(formatted.Text, "[laughter]") {
		t.Fatalf("request = %q", formatted.Text)
	}
	want := formatter.ExpressionSpan{Start: 3, End: 12, Sound: sound(frames.TextTypeExpressionSound, "laugh")}
	if !slices.ContainsFunc(formatted.Spans, func(s formatter.ExpressionSpan) bool { return reflect.DeepEqual(s, want) }) {
		t.Fatalf("spans = %+v", formatted.Spans)
	}
}

func TestADeliverySentAsASettingIsNamedAsTheServiceNamesIt(t *testing.T) {
	formatted := formattedWith(ttsminimax.MiniMaxExpressionAdapter{}, "<<warm>> Hi there.")

	if formatted.Settings["emotion"] != "happy" {
		t.Fatalf("settings = %v", formatted.Settings)
	}
	if !slices.Equal(formatted.Spans[0].Delivery, []string{"warm"}) {
		t.Fatalf("delivery = %v", formatted.Spans[0].Delivery)
	}
}

func TestANameOfTheServicesOwnIsSentAsWritten(t *testing.T) {
	formatted := formattedWith(ttselevenlabs.ElevenLabsAudioTagExpressionAdapter{}, "<<mischievously>> Hi there.")

	if formatted.Text != "[mischievously] Hi there." {
		t.Fatalf("request = %q", formatted.Text)
	}
	if !slices.Equal(formatted.Spans[0].Delivery, []string{"mischievously"}) {
		t.Fatalf("delivery = %v", formatted.Spans[0].Delivery)
	}
}

func TestAUniversalDeliveryAFixedListLacksIsLeftOut(t *testing.T) {
	formatted := formattedWith(ttsminimax.MiniMaxExpressionAdapter{}, "<<grateful>> Thanks so much.")

	if _, ok := formatted.Settings["emotion"]; ok {
		t.Fatalf("settings = %v", formatted.Settings)
	}
	if len(formatted.Spans[0].Delivery) != 0 {
		t.Fatalf("delivery = %v", formatted.Spans[0].Delivery)
	}
}

func TestAUniversalNameIsPerformedByTheFirstNameTheServicePerforms(t *testing.T) {
	c := ttscartesia.CartesiaExpressionAdapter{}.Capabilities()

	for _, tt := range []struct{ name, want string }{
		{"warm", "content"}, {"calm", "calm"}, {"grateful", "grateful"},
	} {
		if got, _ := c.DeliveryName(tt.name); got != tt.want {
			t.Errorf("DeliveryName(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
	if got, _ := c.SoundName("chuckle"); got != "laughter" {
		t.Errorf("SoundName(chuckle) = %q", got)
	}
	if got, ok := c.DeliveryName("mischievously"); ok {
		t.Errorf("DeliveryName(mischievously) = %q, want none", got)
	}
}

func TestAUniversalNameKeepsItsKindInTheOtherBrackets(t *testing.T) {
	p := formatter.NewExpressionFormatter(func() ttsadapter.ExpressionAdapter {
		return ttselevenlabs.ElevenLabsAudioTagExpressionAdapter{}
	}, "tts")

	checkShapes(t, shapes(p.Parse("[[warm]] Hi. <<laugh>>", false)), []shape{
		words(" Hi. ", "warm"),
		{kind: "sound", text: "laugh"},
	})
}
