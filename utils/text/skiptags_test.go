package text_test

import (
	"slices"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/text"
)

// spellTag is the tag pair most of these tests aggregate around.
func spellTag() []text.StartEndTags {
	return []text.StartEndTags{{Start: "<spell>", End: "</spell>"}}
}

func newSkipTags(
	t *testing.T, by frames.AggregationType, tags []text.StartEndTags,
) *text.SkipTagsAggregator {
	t.Helper()
	tok, err := text.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	return text.NewSkipTagsAggregator(by, tok, tags)
}

// texts is the aggregated text of each unit, for comparing a run of them.
func texts(aggs []text.Aggregation) []string {
	out := make([]string, 0, len(aggs))
	for _, a := range aggs {
		out = append(out, a.Text)
	}
	return out
}

func equalStrings(a, b []string) bool { return slices.Equal(a, b) }

// With no tag in the text the aggregator behaves as the simple one does, right
// down to waiting for the lookahead that confirms the final boundary.
func TestSkipTagsWithoutTags(t *testing.T) {
	a := newSkipTags(t, frames.AggregationSentence, spellTag())

	if got := a.Aggregate("Hello there!"); len(got) != 0 {
		t.Fatalf("a trailing mark should wait for lookahead, got %+v", got)
	}
	rest, ok := a.Flush()
	if !ok || rest.Text != "Hello there!" {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, "Hello there!")
	}
	if rest.Type != frames.AggregationSentence {
		t.Fatalf("type = %q, want %q", rest.Type, frames.AggregationSentence)
	}
	if got := a.Text().Text; got != "" {
		t.Fatalf("buffer after flush = %q, want empty", got)
	}
}

// The periods inside a tagged run end no sentence, so the whole line stays in
// one piece.
func TestSkipTagsKeepsATaggedRunWhole(t *testing.T) {
	a := newSkipTags(t, frames.AggregationSentence, spellTag())

	const line = "My address is <spell>foo.bar@example.com</spell>."
	if got := a.Aggregate(line); len(got) != 0 {
		t.Fatalf("nothing should complete inside a tag, got %+v", got)
	}
	if got := a.Text().Text; got != line {
		t.Fatalf("buffer = %q, want the whole line", got)
	}
	rest, ok := a.Flush()
	if !ok || rest.Text != line {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, line)
	}
}

// A tag marks text rather than replacing it, so a run whose closing tag never
// arrived is still spoken. The tag state resets with the buffer, so the next
// response is not stuck inside a tag that never closed.
func TestSkipTagsFlushesAnUnclosedTag(t *testing.T) {
	a := newSkipTags(t, frames.AggregationSentence, spellTag())

	const line = "My address is <spell>foo@example.com"
	if got := a.Aggregate(line); len(got) != 0 {
		t.Fatalf("nothing should complete inside a tag, got %+v", got)
	}
	rest, ok := a.Flush()
	if !ok || rest.Text != line {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, line)
	}

	got := a.Aggregate("Hi there. Next")
	if len(got) != 1 || got[0].Text != "Hi there." {
		t.Fatalf("aggregations = %+v, want the aggregator out of the tag", got)
	}
}

// Aggregating by token passes text straight through when no tag is open.
func TestSkipTagsTokenPassesTextThrough(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	got := a.Aggregate("Hello!")
	if len(got) != 1 || got[0].Text != "Hello!" || got[0].Type != frames.AggregationToken {
		t.Fatalf("aggregations = %+v, want one token %q", got, "Hello!")
	}
}

// Inside a tag the text is held until the closing tag arrives, so the tag is
// never handed over in pieces.
func TestSkipTagsTokenHoldsATaggedRun(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	if got := a.Aggregate("<spell>foo@bar"); len(got) != 0 {
		t.Fatalf("an open tag should hold its text, got %+v", got)
	}
	got := a.Aggregate("</spell>")
	if len(got) != 1 || got[0].Text != "<spell>foo@bar</spell>" {
		t.Fatalf("aggregations = %+v, want the tagged run whole", got)
	}
}

// Flushing by token returns what is held behind an unclosed tag rather than
// dropping it, and leaves the aggregator ready for the next response.
func TestSkipTagsTokenFlushesAnUnclosedTag(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	a.Aggregate("<spell>unclosed")
	rest, ok := a.Flush()
	if !ok || rest.Text != "<spell>unclosed" || rest.Type != frames.AggregationToken {
		t.Fatalf("flush = %+v (%v), want the held text back", rest, ok)
	}
	if got := a.Text().Text; got != "" {
		t.Fatalf("buffer after flush = %q, want empty", got)
	}
	got := a.Aggregate("more text")
	if len(got) != 1 || got[0].Text != "more text" {
		t.Fatalf("aggregations = %+v, want the aggregator reusable", got)
	}
}

// Nothing is buffered by token once the text has passed through, so there is
// nothing left to flush.
func TestSkipTagsTokenFlushesNothingWhenEmpty(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	a.Aggregate("Hello!")
	if rest, ok := a.Flush(); ok {
		t.Fatalf("flush = %+v, want nothing left", rest)
	}
}

// Tokens arriving one at a time: the text either side of the tag passes
// straight through, and the tagged run is held until it closes.
func TestSkipTagsTokenAroundATag(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	var got []text.Aggregation
	for _, tok := range []string{"Hi ", "<spell>", "X", "</spell>", " bye"} {
		got = append(got, a.Aggregate(tok)...)
	}
	want := []string{"Hi ", "<spell>X</spell>", " bye"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}

// A start tag split across chunks must not leak as ordinary text. The piece
// that could open a tag is held until the next chunk settles it.
func TestSkipTagsTokenReassemblesASplitStartTag(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	var got []text.Aggregation
	for _, tok := range []string{"Call ", "<spe", "ll>a b c</spell>", " now"} {
		got = append(got, a.Aggregate(tok)...)
	}
	want := []string{"Call ", "<spell>a b c</spell>", " now"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}

// The same, split three ways: nothing leaks while the tag reassembles.
func TestSkipTagsTokenReassemblesAcrossThreeChunks(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	var got []text.Aggregation
	for _, tok := range []string{"<sp", "el", "l>abc</spell>"} {
		got = append(got, a.Aggregate(tok)...)
	}
	want := []string{"<spell>abc</spell>"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}

// A chunk that is nothing but the start of a tag yields nothing and stays
// buffered.
func TestSkipTagsTokenHoldsAPartialTagChunk(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	if got := a.Aggregate("<spe"); len(got) != 0 {
		t.Fatalf("a partial tag should be held, got %+v", got)
	}
	if got := a.Text().Text; got != "<spe" {
		t.Fatalf("buffer = %q, want the partial tag held", got)
	}
}

// A trailing character that could open a tag is held, then released once the
// next chunk shows it opens nothing.
func TestSkipTagsTokenReleasesAHeldPrefix(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	got := a.Aggregate("Hi <")
	if len(got) != 1 || got[0].Text != "Hi " {
		t.Fatalf("aggregations = %+v, want the trailing %q held back", got, "<")
	}
	got = a.Aggregate(" there")
	if len(got) != 1 || got[0].Text != "< there" {
		t.Fatalf("aggregations = %+v, want the held character released", got)
	}
}

// A chunk carrying both text to release and a trailing partial tag keeps the
// tag working: what is held back is scanned again once it is complete.
func TestSkipTagsTokenRescansWhatItHeldBack(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	var got []text.Aggregation
	for _, tok := range []string{"Call <spe", "ll>a b", " c</sp", "ell> now"} {
		got = append(got, a.Aggregate(tok)...)
	}
	want := []string{"Call ", "<spell>a b c</spell> now"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}

// With more than one pair registered, a split start tag of either pair is held
// and reassembles.
func TestSkipTagsTokenReassemblesEitherPair(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, []text.StartEndTags{
		{Start: "<spell>", End: "</spell>"},
		{Start: "<code>", End: "</code>"},
	})

	var got []text.Aggregation
	for _, tok := range []string{
		"Call ", "<cod", "e>x</code>", " and ", "<spe", "ll>y</spell>", " done",
	} {
		got = append(got, a.Aggregate(tok)...)
	}
	want := []string{"Call ", "<code>x</code>", " and ", "<spell>y</spell>", " done"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}

// A tag opened after an earlier one closed is scanned like any other. The scan
// offset has to come back to the front of what is left when text is released,
// or every tag after the first is missed.
func TestSkipTagsTokenSeesASecondTag(t *testing.T) {
	a := newSkipTags(t, frames.AggregationToken, spellTag())

	var got []text.Aggregation
	for _, tok := range []string{"<spell>abc</spell>", "<spell>de", "f</spell>", " done"} {
		got = append(got, a.Aggregate(tok)...)
	}
	want := []string{"<spell>abc</spell>", "<spell>def</spell>", " done"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}
