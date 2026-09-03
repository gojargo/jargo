package text_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/text"
)

// recorder collects the matches a handler was called with.
type recorder struct {
	calls []text.Aggregation
}

func (r *recorder) handle(m text.Aggregation) { r.calls = append(r.calls, m) }

func (r *recorder) only(t *testing.T) text.Aggregation {
	t.Helper()
	if len(r.calls) != 1 {
		t.Fatalf("handler called %d times, want once: %+v", len(r.calls), r.calls)
	}
	return r.calls[0]
}

// newPatternPair builds an aggregator carrying the two patterns most of these
// tests use: one whose match is removed, and one returned on its own.
func newPatternPair(
	t *testing.T, by frames.AggregationType,
) (*text.PatternPairAggregator, *recorder, *recorder) {
	t.Helper()
	tok, err := text.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	a := text.NewPatternPairAggregator(by, tok)
	if err := a.AddPattern("test_pattern", "<test>", "</test>", text.MatchRemove); err != nil {
		t.Fatal(err)
	}
	if err := a.AddPattern("code_pattern", "<code>", "</code>", text.MatchAggregate); err != nil {
		t.Fatal(err)
	}
	testH, codeH := &recorder{}, &recorder{}
	a.OnPatternMatch("test_pattern", testH.handle)
	a.OnPatternMatch("code_pattern", codeH.handle)
	return a, testH, codeH
}

// A built-in aggregation type describes how text is grouped by default, so it
// cannot also name a pattern.
func TestPatternPairRejectsAReservedType(t *testing.T) {
	a, _, _ := newPatternPair(t, frames.AggregationSentence)
	for _, typ := range []frames.AggregationType{
		frames.AggregationSentence, frames.AggregationWord, frames.AggregationToken,
	} {
		if err := a.AddPattern(typ, "<x>", "</x>", text.MatchRemove); err == nil {
			t.Errorf("AddPattern(%q) = nil, want an error", typ)
		}
	}
}

// A matched pattern is reported to its handler and taken out of the text, which
// then aggregates as though the run had never been there.
func TestPatternPairRemovesAMatch(t *testing.T) {
	a, testH, _ := newPatternPair(t, frames.AggregationSentence)

	got := a.Aggregate("Hello <test>pattern content</test>!")
	if len(got) != 0 {
		t.Fatalf("a trailing mark should wait for lookahead, got %+v", got)
	}
	m := testH.only(t)
	if m.Type != "test_pattern" || m.Text != "pattern content" ||
		m.RawText != "<test>pattern content</test>" {
		t.Fatalf("match = %+v, want the content and the text it was cut from", m)
	}

	got = a.Aggregate(" This is another sentence.")
	if len(got) != 1 || got[0].Text != "Hello !" || got[0].Type != frames.AggregationSentence {
		t.Fatalf("aggregations = %+v, want one sentence %q", got, "Hello !")
	}
	rest, ok := a.Flush()
	if !ok || rest.Text != "This is another sentence." {
		t.Fatalf("flush = %+v (%v), want the last sentence", rest, ok)
	}
	if got := a.Text().Text; got != "" {
		t.Fatalf("buffer after flush = %q, want empty", got)
	}
}

// A pattern registered to be returned on its own releases the text before it
// first, complete sentence or not, then the match as a unit of its own.
func TestPatternPairReturnsAMatchOnItsOwn(t *testing.T) {
	a, _, codeH := newPatternPair(t, frames.AggregationSentence)

	got := a.Aggregate("Here is code <code>pattern content</code> This is another sentence.")
	if len(got) != 2 {
		t.Fatalf("aggregations = %+v, want the text before the run and the run", got)
	}
	if got[0].Text != "Here is code" || got[0].Type != frames.AggregationSentence {
		t.Fatalf("first = %+v, want %q as a sentence", got[0], "Here is code")
	}
	if got[1].Text != "pattern content" || got[1].Type != "code_pattern" {
		t.Fatalf("second = %+v, want the run under its own type", got[1])
	}
	m := codeH.only(t)
	if m.Text != "pattern content" || m.RawText != "<code>pattern content</code>" {
		t.Fatalf("match = %+v, want the content and the text it was cut from", m)
	}

	rest, ok := a.Flush()
	if !ok || rest.Text != "This is another sentence." {
		t.Fatalf("flush = %+v (%v), want the last sentence", rest, ok)
	}
}

// A run that has not closed holds everything back, and the buffer reports the
// pattern it is partway through.
func TestPatternPairHoldsAnIncompleteMatch(t *testing.T) {
	a, testH, _ := newPatternPair(t, frames.AggregationSentence)

	if got := a.Aggregate("Hello <test>pattern content"); len(got) != 0 {
		t.Fatalf("an open run should hold its text, got %+v", got)
	}
	if len(testH.calls) != 0 {
		t.Fatalf("handler called for a run that never closed: %+v", testH.calls)
	}
	if got := a.Text(); got.Text != "Hello <test>pattern content" || got.Type != "test_pattern" {
		t.Fatalf("buffer = %+v, want the text held under its pattern", got)
	}
	a.Reset()
	if got := a.Text().Text; got != "" {
		t.Fatalf("buffer after reset = %q, want empty", got)
	}
}

// Several patterns run side by side, each with its own action: one removed, one
// kept in the text with only its delimiters taken out.
func TestPatternPairRunsSeveralPatterns(t *testing.T) {
	a, _, _ := newPatternPair(t, frames.AggregationSentence)
	voiceH, emH := &recorder{}, &recorder{}
	if err := a.AddPattern("voice", "<voice>", "</voice>", text.MatchRemove); err != nil {
		t.Fatal(err)
	}
	if err := a.AddPattern("emphasis", "<em>", "</em>", text.MatchKeep); err != nil {
		t.Fatal(err)
	}
	a.OnPatternMatch("voice", voiceH.handle)
	a.OnPatternMatch("emphasis", emH.handle)

	got := a.Aggregate("Hello <voice>female</voice> I am <em>very</em> excited to meet you!")
	if len(got) != 0 {
		t.Fatalf("a trailing mark should wait for lookahead, got %+v", got)
	}
	if m := voiceH.only(t); m.Type != "voice" || m.Text != "female" {
		t.Fatalf("voice match = %+v", m)
	}
	if m := emH.only(t); m.Type != "emphasis" || m.Text != "very" {
		t.Fatalf("emphasis match = %+v", m)
	}

	rest, ok := a.Flush()
	want := "Hello  I am <em>very</em> excited to meet you!"
	if !ok || rest.Text != want {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, want)
	}
}

// A run spanning several sentences is one match, and the sentences inside it
// are not cut apart.
func TestPatternPairMatchesAcrossSentences(t *testing.T) {
	a, testH, _ := newPatternPair(t, frames.AggregationSentence)

	got := a.Aggregate(
		"Hello <test>This is sentence one. This is sentence two.</test> Final sentence.")
	if len(got) != 0 {
		t.Fatalf("a trailing mark should wait for lookahead, got %+v", got)
	}
	if m := testH.only(t); m.Text != "This is sentence one. This is sentence two." {
		t.Fatalf("match = %+v, want both sentences in one match", m)
	}
	rest, ok := a.Flush()
	if !ok || rest.Text != "Hello  Final sentence." {
		t.Fatalf("flush = %+v (%v), want the run taken out", rest, ok)
	}
}

// A run to be removed whose closing delimiter never arrived is degraded the way
// a closed one would have been: the text before it is kept, the run is dropped,
// and its handler is never called.
func TestPatternPairFlushDropsAnUnclosedRemoval(t *testing.T) {
	a, testH, _ := newPatternPair(t, frames.AggregationSentence)

	a.Aggregate("Well <test>pattern content")
	rest, ok := a.Flush()
	if !ok || rest.Text != "Well" {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, "Well")
	}
	if strings.Contains(rest.Text, "<test>") || strings.Contains(rest.Text, "pattern content") {
		t.Fatalf("flush = %q, want the unclosed run dropped", rest.Text)
	}
	if len(testH.calls) != 0 {
		t.Fatalf("handler called for a run that never closed: %+v", testH.calls)
	}
	if got := a.Text().Text; got != "" {
		t.Fatalf("buffer after flush = %q, want empty", got)
	}
}

// Nothing before the unclosed run means nothing is left to say.
func TestPatternPairFlushDropsARunWithNothingBeforeIt(t *testing.T) {
	a, testH, _ := newPatternPair(t, frames.AggregationSentence)

	a.Aggregate("<test>pattern content")
	if rest, ok := a.Flush(); ok {
		t.Fatalf("flush = %+v, want nothing left", rest)
	}
	if len(testH.calls) != 0 {
		t.Fatalf("handler called for a run that never closed: %+v", testH.calls)
	}
}

// The aggregator is usable again once a flush has dropped an unclosed run.
func TestPatternPairFlushLeavesItReusable(t *testing.T) {
	a, _, _ := newPatternPair(t, frames.AggregationSentence)

	a.Aggregate("Well <test>pattern content")
	a.Flush()
	if got := a.Text().Text; got != "" {
		t.Fatalf("buffer after flush = %q, want empty", got)
	}

	got := a.Aggregate("New <test>value</test> sentence.")
	rest, _ := a.Flush()
	var b strings.Builder
	for _, g := range got {
		b.WriteString(g.Text)
	}
	b.WriteString(rest.Text)
	combined := b.String()
	if strings.Contains(combined, "value") {
		t.Fatalf("combined = %q, want the run removed", combined)
	}
	if !strings.Contains(combined, "New") {
		t.Fatalf("combined = %q, want the text around the run kept", combined)
	}
}

// A closed run earlier in the buffer must not be mistaken for the unclosed one
// that follows it.
func TestPatternPairFlushFindsTheUnclosedOccurrence(t *testing.T) {
	a, testH, _ := newPatternPair(t, frames.AggregationSentence)

	got := a.Aggregate("Start <test>closed</test> middle <test>unclosed")
	rest, _ := a.Flush()
	var b strings.Builder
	for _, g := range got {
		b.WriteString(g.Text)
	}
	b.WriteString(rest.Text)
	combined := b.String()
	if !strings.Contains(combined, "Start") || !strings.Contains(combined, "middle") {
		t.Fatalf("combined = %q, want the text around the runs kept", combined)
	}
	if strings.Contains(combined, "closed") {
		t.Fatalf("combined = %q, want both runs dropped", combined)
	}
	testH.only(t)
}

// Two unclosed runs of different patterns: the cut is made at whichever opens
// first in the text, not whichever pattern was registered first.
func TestPatternPairFlushCutsAtTheEarliestRun(t *testing.T) {
	tok, err := text.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	a := text.NewPatternPairAggregator(frames.AggregationSentence, tok)
	// Registered in the reverse of the order they appear in the text.
	if err := a.AddPattern("voice", "<voice>", "</voice>", text.MatchRemove); err != nil {
		t.Fatal(err)
	}
	if err := a.AddPattern("test2", "<test>", "</test>", text.MatchRemove); err != nil {
		t.Fatal(err)
	}
	voiceH, testH := &recorder{}, &recorder{}
	a.OnPatternMatch("voice", voiceH.handle)
	a.OnPatternMatch("test2", testH.handle)

	if got := a.Aggregate("Hi <test>foo <voice>bar"); len(got) != 0 {
		t.Fatalf("an open run should hold its text, got %+v", got)
	}
	rest, ok := a.Flush()
	if !ok || rest.Text != "Hi" {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, "Hi")
	}
	if len(voiceH.calls) != 0 || len(testH.calls) != 0 {
		t.Fatal("a handler was called for a run that never closed")
	}
}

// A run whose content is kept is spoken whether or not it closed, so an
// unclosed one is kept verbatim rather than dropped.
func TestPatternPairFlushKeepsAnUnclosedKeep(t *testing.T) {
	a, _, _ := newPatternPair(t, frames.AggregationSentence)
	if err := a.AddPattern("emphasis", "<em>", "</em>", text.MatchKeep); err != nil {
		t.Fatal(err)
	}

	const line = "Well <em>unclosed content"
	a.Aggregate(line)
	rest, ok := a.Flush()
	if !ok || rest.Text != line {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, line)
	}
}

// A closed kept run earlier in the buffer is not mistaken for the unclosed one.
func TestPatternPairFlushKeepsBothKeptRuns(t *testing.T) {
	a, _, _ := newPatternPair(t, frames.AggregationSentence)
	if err := a.AddPattern("emphasis", "<em>", "</em>", text.MatchKeep); err != nil {
		t.Fatal(err)
	}

	const line = "Hello <em>bold</em> world <em>unclosed"
	a.Aggregate(line)
	rest, ok := a.Flush()
	if !ok || rest.Text != line {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, line)
	}
}

// A run to be returned on its own is never spoken as it stands, so an unclosed
// one is dropped and its handler is not called.
func TestPatternPairFlushDropsAnUnclosedAggregate(t *testing.T) {
	a, _, codeH := newPatternPair(t, frames.AggregationSentence)

	got := a.Aggregate("Before <code>unclosed content")
	if len(got) != 1 || got[0].Text != "Before" {
		t.Fatalf("aggregations = %+v, want the text before the run released", got)
	}
	if rest, ok := a.Flush(); ok {
		t.Fatalf("flush = %+v, want the unclosed run dropped", rest)
	}
	if len(codeH.calls) != 0 {
		t.Fatalf("handler called for a run that never closed: %+v", codeH.calls)
	}
}

// A stream cut off partway through an opening delimiter has the partial
// delimiter trimmed rather than spoken.
func TestPatternPairFlushTrimsAPartialDelimiter(t *testing.T) {
	a, _, _ := newPatternPair(t, frames.AggregationSentence)

	a.Aggregate("Hello <te")
	rest, ok := a.Flush()
	if !ok || rest.Text != "Hello" {
		t.Fatalf("flush = %+v (%v), want %q", rest, ok, "Hello")
	}
}

// newThinkPair builds a token-mode aggregator carrying one removed pattern.
func newThinkPair(t *testing.T) (*text.PatternPairAggregator, *recorder) {
	t.Helper()
	tok, err := text.NewPunktEnglish()
	if err != nil {
		t.Fatal(err)
	}
	a := text.NewPatternPairAggregator(frames.AggregationToken, tok)
	if err := a.AddPattern("think", "<think>", "</think>", text.MatchRemove); err != nil {
		t.Fatal(err)
	}
	h := &recorder{}
	a.OnPatternMatch("think", h.handle)
	return a, h
}

// Aggregating by token passes text carrying no pattern straight through.
func TestPatternPairTokenPassesTextThrough(t *testing.T) {
	a, _ := newThinkPair(t)

	var got []text.Aggregation
	for _, tok := range []string{"Hello", " world", "."} {
		got = append(got, a.Aggregate(tok)...)
	}
	want := []string{"Hello", " world", "."}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
	for _, g := range got {
		if g.Type != frames.AggregationToken {
			t.Fatalf("type = %q, want %q", g.Type, frames.AggregationToken)
		}
	}
}

// A pattern is recognized just the same when the text arrives token by token.
func TestPatternPairTokenFindsAPattern(t *testing.T) {
	a, h := newThinkPair(t)

	var got []text.Aggregation
	for _, tok := range []string{"Hi ", "<think>", "secret", "</think>", " bye"} {
		got = append(got, a.Aggregate(tok)...)
	}
	if m := h.only(t); m.Text != "secret" {
		t.Fatalf("match = %+v, want the run content", m)
	}
	want := []string{"Hi ", " bye"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}

// A run that has not closed is held rather than passed through.
func TestPatternPairTokenHoldsAnIncompleteRun(t *testing.T) {
	a, h := newThinkPair(t)

	var got []text.Aggregation
	for _, tok := range []string{"Hi ", "<think>", "partial"} {
		got = append(got, a.Aggregate(tok)...)
	}
	if !equalStrings(texts(got), []string{"Hi "}) {
		t.Fatalf("aggregations = %q, want only the text before the run", texts(got))
	}
	if len(h.calls) != 0 {
		t.Fatalf("handler called for a run that never closed: %+v", h.calls)
	}
}

// Flushing by token drops a removed run that never closed.
func TestPatternPairTokenFlushDropsAnUnclosedRun(t *testing.T) {
	a, h := newThinkPair(t)

	var got []text.Aggregation
	for _, tok := range []string{"Hi ", "<think>", "secret"} {
		got = append(got, a.Aggregate(tok)...)
	}
	if !equalStrings(texts(got), []string{"Hi "}) {
		t.Fatalf("aggregations = %q, want only the text before the run", texts(got))
	}
	if rest, ok := a.Flush(); ok {
		t.Fatalf("flush = %+v, want the unclosed run dropped", rest)
	}
	if len(h.calls) != 0 {
		t.Fatalf("handler called for a run that never closed: %+v", h.calls)
	}
	if got := a.Text().Text; got != "" {
		t.Fatalf("buffer after flush = %q, want empty", got)
	}
}

// An opening delimiter split across chunks must not leak as ordinary text, or
// the run it opens is never recognized and its content gets spoken.
func TestPatternPairTokenReassemblesASplitDelimiter(t *testing.T) {
	a, h := newThinkPair(t)

	var got []text.Aggregation
	for _, tok := range []string{"Hi ", "<thin", "k>secret</think>", " bye"} {
		got = append(got, a.Aggregate(tok)...)
	}
	if m := h.only(t); m.Text != "secret" {
		t.Fatalf("match = %+v, want the run content", m)
	}
	want := []string{"Hi ", " bye"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}

// The same, split three ways.
func TestPatternPairTokenReassemblesAcrossThreeChunks(t *testing.T) {
	a, h := newThinkPair(t)

	var got []text.Aggregation
	for _, tok := range []string{"<th", "in", "k>secret</think>"} {
		got = append(got, a.Aggregate(tok)...)
	}
	if m := h.only(t); m.Text != "secret" {
		t.Fatalf("match = %+v, want the run content", m)
	}
	if len(got) != 0 {
		t.Fatalf("aggregations = %+v, want nothing leaked while the delimiter reassembled", got)
	}
}

// A chunk that is nothing but the start of a delimiter yields nothing.
func TestPatternPairTokenHoldsAPartialDelimiterChunk(t *testing.T) {
	a, _ := newThinkPair(t)

	if got := a.Aggregate("<thin"); len(got) != 0 {
		t.Fatalf("a partial delimiter should be held, got %+v", got)
	}
	if got := a.Text().Text; got != "<thin" {
		t.Fatalf("buffer = %q, want the partial delimiter held", got)
	}
}

// A trailing character that could open a delimiter is held, then released once
// the next chunk shows it opens nothing.
func TestPatternPairTokenReleasesAHeldPrefix(t *testing.T) {
	a, _ := newThinkPair(t)

	got := a.Aggregate("Hello <")
	if len(got) != 1 || got[0].Text != "Hello " {
		t.Fatalf("aggregations = %+v, want the trailing %q held back", got, "<")
	}
	got = a.Aggregate(" world")
	if len(got) != 1 || got[0].Text != "< world" {
		t.Fatalf("aggregations = %+v, want the held character released", got)
	}
}

// A split delimiter of a pattern returned on its own reassembles, so its
// content arrives as a unit instead of leaking in fragments.
func TestPatternPairTokenReassemblesAnAggregateDelimiter(t *testing.T) {
	a, _ := newThinkPair(t)
	codeH := &recorder{}
	if err := a.AddPattern("code_pattern", "<code>", "</code>", text.MatchAggregate); err != nil {
		t.Fatal(err)
	}
	a.OnPatternMatch("code_pattern", codeH.handle)

	var got []text.Aggregation
	for _, tok := range []string{"Here is code ", "<cod", "e>pattern content</code>", " more"} {
		got = append(got, a.Aggregate(tok)...)
	}
	if m := codeH.only(t); m.Text != "pattern content" {
		t.Fatalf("match = %+v, want the run content", m)
	}
	if len(got) != 3 {
		t.Fatalf("aggregations = %+v, want three", got)
	}
	if got[0].Type != frames.AggregationToken || got[0].Text != "Here is code " {
		t.Fatalf("first = %+v", got[0])
	}
	if got[1].Type != "code_pattern" || got[1].Text != "pattern content" {
		t.Fatalf("second = %+v", got[1])
	}
	if got[2].Type != frames.AggregationToken || got[2].Text != " more" {
		t.Fatalf("third = %+v", got[2])
	}
}

// A split delimiter of a kept pattern reassembles too, and the delimiters stay
// in the text.
func TestPatternPairTokenReassemblesAKeptDelimiter(t *testing.T) {
	a, _ := newThinkPair(t)
	keepH := &recorder{}
	if err := a.AddPattern("em", "<em>", "</em>", text.MatchKeep); err != nil {
		t.Fatal(err)
	}
	a.OnPatternMatch("em", keepH.handle)

	var got []text.Aggregation
	for _, tok := range []string{"very <e", "m>excited</em> today"} {
		got = append(got, a.Aggregate(tok)...)
	}
	if m := keepH.only(t); m.Text != "excited" {
		t.Fatalf("match = %+v, want the run content", m)
	}
	want := []string{"very ", "<em>excited</em> today"}
	if !equalStrings(texts(got), want) {
		t.Fatalf("aggregations = %q, want %q", texts(got), want)
	}
}

// An interruption discards the buffer and the scan offset. The registered
// patterns survive it, as they survive a reset: they describe the text this
// aggregator reads, not the response it was partway through.
func TestPatternPairHandleInterruptionKeepsThePatterns(t *testing.T) {
	a, testH, _ := newPatternPair(t, frames.AggregationSentence)

	if got := a.Aggregate("Hello <test>pattern content"); len(got) != 0 {
		t.Fatalf("an open run should hold its text, got %+v", got)
	}
	a.HandleInterruption()
	if got := a.Text().Text; got != "" {
		t.Fatalf("buffer after an interruption = %q, want empty", got)
	}
	if len(testH.calls) != 0 {
		t.Fatalf("handler called for a run the interruption cut off: %+v", testH.calls)
	}

	// The pattern is still registered, so a complete run after the interruption
	// is matched and handed to the same handler.
	a.Aggregate("<test>after</test> Done. X")
	if got := testH.only(t); got.Text != "after" {
		t.Fatalf("handler saw %q, want the pattern to have survived as %q", got.Text, "after")
	}
}
