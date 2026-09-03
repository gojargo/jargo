package text

import (
	"strings"

	"github.com/gojargo/jargo/frames"
)

// SkipTagsAggregator groups text into sentences the way SimpleAggregator does,
// except between a pair of tags, where it never looks for a sentence boundary.
//
// It is for text carrying delimited runs that must reach the synthesizer whole.
// A pronunciation tag such as <spell>A.B.C.</spell> is full of periods that end
// no sentence, and splitting it would hand the synthesizer half a tag. Once a
// start tag is seen the aggregator keeps accumulating unconditionally until the
// matching end tag arrives, whatever punctuation it passes on the way.
//
// The tags themselves are pass-through markers. They are not removed, because
// the synthesizer is what interprets them.
type SkipTagsAggregator struct {
	*SimpleAggregator
	tags            []StartEndTags
	currentTag      *StartEndTags
	currentTagIndex int
}

// NewSkipTagsAggregator builds an aggregator that groups text by aggregateBy,
// finding sentence boundaries with tokenizer, and holds off on the boundaries
// between any of the given tag pairs.
func NewSkipTagsAggregator(
	aggregateBy frames.AggregationType, tokenizer SentenceTokenizer, tags []StartEndTags,
) *SkipTagsAggregator {
	return &SkipTagsAggregator{
		SimpleAggregator: NewSimpleAggregator(aggregateBy, tokenizer),
		tags:             tags,
	}
}

// Aggregate implements Aggregator.
//
// Aggregating by token passes text straight through, except that a run inside a
// tag is held until the closing tag arrives, so the tag is never handed over in
// pieces. Text ending in what could be the start of a tag ("<spe" of "<spell>")
// is held back too, until the next piece says whether it opens one.
func (a *SkipTagsAggregator) Aggregate(text string) []Aggregation {
	if a.aggregationType == frames.AggregationToken {
		for _, r := range text {
			a.text += string(r)
			a.currentTag, a.currentTagIndex = ParseStartEndTags(
				a.text, a.tags, a.currentTag, a.currentTagIndex)
		}
		if a.currentTag != nil || a.text == "" {
			return nil
		}
		starts := make([]string, 0, len(a.tags))
		for _, t := range a.tags {
			starts = append(starts, t.Start)
		}
		held := LongestTrailingPartialMatch(a.text, starts)
		size := len(a.text) - held
		if size <= 0 {
			return nil
		}
		content := a.text[:size]
		a.text = a.text[size:]
		// What is held back has not been scanned as a tag yet, so the scan
		// starts again from the front of what is left.
		a.currentTagIndex = 0
		return []Aggregation{{Text: content, Type: frames.AggregationToken}}
	}

	var out []Aggregation
	for _, r := range text {
		a.text += string(r)
		a.currentTag, a.currentTagIndex = ParseStartEndTags(
			a.text, a.tags, a.currentTag, a.currentTagIndex)
		if a.currentTag != nil {
			// Inside a tag, so no punctuation here ends a sentence.
			continue
		}
		if agg, ok := a.checkSentenceWithLookahead(r); ok {
			out = append(out, agg)
		}
	}
	return out
}

// Flush implements Aggregator.
//
// Aggregating by sentence returns what is buffered as it stands. The tags mark
// text rather than replacing it, so a run whose closing tag never arrived is
// still spoken. Aggregating by token returns whatever is still held behind an
// unclosed tag, rather than dropping it.
func (a *SkipTagsAggregator) Flush() (Aggregation, bool) {
	if a.aggregationType == frames.AggregationToken {
		if a.text == "" {
			return Aggregation{}, false
		}
		rest := a.text
		a.Reset()
		return Aggregation{Text: rest, Type: frames.AggregationToken}, true
	}
	rest, ok := a.SimpleAggregator.Flush()
	if ok {
		// The embedded flush clears its own buffer, which it reaches directly
		// rather than through this type, so the tag state it knows nothing
		// about has to be cleared here. Without it a response ending inside an
		// unclosed tag would leave the next one scanning from an offset past
		// the end of an empty buffer, still believing it is inside the tag.
		a.clearTags()
	}
	return rest, ok
}

// clearTags forgets the tag the scan was inside and where it had reached.
func (a *SkipTagsAggregator) clearTags() {
	a.currentTag = nil
	a.currentTagIndex = 0
}

// HandleInterruption implements Aggregator. It clears the buffer and the tag
// the scan was inside, so the turn after the interruption starts outside any
// tag.
func (a *SkipTagsAggregator) HandleInterruption() {
	a.SimpleAggregator.HandleInterruption()
	a.clearTags()
}

// Reset implements Aggregator. It clears the buffer and the tag the scan was
// inside, so the next response starts outside any tag.
func (a *SkipTagsAggregator) Reset() {
	a.SimpleAggregator.Reset()
	a.clearTags()
}

// Text reports what is buffered but not yet complete.
func (a *SkipTagsAggregator) Text() Aggregation {
	return Aggregation{Text: strings.Trim(a.text, " "), Type: frames.AggregationSentence}
}

var _ Aggregator = (*SkipTagsAggregator)(nil)
