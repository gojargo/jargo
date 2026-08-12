package text

import (
	"strings"

	"github.com/gojargo/jargo/frames"
)

// Aggregation is a run of text an aggregator has completed, and how it was
// aggregated.
type Aggregation struct {
	// Text is the aggregated text.
	Text string
	// Type is how it was aggregated.
	Type frames.AggregationType
	// RawText is the text this unit was cut from, delimiters and all, set when
	// the unit came from a matched pattern such as a code block. Empty for an
	// ordinary aggregation, where Text is already the written form.
	RawText string
}

// Original returns the text to record as written: RawText when the unit was cut
// from a larger match, and Text otherwise.
func (a Aggregation) Original() string {
	if a.RawText != "" {
		return a.RawText
	}
	return a.Text
}

// Aggregator groups streamed text into the units a synthesizer is given.
type Aggregator interface {
	// Type is how this aggregator groups text. A caller reads it to know what to
	// expect before any text has been aggregated: grouping by token, say, means
	// no unit ever completes a sentence.
	Type() frames.AggregationType
	// Aggregate folds text into the buffer and returns every unit it completed.
	Aggregate(text string) []Aggregation
	// Flush returns whatever is left in the buffer, for the end of a response.
	Flush() (Aggregation, bool)
	// Reset clears the buffer.
	Reset()
}

// SimpleAggregator groups text into sentences, or passes each token straight
// through when aggregating by token.
//
// A sentence boundary is confirmed by lookahead: the mark that could end a
// sentence is not acted on until the first non-whitespace character after it
// has arrived. That is what tells "$29." apart from "$29. Next", since only the
// character following the period says whether the text ended there. Whitespace
// alone says nothing, because it appears in both.
type SimpleAggregator struct {
	aggregationType frames.AggregationType
	tokenizer       SentenceTokenizer
	text            string
	needsLookahead  bool
}

// NewSimpleAggregator builds an aggregator that groups text by aggregateBy,
// finding sentence boundaries with tokenizer.
func NewSimpleAggregator(aggregateBy frames.AggregationType, tokenizer SentenceTokenizer) *SimpleAggregator {
	return &SimpleAggregator{aggregationType: aggregateBy, tokenizer: tokenizer}
}

// Type implements Aggregator.
func (a *SimpleAggregator) Type() frames.AggregationType { return a.aggregationType }

// Aggregate implements Aggregator. Aggregating by token returns the text as it
// arrives; aggregating by sentence walks it a character at a time, so a chunk
// carrying more than one boundary completes more than one sentence.
func (a *SimpleAggregator) Aggregate(text string) []Aggregation {
	if a.aggregationType == frames.AggregationToken {
		if text == "" {
			return nil
		}
		return []Aggregation{{Text: text, Type: frames.AggregationToken}}
	}
	var out []Aggregation
	for _, r := range text {
		a.text += string(r)
		if agg, ok := a.checkSentenceWithLookahead(r); ok {
			out = append(out, agg)
		}
	}
	return out
}

// checkSentenceWithLookahead reports the sentence the latest character
// completed, if it completed one. Callers pass the character just appended.
func (a *SimpleAggregator) checkSentenceWithLookahead(r rune) (Aggregation, bool) {
	if a.needsLookahead {
		if strings.TrimSpace(string(r)) == "" {
			// Still whitespace, so nothing has been confirmed yet.
			return Aggregation{}, false
		}
		a.needsLookahead = false
		end := a.tokenizer.MatchEndOfSentence(a.text)
		if end <= 0 {
			// Not a boundary after all; keep accumulating.
			return Aggregation{}, false
		}
		sentence := a.text[:end]
		a.text = a.text[end:]
		return Aggregation{Text: strings.Trim(sentence, " "), Type: frames.AggregationSentence}, true
	}
	if IsSentenceEnding(r) {
		// Wait for the next non-whitespace character before deciding.
		a.needsLookahead = true
	}
	return Aggregation{}, false
}

// Flush implements Aggregator. Aggregating by token buffers nothing, so there is
// never anything left.
func (a *SimpleAggregator) Flush() (Aggregation, bool) {
	if a.aggregationType == frames.AggregationToken || a.text == "" {
		return Aggregation{}, false
	}
	rest := a.text
	a.Reset()
	return Aggregation{Text: strings.Trim(rest, " "), Type: frames.AggregationSentence}, true
}

// Reset implements Aggregator.
func (a *SimpleAggregator) Reset() {
	a.text = ""
	a.needsLookahead = false
}

// Text reports what is buffered but not yet complete.
func (a *SimpleAggregator) Text() Aggregation {
	return Aggregation{Text: strings.Trim(a.text, " "), Type: frames.AggregationSentence}
}

// Buffer reports the raw buffered text, untrimmed. A caller running channels of
// text in parallel compares against it to tell whether its own buffer still
// mirrors this one.
func (a *SimpleAggregator) Buffer() string { return a.text }

var _ Aggregator = (*SimpleAggregator)(nil)
