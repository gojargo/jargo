package text

import (
	"strings"
	"unicode"
	"unicode/utf8"

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
	// Text reports what is buffered but not yet complete.
	Text() Aggregation
	// Aggregate folds text into the buffer and returns every unit it completed.
	Aggregate(text string) []Aggregation
	// Flush returns whatever is left in the buffer, for the end of a response.
	Flush() (Aggregation, bool)
	// HandleInterruption is called when the turn this aggregator was reading was
	// cut off. It is separate from Reset so an aggregator can treat the two
	// differently: what is half-aggregated when the user barges in was still
	// going to be spoken, where a reset is a deliberate clearing of the buffer.
	// The aggregators here discard the buffer either way; one that wants to keep
	// what it had gathered says so by implementing this differently.
	HandleInterruption()
	// Reset clears the buffer.
	Reset()
	// Language is the normalized language code the sentence boundaries are
	// found in.
	Language() string
	// SetLanguage selects the language sentence boundaries are found in,
	// without discarding what is buffered. Call it between generations, to keep
	// one language for all the text of a generation.
	SetLanguage(language string)
}

// SimpleAggregator groups text into sentences, or passes each token straight
// through when aggregating by token.
//
// A sentence boundary is confirmed by lookahead: the mark that could end a
// sentence is not acted on until text after it has arrived. That is what tells
// "$29." apart from "$29. Next", since only what follows the period says whether
// the text ended there. Whitespace alone says nothing, because it appears in
// both, and neither does more punctuation: the second dot of "..." is part of
// the same mark.
type SimpleAggregator struct {
	aggregationType frames.AggregationType
	language        string
	text            string
	lookahead       lookaheadState
}

// lookaheadState tracks the lookahead after a candidate boundary, across
// chunks, so a clear boundary is emitted early and an unclear one is retried
// once the following word ends.
//
// A clear boundary is emitted at the first character after it: "Hello. N" is
// released without waiting for "Next ". One the sentence rules cannot resolve at
// that point is retried only when the following word is complete. Checking a
// partial word would be wrong: in "Albert I. Douglas" the prefix "Do" could be
// taken for a sentence starter and split the name after its initial.
type lookaheadState int

const (
	// lookaheadIdle means there is no candidate boundary to check.
	lookaheadIdle lookaheadState = iota
	// lookaheadAwaitingCharacter means a candidate is waiting for the first
	// character after it that is neither whitespace nor punctuation.
	lookaheadAwaitingCharacter
	// lookaheadAwaitingWord means the first check was made at a character that
	// does not start a word, such as an opening quote, and the candidate is
	// waiting for the word itself.
	lookaheadAwaitingWord
	// lookaheadInWord means the candidate is waiting for the end of the word
	// that follows it, to be checked once more there.
	lookaheadInWord
)

// NewSimpleAggregator builds an aggregator that groups text by aggregateBy,
// finding sentence boundaries in language; empty uses English.
func NewSimpleAggregator(aggregateBy frames.AggregationType, language string) *SimpleAggregator {
	return &SimpleAggregator{aggregationType: aggregateBy, language: ResolveSentenceTokenizerLanguage(language)}
}

// Language implements Aggregator.
func (a *SimpleAggregator) Language() string { return a.language }

// SetLanguage implements Aggregator.
func (a *SimpleAggregator) SetLanguage(language string) {
	a.language = ResolveSentenceTokenizerLanguage(language)
}

// NewTokenAggregator builds an aggregator that hands text on as it arrives,
// grouping nothing. It never looks for a sentence boundary: a service that
// streams tokens sends each one as the model wrote it, trading the naturalness
// sentence-sized synthesis gives for the latency of not waiting for one to
// finish.
func NewTokenAggregator() *SimpleAggregator {
	return &SimpleAggregator{aggregationType: frames.AggregationToken, language: ResolveSentenceTokenizerLanguage("")}
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
//
// Punctuation starts a candidate, and the sentence rules decide whether it
// ends a sentence:
//
//   - "Hello." or "Hello. " keeps buffering until meaningful text follows.
//   - "Hello. N" is checked at once and emits "Hello.", where "Dr. N" and
//     "$29.9" stay buffered as an abbreviation and a decimal.
//   - "...you and I. D", left unresolved, is retried when the word "Did" ends.
//     The prefixes in between are not checked: "Do" in "Albert I. Douglas"
//     could be taken for a sentence starter.
//   - "...you and I. Did?": the question mark ends the lookahead word and
//     starts a new candidate. The earlier period is checked first, without the
//     "?", and then the "?" waits for its own lookahead.
//
// A confirmed boundary emits only the text before it; what follows stays in
// the buffer. After the retry at the end of the word, a candidate is not
// checked again even if it is still unresolved. Flush returns whatever is left
// at the end of the response.
func (a *SimpleAggregator) checkSentenceWithLookahead(r rune) (Aggregation, bool) {
	isPunctuation := IsSentenceEnding(r)
	var (
		result Aggregation
		ok     bool
	)

	// The lookahead asks for a check at the first character after the
	// candidate and at the end of the following word, not at every character in
	// between.
	if a.advanceLookahead(r, isPunctuation) {
		// For "...I. Did?", check "...I. Did" for the earlier boundary. With the
		// "?" included, the check that accepts text ending on punctuation could
		// take the whole buffer. This slice is only for the check; the "?" stays
		// in the buffer.
		candidate := a.text
		if isPunctuation {
			candidate = a.text[:len(a.text)-utf8.RuneLen(r)]
		}
		if end := matchEndOfSentence(candidate, a.language); end > 0 {
			result = Aggregation{Text: strings.Trim(a.text[:end], " "), Type: frames.AggregationSentence}
			ok = true
			// Keep the lookahead text for the next sentence ("N" in "Hello. N").
			a.text = a.text[end:]
			a.lookahead = lookaheadIdle
		}
	}

	// A trailing "?" in "...I. Did?" needs its own lookahead, whether or not the
	// check above found a boundary at the earlier period.
	if isPunctuation {
		a.lookahead = lookaheadAwaitingCharacter
	}

	return result, ok
}

// advanceLookahead moves the pending candidate on by one character and reports
// whether the buffer is ready for a boundary check.
//
// The check is made once at the first character that is neither whitespace nor
// punctuation, and once more when the word that follows ends. An opening quote
// or a symbol does not start a word. When the first check finds no boundary:
//
//	'I. '      -> awaiting a character
//	'I. "'     -> awaiting a word (checked at the quote)
//	'I. "D'    -> in the word (no check)
//	'I. "Did ' -> idle (retried at the space ending the word)
func (a *SimpleAggregator) advanceLookahead(r rune, isPunctuation bool) bool {
	switch a.lookahead {
	case lookaheadAwaitingCharacter:
		if unicode.IsSpace(r) || isPunctuation {
			return false
		}
		if isAlnum(r) {
			a.lookahead = lookaheadInWord
		} else {
			a.lookahead = lookaheadAwaitingWord
		}
		// An ordinary boundary such as "Hello. N" is emitted at once.
		return true
	case lookaheadAwaitingWord:
		if isAlnum(r) {
			a.lookahead = lookaheadInWord
		}
		return false
	case lookaheadInWord:
		// Never retry at a partial word such as "Do" inside "Douglas".
		if unicode.IsSpace(r) || isPunctuation {
			a.lookahead = lookaheadIdle
			return true
		}
		return false
	default:
		return false
	}
}

// isAlnum reports whether r is a letter or a number, the characters a word is
// made of.
func isAlnum(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) }

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

// HandleInterruption implements Aggregator. What was half-gathered belongs to
// the turn that was cut off, so it is discarded.
//
// The buffer is cleared here rather than by calling Reset, so that a type
// embedding this one overriding Reset does not change what an interruption
// does: a Go method called on the embedded value dispatches to the embedded
// value, and the two are kept apart deliberately.
func (a *SimpleAggregator) HandleInterruption() {
	a.text = ""
	a.lookahead = lookaheadIdle
}

// Reset implements Aggregator.
func (a *SimpleAggregator) Reset() {
	a.text = ""
	a.lookahead = lookaheadIdle
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
