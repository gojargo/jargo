package text

import (
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"

	"github.com/gojargo/jargo/frames"
)

// MatchAction is what an aggregator does with a pattern once both its
// delimiters have arrived.
type MatchAction string

const (
	// MatchRemove drops the match and its delimiters from the text. Aggregation
	// carries on as though the run had never been there, which is what a model
	// asked to mark up its own reasoning wants: the markup is read by the
	// handler and never spoken.
	MatchRemove MatchAction = "remove"
	// MatchKeep drops the delimiters and keeps what was between them, which then
	// aggregates as ordinary text. A handler still sees the match, so this is
	// how to be told about a run without changing what is spoken.
	MatchKeep MatchAction = "keep"
	// MatchAggregate drops the delimiters and returns what was between them as a
	// unit of its own, whole rather than cut into sentences. Any text before the
	// opening delimiter is returned first, complete sentence or not.
	MatchAggregate MatchAction = "aggregate"
)

// ErrReservedAggregationType is returned by AddPattern for a type that names
// how text is grouped by default, which cannot also name a pattern.
var ErrReservedAggregationType = errors.New("aggregation type is reserved")

// PatternHandler is called with each completed match of the pattern it was
// registered for.
type PatternHandler func(match Aggregation)

// pattern is one registered delimiter pair and what to do with what it matches.
type pattern struct {
	typ    frames.AggregationType
	start  string
	end    string
	action MatchAction
	// re matches the opening delimiter, the content, and the closing delimiter.
	re *regexp.Regexp
}

// PatternPairAggregator groups text into sentences the way SimpleAggregator
// does, and additionally recognizes runs delimited by a registered pair.
//
// It is for a model whose output carries structure the speech should not: a
// reasoning block to be dropped, a code block to be spoken as one unit rather
// than cut at every period, a speaker tag that should switch voices. A handler
// registered for a pattern is called as each match completes, and the action
// the pattern was registered with decides whether what matched is dropped, kept
// as ordinary text, or returned as a unit of its own.
//
// A pattern split across several pieces of arriving text is still recognized.
type PatternPairAggregator struct {
	*SimpleAggregator
	// patterns is kept in the order the patterns were added, because a text
	// matching more than one is resolved by taking the first that matches.
	patterns []*pattern
	handlers map[frames.AggregationType]PatternHandler
	// lastProcessed is how far into the buffer matches have already been
	// reported, so a handler is called once per match rather than once per
	// character that arrives after it.
	lastProcessed int
}

// NewPatternPairAggregator builds an aggregator that groups text by aggregateBy
// and finds sentence boundaries with tokenizer. Register the delimiter pairs to
// recognize with AddPattern.
func NewPatternPairAggregator(
	aggregateBy frames.AggregationType, tokenizer SentenceTokenizer,
) *PatternPairAggregator {
	return &PatternPairAggregator{
		SimpleAggregator: NewSimpleAggregator(aggregateBy, tokenizer),
		handlers:         map[frames.AggregationType]PatternHandler{},
	}
}

// AddPattern registers a delimiter pair to recognize, under a type naming what
// it delimits ("code", "speaker", "reasoning"). The type is what a handler is
// registered against and what the resulting unit is labeled with, so it cannot
// be one of the built-in aggregation types.
func (a *PatternPairAggregator) AddPattern(
	typ frames.AggregationType, start, end string, action MatchAction,
) error {
	switch typ {
	case frames.AggregationSentence, frames.AggregationWord, frames.AggregationToken,
		frames.AnyAggregation:
		return fmt.Errorf("%w: %q", ErrReservedAggregationType, typ)
	}
	a.patterns = append(a.patterns, &pattern{
		typ:    typ,
		start:  start,
		end:    end,
		action: action,
		// Ungreedy, so the nearest closing delimiter ends the match rather than
		// the last one in the buffer. The dot spans newlines: a code block is
		// several lines and is still one match.
		re: regexp.MustCompile("(?s)" + regexp.QuoteMeta(start) + "(.*?)" + regexp.QuoteMeta(end)),
	})
	return nil
}

// OnPatternMatch registers the handler called each time a pattern of the given
// type completes.
func (a *PatternPairAggregator) OnPatternMatch(typ frames.AggregationType, h PatternHandler) {
	a.handlers[typ] = h
}

// patternFor returns the registered pattern of the given type.
func (a *PatternPairAggregator) patternFor(typ frames.AggregationType) *pattern {
	for _, p := range a.patterns {
		if p.typ == typ {
			return p
		}
	}
	return nil
}

// Aggregate implements Aggregator.
func (a *PatternPairAggregator) Aggregate(text string) []Aggregation {
	var out []Aggregation

	for _, r := range text {
		a.text += string(r)

		matches, processed := a.processCompletePatterns(a.text, a.lastProcessed)
		// Measured against the buffer as it was scanned, before anything a
		// removal takes out of it, so a match already reported stays reported.
		a.lastProcessed = len(a.text)
		a.text = processed

		if len(matches) > 0 {
			if len(matches) > 1 {
				types := make([]string, 0, len(matches))
				for _, m := range matches {
					types = append(types, string(m.Type))
				}
				slog.Warn("several patterns matched the same text, taking the first",
					"types", strings.Join(types, ", "))
			}
			if p := a.patternFor(matches[0].Type); p != nil && p.action == MatchAggregate {
				a.text = ""
				out = append(out, matches[0])
				continue
			}
		}

		if start, p := a.matchStartOfPattern(a.text); p != nil {
			if start == 0 || p.action != MatchAggregate {
				// Inside a run that has not closed yet, so nothing here can
				// complete a sentence.
				continue
			}
			// A run about to be returned on its own starts partway through the
			// buffer. What comes before it is released now rather than waiting
			// for a boundary that the run would interrupt anyway.
			result := a.text[:start]
			a.text = a.text[start:]
			out = append(out, Aggregation{
				Text:    strings.TrimSpace(result),
				Type:    a.plainType(),
				RawText: result,
			})
			continue
		}

		if a.aggregationType != frames.AggregationToken {
			if agg, ok := a.checkSentenceWithLookahead(r); ok {
				agg.RawText = agg.Text
				out = append(out, agg)
			}
		}
	}

	if agg, ok := a.releaseToken(); ok {
		out = append(out, agg)
	}

	return out
}

// releaseToken releases what is left in the buffer once a whole piece of text
// has been scanned, when aggregating by token. Nothing is released while a run
// is open, or for a buffer ending in what could be the start of one.
func (a *PatternPairAggregator) releaseToken() (Aggregation, bool) {
	if a.aggregationType != frames.AggregationToken || a.text == "" {
		return Aggregation{}, false
	}
	if _, p := a.matchStartOfPattern(a.text); p != nil {
		return Aggregation{}, false
	}
	starts := make([]string, 0, len(a.patterns))
	for _, pat := range a.patterns {
		starts = append(starts, pat.start)
	}
	held := LongestTrailingPartialMatch(a.text, starts)
	size := len(a.text) - held
	if size <= 0 {
		return Aggregation{}, false
	}
	content := a.text[:size]
	a.text = a.text[size:]
	a.lastProcessed = len(a.text)
	return Aggregation{Text: content, Type: frames.AggregationToken, RawText: content}, true
}

// plainType is the type given to text that matched no pattern.
func (a *PatternPairAggregator) plainType() frames.AggregationType {
	if a.aggregationType == frames.AggregationToken {
		return frames.AggregationToken
	}
	return frames.AggregationSentence
}

// processCompletePatterns reports the matches that have completed in text and
// returns the text left once the removals have been taken out. A match ending at
// or before reported was already announced by an earlier call, so its handler is
// not called again and it is not removed twice.
func (a *PatternPairAggregator) processCompletePatterns(
	text string, reported int,
) ([]Aggregation, string) {
	var all []Aggregation
	processed := text

	for _, p := range a.patterns {
		for _, loc := range p.re.FindAllStringSubmatchIndex(processed, -1) {
			full := processed[loc[0]:loc[1]]
			content := processed[loc[2]:loc[3]]
			match := Aggregation{
				Text:    strings.TrimSpace(content),
				Type:    p.typ,
				RawText: full,
			}

			already := loc[1] <= reported
			if !already {
				if h := a.handlers[p.typ]; h != nil {
					h(match)
				}
			}

			if p.action == MatchRemove {
				if !already {
					processed = strings.Replace(processed, full, "", 1)
				}
				continue
			}
			all = append(all, match)
		}
	}

	return all, processed
}

// matchStartOfPattern reports where a run was opened and not yet closed, and
// which pattern opened it. A negative offset and a nil pattern mean every run in
// the text is closed.
func (a *PatternPairAggregator) matchStartOfPattern(text string) (int, *pattern) {
	for _, p := range a.patterns {
		// More openings than closings means one of them is still open. Only one
		// run is ever open at a time, so the first opening is that one.
		if strings.Count(text, p.start) > strings.Count(text, p.end) {
			return strings.Index(text, p.start), p
		}
	}
	return -1, nil
}

// Flush implements Aggregator.
//
// A run whose closing delimiter never arrived is degraded the way a closed one
// would have been. What was to be removed or returned on its own is dropped,
// since neither is ever spoken as it stands, along with the opening delimiter
// and everything after it. What was to be kept is kept, since a closed run of
// that kind is spoken too. A buffer ending partway through an opening delimiter
// is trimmed on the same grounds, which also trims a stray character that merely
// looks like the start of one.
func (a *PatternPairAggregator) Flush() (Aggregation, bool) {
	result := a.text
	if cut := a.earliestDroppableStart(result); cut >= 0 {
		result = result[:cut]
	}
	result = a.trimTrailingPartialStart(result)

	a.Reset()

	stripped := strings.TrimSpace(result)
	if stripped == "" {
		return Aggregation{}, false
	}
	return Aggregation{Text: stripped, Type: a.plainType(), RawText: result}, true
}

// earliestDroppableStart reports where the earliest unclosed opening delimiter
// that Flush should cut at begins, or -1 when there is none. A pattern whose
// content is kept is not a candidate, because an unclosed run of that kind is
// kept rather than cut.
func (a *PatternPairAggregator) earliestDroppableStart(text string) int {
	earliest := -1

	for _, p := range a.patterns {
		if p.action == MatchKeep {
			continue
		}
		spans := p.re.FindAllStringIndex(text, -1)

		from := 0
		for {
			idx := strings.Index(text[from:], p.start)
			if idx < 0 {
				break
			}
			idx += from

			// An opening delimiter inside a run that did close belongs to that
			// run. Skip the whole run so a delimiter nested in its content is
			// not read as one left open.
			if span := spanContaining(spans, idx); span != nil {
				from = span[1]
				continue
			}

			if earliest < 0 || idx < earliest {
				earliest = idx
			}
			break
		}
	}

	return earliest
}

// spanContaining returns the span holding idx, or nil when none does.
func spanContaining(spans [][]int, idx int) []int {
	for _, s := range spans {
		if s[0] <= idx && idx < s[1] {
			return s
		}
	}
	return nil
}

// trimTrailingPartialStart removes an opening delimiter the text ends partway
// through. A pattern whose content is kept is left alone, because its delimiter
// is kept whole or in part.
func (a *PatternPairAggregator) trimTrailingPartialStart(text string) string {
	trim := 0
	for _, p := range a.patterns {
		if p.action == MatchKeep {
			continue
		}
		for length := min(len(p.start)-1, len(text)); length > 0; length-- {
			if strings.HasSuffix(text, p.start[:length]) {
				trim = max(trim, length)
				break
			}
		}
	}
	if trim == 0 {
		return text
	}
	return text[:len(text)-trim]
}

// Reset implements Aggregator. The registered patterns and their handlers
// survive it: they describe the text this aggregator reads, not the response it
// is partway through.
func (a *PatternPairAggregator) Reset() {
	a.SimpleAggregator.Reset()
	a.lastProcessed = 0
}

// Text reports what is buffered but not yet complete, labeled with the pattern
// it is partway through when it is inside one.
func (a *PatternPairAggregator) Text() Aggregation {
	typ := frames.AggregationSentence
	if _, p := a.matchStartOfPattern(a.text); p != nil {
		typ = p.typ
	}
	return Aggregation{Text: strings.TrimSpace(a.text), Type: typ}
}

var _ Aggregator = (*PatternPairAggregator)(nil)
