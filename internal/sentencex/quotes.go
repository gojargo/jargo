package sentencex

import (
	"strings"
	"sync"
	"unicode/utf8"
)

// Quote handling: detecting quote ranges, classifying quote candidates,
// pairing space-padded quotes the quote matcher misses, tagging symmetric-pair
// mispairings, and extending sentence boundaries past closers (both
// terminators sitting just inside a quote and orphan trailing closers).

// quoteMispairing labels a symmetric-pair quote range that may have been
// paired across a sentence break.
type quoteMispairing int

const (
	mispairingNone quoteMispairing = iota
	mispairingCertain
	mispairingPossible
)

// hasPossibleInlineSentenceBreak reports whether span holds the shape
// .[ \t]+[A-Z].
func hasPossibleInlineSentenceBreak(span string) bool {
	for _, dot := range matchIndices(span, ".") {
		tail := span[dot+1:]
		blanks := strings.IndexFunc(tail, func(c rune) bool { return c != ' ' && c != '\t' })
		if blanks < 0 {
			continue
		}
		if blanks > 0 && isASCIIUpper(rune(tail[blanks])) {
			return true
		}
	}
	return false
}

// isSymmetricQuoteCloser reports whether closer closes a symmetric quote pair,
// one whose opener and closer are the same symbol, like '. It catches
// space-padded cases the quote matcher fails to pair.
func isSymmetricQuoteCloser(closer string) bool {
	for i := range quotePairs {
		if quotePairs[i].open == quotePairs[i].close && quotePairs[i].close == closer {
			return true
		}
	}
	return false
}

// isOrphanCloser reports whether the closer at boundary in paragraph is an
// orphan: it has no matching opener earlier in the paragraph, or it is a lone
// stray. Asymmetric closers (», ”) are unambiguous and always orphans once
// the quote matcher has had its chance to pair them. Symmetric closers look
// at the closer's distribution across the paragraph, ignoring the ones a
// paired quote range consumed and contractions like wasn't.
func isOrphanCloser(
	paragraph string,
	boundary int,
	closer string,
	ranges []skippableRange,
	cache *orphanCloserPositions,
) bool {
	if !isSymmetricQuoteCloser(closer) {
		return true
	}

	before, total := cache.beforeAndTotal(paragraph, closer, boundary, ranges)
	atOrAfter := total - before
	unmatchedOpenerBefore := before%2 == 1
	loneStrayAtBoundary := before == 0 && atOrAfter == 1

	return unmatchedOpenerBefore || loneStrayAtBoundary
}

// identifyQuotePair returns the first quote pair whose opener starts span.
func identifyQuotePair(span string) *quotePair {
	for i := range quotePairs {
		if strings.HasPrefix(span, quotePairs[i].open) {
			return &quotePairs[i]
		}
	}
	return nil
}

// collectQuoteRanges appends every quote range of text to out. The fast path
// scans for pairs other than ', backtick and the CJK ones; the full scan uses
// the quote matcher, then also catches space-padded '…' and `…` pairs.
func collectQuoteRanges(text string, out []skippableRange) []skippableRange {
	// Fast path: no ', no backtick and no 0xE3 lead byte (CJK 《》 and 「」).
	if !strings.ContainsAny(text, "'`") && strings.IndexByte(text, 0xE3) < 0 {
		return scanUnambiguousQuotes(text, out)
	}

	out = appendRegexQuotePairs(text, out)
	return appendSpacePaddedQuotePairs(text, out)
}

// appendRegexQuotePairs appends every quote matcher match in text as a quote
// range.
func appendRegexQuotePairs(text string, out []skippableRange) []skippableRange {
	for _, m := range quotesFindAll(text) {
		out = append(out, newQuoteRange(m[0], m[1], identifyQuotePair(text[m[0]:])))
	}
	return out
}

// cleanQuoteOpener maps a single-character opener to its quote pair.
type cleanQuoteOpener struct {
	opener rune
	pair   *quotePair
}

// cleanQuoteOpeners are the single-character quote openers of the pairs whose
// opener and closer contain no ' and no backtick.
//
//nolint:gochecknoglobals // built once
var cleanQuoteOpeners = sync.OnceValue(func() []cleanQuoteOpener {
	isClean := func(s string) bool { return !strings.ContainsAny(s, "'`") }

	var out []cleanQuoteOpener
	for i := range quotePairs {
		p := &quotePairs[i]
		if !isClean(p.open) || !isClean(p.close) {
			continue
		}
		c, size := utf8.DecodeRuneInString(p.open)
		if size == len(p.open) {
			out = append(out, cleanQuoteOpener{opener: c, pair: p})
		}
	}
	return out
})

// cleanOpenerPair returns the clean pair opened by c, or nil.
func cleanOpenerPair(c rune) *quotePair {
	for _, o := range cleanQuoteOpeners() {
		if o.opener == c {
			return o.pair
		}
	}
	return nil
}

// scanUnambiguousQuotes is the fast path of collectQuoteRanges, for a text
// with no ', no backtick and no 0xE3 lead byte (the CJK variants).
func scanUnambiguousQuotes(text string, out []skippableRange) []skippableRange {
	cursor := 0
	// A " or the lead byte 0xC2 (guillemets like «) or 0xE2 (curly quotes).
	for {
		rel := indexAnyByte(text[cursor:], '"', 0xC2, 0xE2)
		if rel < 0 {
			return out
		}
		opener := cursor + rel

		c, size := utf8.DecodeRuneInString(text[opener:])

		pair := cleanOpenerPair(c)
		if pair == nil {
			cursor = opener + size
			continue
		}

		contentStart := opener + len(pair.open)
		if off := strings.Index(text[contentStart:], pair.close); off >= 0 {
			end := contentStart + off + len(pair.close)
			out = append(out, newQuoteRange(opener, end, pair))
			cursor = end
		} else {
			// An opener with no closer.
			cursor = contentStart
		}
	}
}

// indexAnyByte returns the offset of the first byte of s equal to a, b or c,
// or -1.
func indexAnyByte(s string, a, b, c byte) int {
	for i := range len(s) {
		if s[i] == a || s[i] == b || s[i] == c {
			return i
		}
	}
	return -1
}

// isSymmetricQuoteRange reports whether r is a quote range of a symmetric
// pair.
func isSymmetricQuoteRange(r *skippableRange) bool {
	return r.quotePair != nil && r.quotePair.open == r.quotePair.close
}

// parityCache remembers the whole-paragraph count parity of the last
// symmetric quote token asked about.
type parityCache struct {
	token string
	odd   bool
	set   bool
}

func (c *parityCache) tokenCountIsOdd(paragraph, token string) bool {
	if c.set && c.token == token {
		return c.odd
	}
	odd := strings.Count(paragraph, token)%2 == 1
	c.token, c.odd, c.set = token, odd, true
	return odd
}

// symmetricTokenCountIsOdd reports whether the paragraph holds an odd number
// of the symmetric quote token that opens r. An odd count has at least one
// orphan, and when that orphan sits before a real downstream opener, the
// quote matcher pairs across a real sentence break.
func symmetricTokenCountIsOdd(paragraph string, r *skippableRange, cache *parityCache) bool {
	if r.quotePair == nil {
		return false
	}
	return cache.tokenCountIsOdd(paragraph, r.quotePair.open)
}

// parenContaining returns the paren range p with p.start < x < p.end, or nil.
// parens must be sorted and disjoint.
func parenContaining(parens []skippableRange, x int) *skippableRange {
	i := partitionPoint(len(parens), func(i int) bool { return parens[i].start < x })
	if i == 0 {
		return nil
	}
	if p := &parens[i-1]; x < p.end {
		return p
	}
	return nil
}

// quotePartiallyOverlapsParens reports whether quote straddles an edge of a
// paren range, one end in and one out. Full containment does not count.
// parens must be sorted by start and disjoint.
func quotePartiallyOverlapsParens(quote *skippableRange, parens []skippableRange) bool {
	if p := parenContaining(parens, quote.start); p != nil && p.end < quote.end {
		return true
	}
	if p := parenContaining(parens, quote.end); p != nil && p.start > quote.start {
		return true
	}
	return false
}

// peelLeadingSymmetricQuote trims leading whitespace from s and, when a
// symmetric quote closer follows, peels one closer and its trailing
// whitespace. It looks through a stray closing apostrophe when checking for a
// comma continuation, as in ". ' , Tim ...".
func peelLeadingSymmetricQuote(s string) string {
	trimmed := trimStart(s)
	if trimmed == "" {
		return trimmed
	}

	// Fast check for ASCII quotes.
	if first := trimmed[0]; first < utf8.RuneSelf && first != '\'' && first != '"' && first != '`' {
		return trimmed
	}

	for i := range quotePairs {
		p := &quotePairs[i]
		if p.open != p.close {
			continue
		}
		if rest, ok := strings.CutPrefix(trimmed, p.close); ok {
			return trimStart(rest)
		}
	}
	return trimmed
}

// isInQuoteRange reports whether idx lies inside a quote range of ranges.
func isInQuoteRange(ranges []skippableRange, idx int) bool {
	pos := partitionPoint(len(ranges), func(i int) bool { return ranges[i].start <= idx })
	for i := range pos {
		if ranges[i].isQuote() && idx < ranges[i].end {
			return true
		}
	}
	return false
}

// neighbors returns the characters just before idx and just after
// idx+len(token), each with false at the edge of the text.
func neighbors(text string, idx int, token string) (rune, bool, rune, bool) {
	prev, hasPrev := lastRune(text[:idx])
	next, hasNext := firstRune(text[idx+len(token):])
	return prev, hasPrev, next, hasNext
}

// isContractionQuote reports whether the token at idx is a contraction, like
// wasn't: sandwiched between two alphanumerics, it is not a quote.
func isContractionQuote(text string, idx int, token string) bool {
	prev, hasPrev, next, hasNext := neighbors(text, idx, token)
	return hasPrev && isAlphanumeric(prev) && hasNext && isAlphanumeric(next)
}

func isQuoteCandidate(text string, idx int, token string, ranges []skippableRange) bool {
	return !isInQuoteRange(ranges, idx) && !isContractionQuote(text, idx, token)
}

// isOpenerShape reports whether the ' or backtick at idx follows the start of
// the text or whitespace and precedes whitespace.
func isOpenerShape(text string, idx int, token string) bool {
	prev, hasPrev, next, hasNext := neighbors(text, idx, token)
	return (!hasPrev || isWhitespace(prev)) && hasNext && isWhitespace(next)
}

// startsNewUtterance reports whether text[from:] starts with one or more ASCII
// blanks followed by an ASCII uppercase letter.
func startsNewUtterance(text string, from int) bool {
	tail := text[from:]
	trimmed := strings.TrimLeft(tail, " \t")
	return len(trimmed) < len(tail) && trimmed != "" && isASCIIUpper(rune(trimmed[0]))
}

// candidateOpensCleanSpan reports whether a quote lies ahead with no inline
// sentence break in between.
func candidateOpensCleanSpan(text string, idx int, token string, ranges []skippableRange) bool {
	if !isOpenerShape(text, idx, token) {
		return false
	}

	contentStart := idx + len(token)
	for _, rel := range matchIndices(text[contentStart:], token) {
		if y := contentStart + rel; isQuoteCandidate(text, y, token, ranges) {
			return !hasPossibleInlineSentenceBreak(text[contentStart:y])
		}
	}
	return false
}

// quoteCandidatesShouldPair reports whether the ' or backtick candidates at
// opener and closer pair into a quote range. opener must be opener shaped.
// The pair is rejected only when the span looks like two back-to-back
// utterances and the closer itself cleanly opens a further quote ahead.
func quoteCandidatesShouldPair(text string, opener, closer int, token string, ranges []skippableRange) bool {
	if !isOpenerShape(text, opener, token) {
		return false
	}

	span := text[opener+len(token) : closer]
	backToBack := startsNewUtterance(text, closer+len(token)) && hasPossibleInlineSentenceBreak(span)

	return !backToBack || !candidateOpensCleanSpan(text, closer, token, ranges)
}

// tagQuoteMispairing labels each symmetric-pair quote range with its
// mispairing.
func tagQuoteMispairing(paragraph string, ranges []skippableRange) {
	var cache parityCache

	var parens []skippableRange
	for _, r := range ranges {
		if r.rangeType == rangeParentheses {
			parens = append(parens, r)
		}
	}

	for i := range ranges {
		slot := &ranges[i]
		if !slot.isQuote() {
			continue
		}

		r := *slot

		var class quoteMispairing
		switch {
		case !isSymmetricQuoteRange(&r):
			class = mispairingNone
		case quotePartiallyOverlapsParens(&r, parens):
			class = mispairingCertain
		case symmetricTokenCountIsOdd(paragraph, &r, &cache):
			class = mispairingPossible
		default:
			class = mispairingNone
		}

		slot.quoteMispairing = class
	}
}

// isSymmetricQuoteMispairing reports whether this symmetric-pair quote range
// (”, ', ") is a likely quote matcher mispairing across a sentence break:
//   - certain: the range straddles a paren boundary, and the override always
//     fires, since a quote pair should not cross a paren edge;
//   - possible: the paragraph has an odd token count (one or more orphans),
//     and the override fires only if [start, end) also looks like a strong
//     sentence break;
//   - none: not a symmetric pair, or evenly balanced; no override.
func isSymmetricQuoteMispairing(l *language, paragraph string, r *skippableRange, start, end int) bool {
	switch r.quoteMispairing {
	case mispairingCertain:
		return true
	case mispairingPossible:
		return l.hasStrongSentenceBreak(paragraph, start, end)
	default:
		return false
	}
}

// appendSpacePaddedQuotePairs appends the '…' and `…` ranges the quote matcher
// could not pair. The guarded alternatives require a word boundary right
// after the opener, so space-padded openers like "' word " go unpaired even
// when they form a real ' … ' pair.
func appendSpacePaddedQuotePairs(text string, ranges []skippableRange) []skippableRange {
	// Fast check.
	if !strings.ContainsAny(text, "'`") {
		return ranges
	}

	for i := range quotePairs {
		pair := &quotePairs[i]
		if !pair.ambiguous || pair.open != pair.close {
			continue
		}

		// Ambiguous symmetric tokens are single-byte ASCII.
		token := pair.close
		if strings.IndexByte(text, token[0]) < 0 {
			continue
		}

		pending := -1
		for _, idx := range matchIndices(text, token) {
			if !isQuoteCandidate(text, idx, token, ranges) {
				continue
			}

			if pending >= 0 && quoteCandidatesShouldPair(text, pending, idx, token, ranges) {
				ranges = append(ranges, newQuoteRange(pending, idx+len(token), pair))
				pending = -1
			} else {
				pending = idx
			}
		}
	}
	return ranges
}

// closerOpensNextSentence reports whether the symmetric closer at boundary
// opens the next sentence rather than trailing the current one, so the caller
// keeps the boundary and lets the closer join what follows.
func closerOpensNextSentence(paragraph string, boundary int, closer string, ranges []skippableRange) bool {
	// A symmetric quote with whitespace or the start of the text on its left,
	// leading into a capitalized word. "Coast.'" fails as its . is not clean.
	prev, hasPrev := lastRune(paragraph[:boundary])
	openerShaped := isSymmetricQuoteCloser(closer) &&
		(!hasPrev || isWhitespace(prev)) &&
		startsNewUtterance(paragraph, boundary+len(closer))
	if !openerShaped {
		return false
	}

	// The same token must also have formed a real pair earlier: a sign it is
	// an opener, not a stray closer. "We do ? ''" has no earlier opener.
	for i := range ranges {
		r := &ranges[i]
		if r.end <= boundary && isSymmetricQuoteRange(r) && strings.HasPrefix(paragraph[r.start:], closer) {
			return true
		}
	}
	return false
}

// orphanCloserPositions records, in ascending order, where the symmetric
// closing quote tokens of a paragraph appear (every ', say). Only real quote
// candidates are kept: contraction apostrophes and marks already inside a
// paired quote are skipped. isOrphanCloser then counts how many fall before
// or after a boundary with a binary search instead of rescanning the
// paragraph on every call.
type orphanCloserPositions struct {
	token     string
	hasToken  bool
	positions []int
}

func (o *orphanCloserPositions) reset() {
	o.hasToken = false
}

// beforeAndTotal returns how many candidate occurrences of token sit strictly
// before boundary (a byte offset), and how many there are in the whole
// paragraph.
func (o *orphanCloserPositions) beforeAndTotal(
	paragraph, token string,
	boundary int,
	ranges []skippableRange,
) (int, int) {
	if !o.hasToken || o.token != token {
		o.positions = o.positions[:0]
		for _, idx := range matchIndices(paragraph, token) {
			if isQuoteCandidate(paragraph, idx, token, ranges) {
				o.positions = append(o.positions, idx)
			}
		}
		o.token, o.hasToken = token, true
	}

	before := partitionPoint(len(o.positions), func(i int) bool { return o.positions[i] < boundary })
	return before, len(o.positions)
}

// rangeStartsAt reports whether some skippable range starts exactly at
// boundary.
func rangeStartsAt(ranges []skippableRange, boundary int, region nonListRegion) bool {
	if !region.binarySearch {
		for i := range ranges {
			if ranges[i].start == boundary {
				return true
			}
		}
		return false
	}

	nonList := ranges[:region.len]
	idx := partitionPoint(len(nonList), func(i int) bool { return nonList[i].start < boundary })
	if idx < len(nonList) && nonList[idx].start == boundary {
		return true
	}
	for _, r := range ranges[region.len:] {
		if r.start == boundary {
			return true
		}
	}
	return false
}

// isInnerTerminator reports whether boundary lies just before the closing mark
// of this quote. It is an opinionated rule that resolves some real-world
// cases of erroneous or ambiguous punctuation: in He said "Hello." the . is
// immediately followed by ", and the sentence should extend past the quote
// rather than break.
func isInnerTerminator(r *skippableRange, text string, boundary int) bool {
	if !r.isQuote() || boundary >= r.end {
		return false
	}

	head := text[:r.end]
	for _, c := range quoteClosersByLen() {
		if strings.HasSuffix(head, c) && boundary+len(c) == r.end {
			return true
		}
	}
	return false
}

// innerTerminatorBoundary returns the offset just past the closer of r where
// the sentence resumes, when boundary is a terminator at the inner edge of the
// range (the . in "... end."), and false otherwise.
func innerTerminatorBoundary(l *language, paragraph string, r *skippableRange, boundary int) (int, bool) {
	if !isInnerTerminator(r, paragraph, boundary) {
		return 0, false
	}

	nextWord := l.getNextWordApprox(paragraph, r.end)
	extend, ok := l.getBoundaryExtend(nextWord)
	if !ok {
		return 0, false
	}
	return r.end + extend, true
}

// extendPastOrphanCloser advances a boundary that sits at an orphan trailing
// quote closer (.' with no opener the quote matcher captured) past the
// closer, the whitespace after it and any stranded terminators that would
// otherwise form a sentence of their own. Otherwise it returns boundary
// unchanged.
func extendPastOrphanCloser(
	l *language,
	paragraph string,
	boundary int,
	ranges []skippableRange,
	region nonListRegion,
	orphanClosers *orphanCloserPositions,
) int {
	// When the next character opens a known quoted range, that quote belongs
	// to the upcoming sentence: do nothing.
	if rangeStartsAt(ranges, boundary, region) {
		return boundary
	}

	// Find an orphan closer starting at boundary, longest first so '' wins
	// over '.
	closer := ""
	for _, c := range quoteClosersByLen() {
		if strings.HasPrefix(paragraph[boundary:], c) &&
			isOrphanCloser(paragraph, boundary, c, ranges, orphanClosers) {
			closer = c
			break
		}
	}
	if closer == "" {
		return boundary
	}

	// When this orphan closer really opens the next sentence, do nothing.
	if closerOpensNextSentence(paragraph, boundary, closer, ranges) {
		return boundary
	}

	advancePastSpace := func(pos int) int {
		if m := spaceAfterSeparator.FindStringIndex(paragraph[pos:]); m != nil {
			return pos + m[1]
		}
		return pos
	}

	boundary = advancePastSpace(boundary + len(closer))
	re := l.getSentenceBreakRegex()

	// Absorb any stranded terminators.
	for {
		m := re.FindStringIndex(paragraph[boundary:])
		if m == nil || m[0] != 0 {
			break
		}
		boundary = advancePastSpace(boundary + m[1])
	}

	return boundary
}
