package sentencex

import (
	"cmp"
	"math"
	"slices"
	"strings"
	"unicode/utf8"
)

// List-item detector for line starts and inline positions.
//
// It scans each paragraph once, line by line, classifying markers both at
// line starts and inline (after a whitespace run). A "list is a sequence"
// sibling rule, with a single winning family per paragraph, keeps false
// positives down. The caller uses the returned offsets to emit sentence
// boundaries and to mark each item span as a list-item skippable range.
//
// Characters are decoded only where Unicode is unavoidable; everything else
// is parsed as bytes.

// unicodeBullets are the bullet characters of the first tier.
//
//nolint:gochecknoglobals // fixed lookup table
var unicodeBullets = [...]rune{
	'•', '◦', '▪', '▫', '■', '□', '●', '○', '⁃', '⁌', '⁍', '◆', '◇', '★', '☆', '➤', '➢', '➣', '▶',
	'▸', '►',
}

// minGapBytes is the minimum byte distance between two same-family candidates
// before the second is kept. Real list items carry content between markers,
// so genuine siblings sit far apart; tightly packed repeats are prose
// patterns (e. e. cummings: two letter-dot hits about 3 bytes apart). 4 is the
// smallest value that drops those while still admitting short real items
// like "1. x" (gap 4).
const minGapBytes = 4

// markerFamily is the kind of a list marker.
type markerFamily int

const (
	familyTier1       markerFamily = iota // Unicode bullets and parenthesized forms: fire on a single line-start match.
	familyBullet                          // * + -
	familyNumeric                         // 1. 1) 1.) 23.
	familyLetterParen                     // a) A) a.) A.)
	familyLetterDot                       // a. b. (lowercase only)
	familyRoman                           // ii. iii) ii.) (two or more letters, and promoted single letters)

	familyCount
)

// allowedInline reports whether the family may match inline. ASCII bullets
// (*, +, -) have no closing punctuation, so inline they are indistinguishable
// from parenthetical or hyphen uses ("Su - 24", "fast - track"), and two such
// uses would otherwise satisfy the sibling rule and emit false boundaries.
// Line-start bullets go through classifyLine and are unaffected. This mirrors
// the exclusion of en and em dashes in matchASCIIBullet.
func (f markerFamily) allowedInline() bool {
	return f != familyBullet
}

// listCandidate is a list marker found at pos.
type listCandidate struct {
	pos       int
	family    markerFamily
	firstByte byte
	lineStart bool
}

// isRomanLetterShape reports whether the candidate is a single-letter a. or
// a) whose letter could read as Roman (i, v, x, l, c, d, m). Used by
// promoteRomanLetters.
func (c *listCandidate) isRomanLetterShape() bool {
	return (c.family == familyLetterDot || c.family == familyLetterParen) && isRomanByte(c.firstByte)
}

// markerHit is a classified marker.
type markerHit struct {
	family    markerFamily
	firstByte byte
}

// detectListItems returns the byte offsets within paragraph of the accepted
// list-item starts, in source order with no duplicates.
func detectListItems(paragraph string) []int {
	candidates := make([]listCandidate, 0, max(len(paragraph)/50, 1))
	lineStart := 0

	for {
		nl := strings.IndexByte(paragraph[lineStart:], '\n')
		if nl < 0 {
			break
		}
		end := lineStart + nl + 1
		candidates = scanLine(paragraph, lineStart, end, candidates)
		lineStart = end
	}

	if lineStart < len(paragraph) {
		candidates = scanLine(paragraph, lineStart, len(paragraph), candidates)
	}

	return finalize(candidates)
}

func scanLine(text string, lineStart, lineEnd int, out []listCandidate) []listCandidate {
	contentStart := firstNonWS(text, lineStart, lineEnd)

	if hit, ok := classifyLine(text[contentStart:lineEnd]); ok {
		out = append(out, listCandidate{pos: lineStart, family: hit.family, firstByte: hit.firstByte, lineStart: true})
	}

	// Fast check for ) and the Unicode bullets, which all have the 0xE2 lead
	// byte.
	if strings.IndexByte(text[contentStart:lineEnd], ')') < 0 && strings.IndexByte(text[contentStart:lineEnd], 0xE2) < 0 {
		return out
	}

	// Each whitespace run is a candidate boundary: the byte after it may begin
	// an inline list marker.
	cursor := contentStart
	for {
		rel := strings.IndexAny(text[cursor:lineEnd], " \t")
		if rel < 0 {
			break
		}
		pos := firstNonWS(text, cursor+rel, lineEnd)
		if pos >= lineEnd || text[pos] == '\n' || text[pos] == '\r' {
			break
		}

		if hit, ok := classifyMarkerAt(text[pos:lineEnd]); ok {
			out = append(out, listCandidate{pos: pos, family: hit.family, firstByte: hit.firstByte})
		}

		cursor = pos
	}
	return out
}

// firstNonWS returns the index of the first byte of text[start:end] that is
// not a space or a tab, or end.
func firstNonWS(text string, start, end int) int {
	i := start
	for i < end && isHorizWS(text[i]) {
		i++
	}
	return i
}

// classifyLine classifies the start of a line, leading indent already
// stripped. It requires a marker followed by at least one space and real
// content.
func classifyLine(line string) (markerHit, bool) {
	if line == "" {
		return markerHit{}, false
	}
	family, n, ok := consumeMarker(line)
	if !ok {
		return markerHit{}, false
	}
	if _, ok := nextContentChar(line[n:]); !ok {
		return markerHit{}, false
	}
	return markerHit{family: family, firstByte: line[0]}, true
}

// classifyMarkerAt classifies a marker found inline, after a whitespace run.
// It is stricter than classifyLine because inline markers compete with prose
// punctuation.
func classifyMarkerAt(s string) (markerHit, bool) {
	if s == "" {
		return markerHit{}, false
	}
	family, n, ok := consumeMarker(s)
	if !ok {
		return markerHit{}, false
	}

	if isBareDotCloser(s, family, n) {
		return markerHit{}, false
	}

	next, ok := nextContentChar(s[n:])
	if !ok || isLowercase(next) {
		return markerHit{}, false
	}

	if !family.allowedInline() {
		return markerHit{}, false
	}

	return markerHit{family: family, firstByte: s[0]}, true
}

// isBareDotCloser reports a bare-dot closer (1., a., ii.), which collides with
// sentence-ending periods in prose. Inline markers must use ) or .). Line-start
// markers are unaffected: the line break itself is the structural signal.
func isBareDotCloser(s string, family markerFamily, markerLen int) bool {
	return s[markerLen-1] == '.' &&
		(family == familyNumeric || family == familyLetterDot || family == familyRoman)
}

// nextContentChar returns the first character after the spaces and tabs that
// open afterMarker, only when at least one space or tab is present and the
// character is real content (not a line terminator).
func nextContentChar(afterMarker string) (rune, bool) {
	n := firstNonWS(afterMarker, 0, len(afterMarker))
	if n == 0 {
		return 0, false
	}

	c, ok := firstRune(afterMarker[n:])
	if !ok || c == '\n' || c == '\r' {
		return 0, false
	}
	return c, true
}

// consumeMarker returns the family and byte length of the marker s starts
// with. The order encodes priority: tier 1 first, and multi-letter Roman
// before single-letter [a-z]\. so ii. is not classified as a letter-dot.
func consumeMarker(s string) (markerFamily, int, bool) {
	if n, ok := matchUnicodeBullet(s); ok {
		return familyTier1, n, true
	}
	if n, ok := matchParenForm(s); ok {
		return familyTier1, n, true
	}
	if n, ok := matchRoman(s); ok {
		return familyRoman, n, true
	}
	if n, ok := matchNumeric(s); ok {
		return familyNumeric, n, true
	}
	if n, ok := matchASCIIBullet(s); ok {
		return familyBullet, n, true
	}
	if n, ok := matchLetterParen(s); ok {
		return familyLetterParen, n, true
	}
	if n, ok := matchLetterDot(s); ok {
		return familyLetterDot, n, true
	}
	return 0, 0, false
}

func matchUnicodeBullet(s string) (int, bool) {
	// Fast check: the Unicode bullets all have the 0xE2 lead byte.
	if s == "" || s[0] != 0xE2 {
		return 0, false
	}
	c, size := utf8.DecodeRuneInString(s)
	if slices.Contains(unicodeBullets[:], c) {
		return size, true
	}
	return 0, false
}

// matchASCIIBullet matches *, + and -. En and em dashes (U+2013, U+2014) are
// intentionally left out: they collide with parenthetical dashes in prose,
// such as a Kazakh year range in parentheses, where the sibling rule would
// falsely activate a list. Real line-start dash bullets are rare, and * or -
// serve instead.
func matchASCIIBullet(s string) (int, bool) {
	if s != "" && (s[0] == '*' || s[0] == '+' || s[0] == '-') {
		return 1, true
	}
	return 0, false
}

func matchNumeric(s string) (int, bool) {
	n := 0
	for n < len(s) && s[n] >= '0' && s[n] <= '9' {
		n++
	}
	if n == 0 {
		return 0, false
	}
	cl, ok := closerLenAt(s, n)
	return n + cl, ok
}

func matchRoman(s string) (int, bool) {
	n := 0
	for n < len(s) && isRomanByte(s[n]) {
		n++
	}
	if n < 2 {
		return 0, false
	}
	cl, ok := closerLenAt(s, n)
	return n + cl, ok
}

// matchLetterParen accepts a) and a.) but not a. alone, which is the
// letter-dot family's.
func matchLetterParen(s string) (int, bool) {
	if len(s) < 2 || !isASCIIAlphabetic(s[0]) {
		return 0, false
	}
	switch {
	case s[1] == ')':
		return 2, true
	case s[1] == '.' && len(s) > 2 && s[2] == ')':
		return 3, true
	default:
		return 0, false
	}
}

func matchLetterDot(s string) (int, bool) {
	if len(s) >= 2 && s[0] >= 'a' && s[0] <= 'z' && s[1] == '.' {
		return 2, true
	}
	return 0, false
}

// matchParenForm matches (1), (12), (a), (A), (ii), (iv): short, unpadded
// inners only. Padded inners ("( 1894 )") and inners of three or more digits
// ("(1894)") are prose year or date citations, not list markers.
func matchParenForm(s string) (int, bool) {
	inside, ok := strings.CutPrefix(s, "(")
	if !ok {
		return 0, false
	}
	closeAt := strings.IndexByte(inside, ')')
	if closeAt < 0 {
		return 0, false
	}
	inner := inside[:closeAt]

	isDigit := func(b byte) bool { return b >= '0' && b <= '9' }
	shortNumeric := len(inner) >= 1 && len(inner) <= 2 && allBytes(inner, isDigit)
	singleLetter := len(inner) == 1 && isASCIIAlphabetic(inner[0])
	roman := inner != "" && allBytes(inner, isRomanByte)

	if shortNumeric || singleLetter || roman {
		return closeAt + 2, true
	}
	return 0, false
}

// closerLenAt returns the length of the marker closer at pos: ., ) or .).
func closerLenAt(s string, pos int) (int, bool) {
	if pos >= len(s) {
		return 0, false
	}
	switch s[pos] {
	case '.':
		if pos+1 < len(s) && s[pos+1] == ')' {
			return 2, true
		}
		return 1, true
	case ')':
		return 1, true
	default:
		return 0, false
	}
}

func allBytes(s string, pred func(byte) bool) bool {
	for i := range len(s) {
		if !pred(s[i]) {
			return false
		}
	}
	return true
}

func isASCIIAlphabetic(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z'
}

func isHorizWS(b byte) bool {
	return b == ' ' || b == '\t'
}

func isRomanByte(b byte) bool {
	if b >= 'A' && b <= 'Z' {
		b += 'a' - 'A'
	}
	switch b {
	case 'i', 'v', 'x', 'l', 'c', 'd', 'm':
		return true
	default:
		return false
	}
}

func finalize(candidates []listCandidate) []int {
	if len(candidates) == 0 {
		return nil
	}

	candidates = sortAndDedup(candidates)
	promoteRomanLetters(candidates)
	candidates = gapPrunePerFamily(candidates)

	winner, ok := pickWinner(candidates)
	if !ok {
		return nil
	}

	var out []int
	for _, c := range candidates {
		if c.family == winner {
			out = append(out, c.pos)
		}
	}
	return out
}

// sortAndDedup sorts by position, preferring the line-start candidate on a tie
// (rare, degenerate inputs only), and keeps one candidate per position.
func sortAndDedup(candidates []listCandidate) []listCandidate {
	slices.SortStableFunc(candidates, func(a, b listCandidate) int {
		if c := cmp.Compare(a.pos, b.pos); c != 0 {
			return c
		}
		switch {
		case a.lineStart == b.lineStart:
			return 0
		case a.lineStart:
			return -1
		default:
			return 1
		}
	})
	return slices.CompactFunc(candidates, func(a, b listCandidate) bool { return a.pos == b.pos })
}

// promoteRomanLetters reclassifies the single-letter, Roman-shaped letter-dot
// and letter-paren candidates as Roman when any multi-letter Roman exists, so
// they count as siblings (i.) ... ii.)).
func promoteRomanLetters(candidates []listCandidate) {
	if !slices.ContainsFunc(candidates, func(c listCandidate) bool { return c.family == familyRoman }) {
		return
	}
	for i := range candidates {
		if candidates[i].isRomanLetterShape() {
			candidates[i].family = familyRoman
		}
	}
}

// gapPrunePerFamily drops the candidates too close to the previous match of
// the same family. A real list item carries content; e. e. cummings
// (letter-dot gap 3) does not. It runs before the family selection so the
// sibling counts reflect the pruned set.
func gapPrunePerFamily(candidates []listCandidate) []listCandidate {
	var familyLast [familyCount]int
	for i := range familyLast {
		familyLast[i] = math.MaxInt
	}

	return slices.DeleteFunc(candidates, func(c listCandidate) bool {
		last := familyLast[c.family]
		keep := last == math.MaxInt || c.pos-last >= minGapBytes
		if keep {
			familyLast[c.family] = c.pos
		}
		return !keep
	})
}

// pickWinner chooses the winning family of the paragraph, and false for no
// list. Tier 1 (Unicode bullets, parenthesized forms) wins on a single
// line-start match or two matches anywhere. Otherwise the tier 2 family with
// the most matches wins, the earliest first match breaking ties.
func pickWinner(candidates []listCandidate) (markerFamily, bool) {
	var counts [familyCount]int
	var firstPos [familyCount]int
	for i := range firstPos {
		firstPos[i] = math.MaxInt
	}
	tier1AtLineStart := false

	for _, c := range candidates {
		counts[c.family]++
		if firstPos[c.family] == math.MaxInt {
			firstPos[c.family] = c.pos
		}
		if c.family == familyTier1 && c.lineStart {
			tier1AtLineStart = true
		}
	}

	if tier1AtLineStart || counts[familyTier1] >= 2 {
		return familyTier1, true
	}

	tier2 := [...]markerFamily{familyBullet, familyNumeric, familyLetterParen, familyLetterDot, familyRoman}

	// The maximum of (count, -first position); on equal keys the last wins.
	var best markerFamily
	found := false
	for _, f := range tier2 {
		if counts[f] < 2 {
			continue
		}
		if !found || counts[f] > counts[best] || counts[f] == counts[best] && firstPos[f] <= firstPos[best] {
			best, found = f, true
		}
	}
	return best, found
}
