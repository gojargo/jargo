package sentencex

import (
	"cmp"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"
)

// Compiled regular expressions shared by the language rules.
//
//nolint:gochecknoglobals // compiled once
var (
	// defaultSentenceBreakRegex matches a run of sentence terminators.
	//
	// Branch 1 (\.(?:[ \t]+\.){2,}) coalesces three or more spaced dots
	// (". . .", ". . . .") into one match. Two-dot ". ." is excluded so a
	// period followed by a leading ellipsis ("raak. ...en") is not eaten as a
	// single run. [ \t] (not \s) keeps newlines intact for paragraph splits.
	//
	// Branch 2 ([!?…](?:[ \t]+[!?…])+) coalesces two or more spaced runs,
	// mixed or homogeneous ("! !", "? ? ?", "! ?", "… !"). + rather than {2,}
	// is safe here: there is no leading-ellipsis equivalent for ! or ?.
	// Both branches must precede the class for leftmost-first alternation.
	defaultSentenceBreakRegex = regexp.MustCompile(
		`\.(?:[ \t]+\.){2,}|[!?…](?:[ \t]+[!?…])+|[` + string(globalSentenceTerminators[:]) + `]+`)

	// continueAfterNonwordRegex matches a lowercase letter or digit, optionally
	// preceded by non-word characters (a space or punctuation). It is the
	// pattern ^\W*[0-9a-z] with the Unicode meaning of \W.
	continueAfterNonwordRegex = regexp.MustCompile(`^[^` + perlWordClass + `]*[0-9a-z]`)

	// ellipsisContinueRegex treats a multi-character terminator run as
	// mid-sentence when the follow-up is whitespace and a lowercase letter or
	// digit ("... no", ". . . what"). Languages with a capitalized word that
	// is ambiguous with a sentence start (English standalone I) extend this.
	ellipsisContinueRegex = regexp.MustCompile(`^[` + whitespaceClass + `]+[0-9a-z]`)
)

// minBinarySearchRanges is the number of skippable ranges above which a binary
// search beats a linear scan.
const minBinarySearchRanges = 64

// maxNextWordBytes caps the look-ahead of getNextWordApprox. If the longest
// special words (abbreviations, starters and so on) ever exceed it, raise it.
const maxNextWordBytes = 30

// startsWithASCIILowercaseOrDigit is the pattern ^[0-9a-z], checked on the
// first byte.
func startsWithASCIILowercaseOrDigit(s string) bool {
	return s != "" && (s[0] >= 'a' && s[0] <= 'z' || s[0] >= '0' && s[0] <= '9')
}

// paragraphBreakAt returns the byte range of the \n[\r]*\n paragraph separator
// that starts at at, and false when none starts there.
func paragraphBreakAt(text string, at int) (int, int, bool) {
	if at >= len(text) || text[at] != '\n' {
		return 0, 0, false
	}
	end := at + 1
	for end < len(text) && text[end] == '\r' {
		end++
	}
	if end < len(text) && text[end] == '\n' {
		return at, end + 1, true
	}
	return 0, 0, false
}

// paragraphBreaks returns the \n[\r]*\n paragraph separators of text as
// [start, end) byte ranges.
func paragraphBreaks(text string) [][2]int {
	var out [][2]int
	cursor := 0
	for {
		i := strings.IndexByte(text[cursor:], '\n')
		if i < 0 {
			return out
		}
		newline := cursor + i
		if start, end, ok := paragraphBreakAt(text, newline); ok {
			out = append(out, [2]int{start, end})
			cursor = end
			continue
		}
		// A lone \n: advance past it and keep scanning.
		cursor = newline + 1
	}
}

// isCodeLikeNumberedToken reports whether a . sits inside a code-like numbered
// token rather than ending a sentence: a digit immediately before, and an
// alphanumeric token with a digit after it with no space, as in the chess move
// 7.Bg5.
func isCodeLikeNumberedToken(head, nextWordApprox string) bool {
	if head == "" || !isASCIIDigit(rune(head[len(head)-1])) {
		return false
	}
	first, ok := firstRune(nextWordApprox)
	if !ok || !isAlphabetic(first) {
		return false
	}
	return strings.ContainsAny(nextWordApprox, "0123456789")
}

// isSingleASCIIUpper reports whether s is exactly one ASCII uppercase letter.
func isSingleASCIIUpper(s string) bool {
	return len(s) == 1 && isASCIIUpper(rune(s[0]))
}

// abbreviationSetContains reports whether set holds word, compared in
// lowercase.
func abbreviationSetContains(set map[string]struct{}, word string) bool {
	if isASCII(word) {
		if strings.IndexFunc(word, isASCIIUpper) < 0 {
			_, ok := set[word]
			return ok
		}
		_, ok := set[strings.ToLower(word)]
		return ok
	}
	_, ok := set[toLowercase(word)]
	return ok
}

// startsWithInitial reports whether s, after leading whitespace, begins with a
// name-initial token: a single uppercase ASCII letter, a ., then the end of
// the text or whitespace. "J. R. Tolkien" triggers; "Jones", "J.R.R." and
// "A.B" do not.
func startsWithInitial(s string) bool {
	s = trimStart(s)
	first, ok := firstRune(s)
	if !ok || !isASCIIUpper(first) {
		return false
	}
	s = s[utf8.RuneLen(first):]
	if second, hasSecond := firstRune(s); !hasSecond || second != '.' {
		return false
	}
	third, hasThird := firstRune(s[1:])
	return !hasThird || isWhitespace(third)
}

// pushIfIncreasing appends boundary only if it advances past the last
// recorded position. boundaries must not be empty.
func pushIfIncreasing(boundaries []int, boundary int) []int {
	if boundary > boundaries[len(boundaries)-1] {
		return append(boundaries, boundary)
	}
	return boundaries
}

// findTerminatorMatches finds the terminator-run matches in text, folding a
// whitespace-separated dot-only follow-up onto a preceding !, ? or … run, so
// "Bravo ! ." and "Happy! . . . no one …" surface as one coalesced
// terminator.
//
// The regular expression already coalesces homogeneous runs ("! !", ". . .")
// and contiguous mixed runs like "! ..." (matched by the contiguous-class
// branch as "!..."). Spaced mixed runs like "! . . ." cannot be expressed
// without lookahead ([!?…][ \t]+\. would eat the first dot of an ellipsis),
// so they arrive here as two matches and are folded only when the follow-up
// is pure dots separated by whitespace.
func findTerminatorMatches(text string, re *regexp.Regexp, out [][2]int) [][2]int {
	out = out[:0]

	// Faster path for ASCII text and the default regular expression.
	if re == defaultSentenceBreakRegex && isASCII(text) {
		return scanASCIIMatches(text, out)
	}

	for _, m := range re.FindAllStringIndex(text, -1) {
		out = foldMatch(out, text, m[0], m[1])
	}
	return out
}

// foldMatch appends [start, end) to out, or folds it into the previous match
// when shouldFold says the two form a single run like "Happy ! . . .".
func foldMatch(out [][2]int, text string, start, end int) [][2]int {
	if len(out) > 0 && shouldFold(text, out[len(out)-1], start, end) {
		out[len(out)-1][1] = end
		return out
	}
	return append(out, [2]int{start, end})
}

// shouldFold reports whether the match [start, end) folds onto the previous
// match: a whitespace-separated, dot-only run like ". . ." following an
// ending in !, ? or ….
func shouldFold(text string, prev [2]int, start, end int) bool {
	prevText := text[prev[0]:prev[1]]
	candidate := text[start:end]
	gap := text[prev[1]:start]

	isBlank := func(c rune) bool { return c == ' ' || c == '\t' }
	last, _ := lastRune(prevText)
	prevIsEmphatic := last == '!' || last == '?' || last == '…'
	candidateIsDotRun := strings.HasPrefix(candidate, ".") &&
		strings.IndexFunc(candidate, func(c rune) bool { return c != '.' && !isBlank(c) }) < 0
	separatedByBlanks := gap != "" && strings.IndexFunc(gap, func(c rune) bool { return !isBlank(c) }) < 0

	return prevIsEmphatic && candidateIsDotRun && separatedByBlanks
}

// scanASCIIMatches is the ASCII fast path of findTerminatorMatches, valid only
// for defaultSentenceBreakRegex:
//   - branch 1 \.(?:[ \t]+\.){2,} (three or more blank-separated dots),
//   - branch 2 [!?](?:[ \t]+[!?])+ (two or more blank-separated ! or ?),
//   - branch 3 [.!?]+ through contiguousRun.
func scanASCIIMatches(text string, out [][2]int) [][2]int {
	cursor := 0
	for {
		rel := strings.IndexAny(text[cursor:], ".!?")
		if rel < 0 {
			return out
		}
		p := cursor + rel

		var end int
		var ok bool
		if text[p] == '.' {
			end, ok = spacedRun(text, p, func(b byte) bool { return b == '.' }, 3)
		} else {
			end, ok = spacedRun(text, p, func(b byte) bool { return b == '!' || b == '?' }, 2)
		}
		if !ok {
			end = contiguousRun(text, p)
		}

		out = foldMatch(out, text, p, end)
		cursor = end
	}
}

// spacedRun scans a space- or tab-separated run of isMember bytes starting at
// p. It returns the offset just past the last member, and false when the run
// has fewer than minTotal members.
func spacedRun(text string, p int, isMember func(byte) bool, minTotal int) (int, bool) {
	count := 1
	end := p + 1
	for {
		k := end
		for k < len(text) && (text[k] == ' ' || text[k] == '\t') {
			k++
		}
		if k > end && k < len(text) && isMember(text[k]) {
			count++
			end = k + 1
		} else {
			break
		}
	}
	return end, count >= minTotal
}

// contiguousRun returns the end offset of the contiguous terminator run
// starting at p.
func contiguousRun(text string, p int) int {
	end := p + 1
	for end < len(text) && (text[end] == '.' || text[end] == '!' || text[end] == '?') {
		end++
	}
	return end
}

// paragraphScratch holds the buffers reused across the paragraphs of a text.
type paragraphScratch struct {
	sentenceBoundaries []int
	matches            [][2]int
	skippableRanges    []skippableRange
	orphanClosers      orphanCloserPositions
}

// nonListRegion describes the sorted, non-list part of a paragraph's
// skippable ranges: len is its number of ranges, and binarySearch whether
// there are enough of them to justify a binary search over them.
type nonListRegion struct {
	len          int
	binarySearch bool
}

// collectSentenceBreaks computes the byte offsets in paragraph where sentences
// break into scratch.sentenceBoundaries, skipping terminators inside quotes,
// parentheses and lists.
func collectSentenceBreaks(l *language, paragraph string, re *regexp.Regexp, scratch *paragraphScratch) {
	scratch.orphanClosers.reset()

	scratch.sentenceBoundaries = append(scratch.sentenceBoundaries[:0], 0)
	boundaries := scratch.sentenceBoundaries

	scratch.matches = findTerminatorMatches(paragraph, re, scratch.matches)
	scratch.skippableRanges = l.getSkippableRanges(paragraph, scratch.skippableRanges)

	nonListLen := len(scratch.skippableRanges)
	region := nonListRegion{
		len:          nonListLen,
		binarySearch: nonListLen > minBinarySearchRanges && !rangesOverlap(scratch.skippableRanges),
	}

	listStarts := detectListItems(paragraph)
	scratch.skippableRanges = addListItemRanges(scratch.skippableRanges, listStarts, len(paragraph))
	ranges := scratch.skippableRanges

	for _, m := range scratch.matches {
		boundary, ok := l.findBoundary(paragraph, m[0], m[1])
		if !ok {
			continue
		}

		var breakAt int
		if r := containingRange(l, paragraph, boundary, m[0], m[1], ranges, region); r != nil {
			breakAt, ok = innerTerminatorBoundary(l, paragraph, r, boundary)
		} else {
			breakAt = extendPastOrphanCloser(l, paragraph, boundary, ranges, region, &scratch.orphanClosers)
		}

		if ok {
			boundaries = pushIfIncreasing(boundaries, breakAt)
		}
	}

	boundaries = mergeListItemBoundaries(boundaries, listStarts)

	if boundaries[len(boundaries)-1] != len(paragraph) {
		boundaries = append(boundaries, len(paragraph))
	}
	scratch.sentenceBoundaries = boundaries
}

// rangesOverlap reports whether any of the start-sorted ranges overlaps
// another.
func rangesOverlap(startSortedRanges []skippableRange) bool {
	maxEnd := 0
	for _, r := range startSortedRanges {
		if r.start < maxEnd {
			return true
		}
		maxEnd = max(maxEnd, r.end)
	}
	return false
}

// selectContainingBinary binary searches the non-list region of ranges for the
// range satisfying isBreak, falling back to the appended list ranges. The
// non-list region must be sorted and free of overlaps.
func selectContainingBinary(
	ranges []skippableRange,
	nonListLen, boundary int,
	isBreak func(*skippableRange) bool,
) *skippableRange {
	nonList, list := ranges[:nonListLen], ranges[nonListLen:]

	if i := partitionPoint(len(nonList), func(i int) bool { return nonList[i].start < boundary }); i > 0 {
		if r := &nonList[i-1]; isBreak(r) {
			return r
		}
	}
	for i := range list {
		if isBreak(&list[i]) {
			return &list[i]
		}
	}
	return nil
}

// containingRange returns the first skippable range that genuinely encloses
// boundary: it contains the offset and is not a symmetric-quote mispairing
// (which only looks like containment). A range means the terminator at
// boundary is suppressed rather than split on.
func containingRange(
	l *language,
	paragraph string,
	boundary, matchStart, matchEnd int,
	ranges []skippableRange,
	region nonListRegion,
) *skippableRange {
	isBreak := func(r *skippableRange) bool {
		return r.contains(boundary) && !isSymmetricQuoteMispairing(l, paragraph, r, matchStart, matchEnd)
	}

	if region.binarySearch {
		return selectContainingBinary(ranges, region.len, boundary, isBreak)
	}
	for i := range ranges {
		if isBreak(&ranges[i]) {
			return &ranges[i]
		}
	}
	return nil
}

// addListItemRanges appends a skippable range for each list-item line span
// (the last one running to paragraphLen), so a terminator inside an item
// does not split it.
func addListItemRanges(ranges []skippableRange, listStarts []int, paragraphLen int) []skippableRange {
	for i := 0; i+1 < len(listStarts); i++ {
		ranges = append(ranges, newSkippableRange(listStarts[i], listStarts[i+1], rangeListItem))
	}
	if len(listStarts) > 0 {
		ranges = append(ranges, newSkippableRange(listStarts[len(listStarts)-1], paragraphLen, rangeListItem))
	}
	return ranges
}

// mergeListItemBoundaries adds each list-item line start as a sentence
// boundary, then sorts and removes duplicates.
func mergeListItemBoundaries(boundaries, listStarts []int) []int {
	if len(listStarts) == 0 {
		return boundaries
	}
	for _, start := range listStarts {
		if start > 0 {
			boundaries = append(boundaries, start)
		}
	}
	slices.Sort(boundaries)
	return slices.Compact(boundaries)
}

// continuesAfterBoundary is the shared rule of the languages that continue a
// sentence before a month name. It reports whether text starts with a
// lowercase letter or digit (after optional non-word characters), or whether
// its first whitespace-delimited word, as is or with its first character
// uppercased, is one of months.
func continuesAfterBoundary(text string, months []string) bool {
	if continueAfterNonwordRegex.MatchString(text) {
		return true
	}

	nextWord := ""
	if fields := strings.Fields(text); len(fields) > 0 {
		nextWord = fields[0]
	}
	nextWord = strings.Trim(nextWord, ".!?")
	if nextWord == "" {
		return false
	}

	first, size := utf8.DecodeRuneInString(nextWord)
	capitalized := charToUppercase(first) + nextWord[size:]

	return slices.Contains(months, nextWord) || slices.Contains(months, capitalized)
}

// skippableRangeType is the kind of region a skippableRange covers.
type skippableRangeType int

const (
	rangeQuote skippableRangeType = iota
	rangeParentheses
	rangeEmail
	rangeListItem
)

// skippableRange is a region of a paragraph whose terminators do not end a
// sentence.
type skippableRange struct {
	start           int
	end             int
	rangeType       skippableRangeType
	quotePair       *quotePair
	quoteMispairing quoteMispairing
}

func newSkippableRange(start, end int, rangeType skippableRangeType) skippableRange {
	return skippableRange{start: start, end: end, rangeType: rangeType}
}

func newQuoteRange(start, end int, pair *quotePair) skippableRange {
	return skippableRange{start: start, end: end, rangeType: rangeQuote, quotePair: pair}
}

// contains reports whether position lies strictly inside the range.
func (r *skippableRange) contains(position int) bool {
	return position > r.start && position < r.end
}

func (r *skippableRange) isQuote() bool {
	return r.rangeType == rangeQuote
}

// language holds the segmentation rules of one language. The zero value is a
// language with the default rules and no word lists. A language customizes a
// rule by setting the matching field; every rule that consults another goes
// through the method, so a customization applies throughout.
type language struct {
	// abbreviations are the known abbreviations, used to prevent false
	// sentence breaks at abbreviation periods ("Dr.", "etc.").
	abbreviations map[string]struct{}
	// sentenceStarters are safe sentence-opener words: function words and
	// auxiliaries that almost never appear capitalized mid-sentence, never
	// proper nouns.
	sentenceStarters map[string]struct{}
	// frontingWords are the words permitted in fronted adverbial phrases,
	// used by prefixIsPurelyFronting.
	frontingWords map[string]struct{}
	// trailingMarkers is the trailing-marker lookup table, or nil for none.
	trailingMarkers *markerTable
	// sentenceBreakRegex matches sentence-terminating punctuation, or is nil
	// for defaultSentenceBreakRegex.
	sentenceBreakRegex *regexp.Regexp

	// lastWord replaces the default getLastWord when set.
	lastWord func(text string) string
	// continueInNextWord replaces the default continueInNextWord when set.
	continueInNextWord func(textAfterBoundary string) bool
	// ellipsisContinuation replaces the default isEllipsisContinuation when
	// set.
	ellipsisContinuation func(textAfterRun string) bool
}

// getSentenceBreakRegex returns the regular expression that matches
// sentence-terminating punctuation.
func (l *language) getSentenceBreakRegex() *regexp.Regexp {
	if l.sentenceBreakRegex != nil {
		return l.sentenceBreakRegex
	}
	return defaultSentenceBreakRegex
}

// segment splits text into sentence and paragraph separator slices.
func (l *language) segment(text string) []string {
	capacity := max(len(text)/50, 1)
	sentences := make([]string, 0, capacity)
	scratch := &paragraphScratch{
		sentenceBoundaries: make([]int, 0, capacity),
		matches:            make([][2]int, 0, capacity),
		skippableRanges:    make([]skippableRange, 0, capacity),
	}
	re := l.getSentenceBreakRegex()

	// Walk each paragraph paired with its trailing separator (none after the
	// last paragraph).
	paraStart := 0
	separators := paragraphBreaks(text)
	for i := 0; i <= len(separators); i++ {
		paraEnd := len(text)
		if i < len(separators) {
			paraEnd = separators[i][0]
		}
		paragraph := text[paraStart:paraEnd]

		collectSentenceBreaks(l, paragraph, re, scratch)

		for j := 0; j+1 < len(scratch.sentenceBoundaries); j++ {
			if sentence := paragraph[scratch.sentenceBoundaries[j]:scratch.sentenceBoundaries[j+1]]; sentence != "" {
				sentences = append(sentences, sentence)
			}
		}

		if i < len(separators) {
			if separator := text[separators[i][0]:separators[i][1]]; separator != "" {
				sentences = append(sentences, separator)
			}
			paraStart = separators[i][1]
		}
	}

	return sentences
}

// getAbbreviationChar returns the character that marks abbreviations.
func (l *language) getAbbreviationChar() string {
	return "."
}

// getBoundaryExtend returns the byte length of the leading run of whitespace
// and terminators in word, and false when word continues the current
// sentence.
func (l *language) getBoundaryExtend(word string) (int, bool) {
	if l.isContinueInNextWord(strings.TrimSpace(word)) || continueAfterNonwordRegex.MatchString(word) {
		return 0, false
	}

	count := 0
	for _, ch := range word {
		if !isWhitespace(ch) && !isSentenceTerminator(ch) {
			break
		}
		count += utf8.RuneLen(ch)
	}
	return count, true
}

// isAbbreviation reports whether the potential boundary after head is really
// the end of a known abbreviation ("Dr. Smith", "etc.").
func (l *language) isAbbreviation(head, _, separator string) bool {
	return l.isAbbreviationFor(l.getLastWord(head), separator)
}

// isAbbreviationFor is isAbbreviation for a caller that already has the
// trailing word.
func (l *language) isAbbreviationFor(lastWord, separator string) bool {
	if l.getAbbreviationChar() != separator || lastWord == "" {
		return false
	}
	return abbreviationSetContains(l.abbreviations, lastWord)
}

// isNameInitialFor detects a name initial: a single uppercase ASCII letter
// followed by a period in a position that looks like part of a name. It
// holds when the token before it in head starts with an uppercase ASCII letter
// ("Albert I.", "George W.") or the token after it is itself an initial
// ("J. R. R. Tolkien", including the sentence-initial position where there is
// no preceding token). It is ASCII-only on both sides, so non-Latin scripts
// are unaffected.
func (l *language) isNameInitialFor(head, lastWord, nextWordApprox string) bool {
	if !isSingleASCIIUpper(lastWord) {
		return false
	}

	// Preceding-token rule: trim the initial and any separators getLastWord
	// splits on (whitespace, . and /), then take the trailing word of what is
	// left.
	prefix := strings.TrimRightFunc(head[:len(head)-len(lastWord)], func(c rune) bool {
		return isWhitespace(c) || c == '.' || c == '/'
	})

	if first, ok := firstRune(l.getLastWord(prefix)); ok && isASCIIUpper(first) {
		return true
	}

	return startsWithInitial(nextWordApprox)
}

// nextWordIsSentenceStarter reports whether the next token in nextWordApprox
// is a known sentence opener. A listed starter strongly signals the start of
// a new sentence and overrides the abbreviation and name-initial paths.
func (l *language) nextWordIsSentenceStarter(nextWordApprox string) bool {
	if len(l.sentenceStarters) == 0 {
		return false
	}

	trimmed := trimStart(nextWordApprox)
	wordEnd := strings.IndexFunc(trimmed, func(c rune) bool {
		return isWhitespace(c) || c == ',' || isSentenceTerminator(c)
	})
	if wordEnd < 0 {
		wordEnd = len(trimmed)
	}
	if wordEnd == 0 {
		return false
	}

	_, ok := l.sentenceStarters[trimmed[:wordEnd]]
	return ok
}

// shouldOverrideAbbrevSuppressionFor is a one-way override that keeps a
// boundary the abbreviation and name-initial path would otherwise suppress.
// It fires when the next word is a registered sentence starter and the
// trailing token:
//   - starts with an uppercase letter: initials ("I."), names ("Penn."),
//     acronyms ("BART."),
//   - is a known multi-dot abbreviation ("w.e.f."),
//   - is a multi-character lowercase abbreviation ("etc.", "man.").
func (l *language) shouldOverrideAbbrevSuppressionFor(head, lastWord string, nextIsStarter bool) bool {
	if !nextIsStarter {
		return false
	}

	if first, ok := firstRune(lastWord); ok && isASCIIUpper(first) {
		return true
	}

	if l.isMultiDotAbbreviation(head, len(lastWord)) {
		return true
	}

	_, ok := l.abbreviations[lastWord]
	return hasSecondRune(lastWord) && ok
}

// isMultiDotAbbreviation reports whether the trailing token of head is a
// multi-dot abbreviation listed in the abbreviation table ("w.e.f", "U.S.").
func (l *language) isMultiDotAbbreviation(head string, tailLen int) bool {
	lastWordFull := l.getLastWordFull(head)
	if len(lastWordFull) <= tailLen {
		return false
	}
	return abbreviationSetContains(l.abbreviations, lastWordFull)
}

// getLastWordFull is getLastWord keeping internal dots, so multi-dot
// abbreviations ("w.e.f", "U.S", "p.m") come back whole. It splits only on
// whitespace and /.
func (l *language) getLastWordFull(text string) string {
	return lastPieceAfter(trimEnd(text), func(c rune) bool { return isWhitespace(c) || c == '/' })
}

// getLastWord returns the last word of text, splitting on whitespace, periods
// and slashes. Abbreviation detection uses it on the text before a potential
// sentence boundary.
func (l *language) getLastWord(text string) string {
	if l.lastWord != nil {
		return l.lastWord(text)
	}
	// Trim trailing whitespace so a stray space before the terminator ("U.S .")
	// does not blank out the last word. / joins route names to abbreviations
	// ("171/U.S") without being a real word boundary, so split on it too.
	return lastPieceAfter(trimEnd(text), func(c rune) bool { return isWhitespace(c) || c == '.' || c == '/' })
}

// isExclamationFor reports whether lastWord is a known exclamation word
// ("Yahoo!") whose mark should not break the sentence.
func (l *language) isExclamationFor(lastWord string) bool {
	if lastWord == "" {
		return false
	}
	for _, w := range exclamationWords {
		if p, ok := strings.CutSuffix(w, "!"); ok && p == lastWord {
			return true
		}
	}
	return false
}

// hasStrongSentenceBreak reports whether the terminator at [start, end) looks
// like a confident sentence end: a single . whose preceding word is a
// symmetric quote closer (”, ", …) and whose follower starts with a capital
// letter, the closer + . + UpperWord shape. It is a structural escape valve
// for symmetric-pair quote ranges that the non-greedy quote matcher may have
// paired across a real boundary.
func (l *language) hasStrongSentenceBreak(paragraph string, start, end int) bool {
	if end-start != 1 || paragraph[start] != '.' {
		return false
	}

	nextWordApprox := l.getNextWordApprox(paragraph, start+1)
	trimmedNext := trimStart(nextWordApprox)
	if first, ok := firstRune(trimmedNext); !ok || !isASCIIUpper(first) {
		return false
	}

	head := paragraph[:start]
	lastWord := l.getLastWord(head)
	if lastWord == "" || isSingleASCIIUpper(lastWord) {
		return false
	}

	if l.isAbbreviation(head, lastWord, ".") || l.isMultiDotAbbreviation(head, len(lastWord)) {
		return false
	}

	if isSymmetricQuoteCloser(lastWord) {
		return true
	}

	return l.nextWordIsSentenceStarter(trimmedNext)
}

// getNextWordApprox returns an approximate substring of the next word or
// words from start, at most maxNextWordBytes long, rounded up to a character
// boundary.
func (l *language) getNextWordApprox(text string, start int) string {
	if start >= len(text) {
		return ""
	}
	end := min(start+maxNextWordBytes, len(text))
	for end < len(text) && !utf8.RuneStart(text[end]) {
		end++
	}
	return text[start:end]
}

// terminatorContinues reports whether the sentence continues past the
// terminator: a lowercase letter, digit or comma follows, an ellipsis
// continues, or a spaced ! or ? precedes a lowercase word.
func (l *language) terminatorContinues(matched, head, nextWordApprox string) bool {
	if hasSecondRune(matched) {
		if l.isEllipsisContinuation(nextWordApprox) {
			return true
		}
		last, ok := lastRune(head)
		return ok && !isWhitespace(last) && startsWithASCIILowercaseOrDigit(nextWordApprox)
	}

	if l.isContinueInNextWord(nextWordApprox) {
		return true
	}

	// For example "Father Came Too ! is a British comedy film".
	return (matched == "!" || matched == "?") &&
		head != "" && (head[len(head)-1] == ' ' || head[len(head)-1] == '\t') &&
		continueAfterNonwordRegex.MatchString(nextWordApprox)
}

// periodSuppressesBoundary reports whether a . terminator should be
// suppressed.
func (l *language) periodSuppressesBoundary(head, lastWord, nextWordApprox string) bool {
	suppress := l.isNameInitialFor(head, lastWord, nextWordApprox) || l.isAbbreviationFor(lastWord, ".")

	marker, hasMarker := classifyTrailingMarker(head, l.getTrailingMarkers())
	if !suppress && !hasMarker {
		return false
	}

	nextIsStarter := l.nextWordIsSentenceStarter(nextWordApprox)
	markerBypass := hasMarker && markerBypassesSuppression(marker, nextWordApprox, nextIsStarter, l)

	return !markerBypass && !l.shouldOverrideAbbrevSuppressionFor(head, lastWord, nextIsStarter)
}

// findBoundary decides where the sentence ends for the terminator match
// [start, end) of text, and returns false when it is not a boundary. It
// weighs abbreviations, exclamations, numbered references and continuation
// patterns, telling true sentence ends from false positives.
func (l *language) findBoundary(text string, start, end int) (int, bool) {
	head := text[:start]
	matched := text[start:end]
	nextWordApprox := l.getNextWordApprox(text, end)

	// Only run the regular expression when a [ is present.
	if strings.IndexByte(nextWordApprox, '[') >= 0 {
		if m := numberedReferenceRegex.FindStringIndex(nextWordApprox); m != nil {
			return end + m[1], true
		}
	}

	if l.terminatorContinues(matched, head, nextWordApprox) {
		return 0, false
	}

	lastWord := l.getLastWord(head)

	if matched == "." {
		if isCodeLikeNumberedToken(head, nextWordApprox) {
			return 0, false
		}
		if l.periodSuppressesBoundary(head, lastWord, nextWordApprox) {
			return 0, false
		}
	}

	if l.isExclamationFor(lastWord) {
		return 0, false
	}

	// Swallow any whitespace after the terminator into the boundary.
	trailingWhitespace := len(nextWordApprox) - len(trimStart(nextWordApprox))
	return end + trailingWhitespace, true
}

// isEllipsisContinuation reports whether the text after a multi-character
// terminator run ("...", "! ?", ". . .") continues the current sentence. By
// default only whitespace then a lowercase letter or digit does.
func (l *language) isEllipsisContinuation(textAfterRun string) bool {
	if l.ellipsisContinuation != nil {
		return l.ellipsisContinuation(textAfterRun)
	}
	return ellipsisContinueRegex.MatchString(textAfterRun)
}

// isContinueInNextWord reports whether the text after a potential boundary
// continues the sentence: by default, when it starts with a lowercase letter
// or digit, or with a comma once a stray symmetric quote is peeled off.
func (l *language) isContinueInNextWord(textAfterBoundary string) bool {
	if l.continueInNextWord != nil {
		return l.continueInNextWord(textAfterBoundary)
	}
	if startsWithASCIILowercaseOrDigit(textAfterBoundary) {
		return true
	}
	return strings.HasPrefix(peelLeadingSymmetricQuote(textAfterBoundary), ",")
}

// getTrailingMarkers returns the trailing-marker table, empty by default.
func (l *language) getTrailingMarkers() *markerTable {
	if l.trailingMarkers != nil {
		return l.trailingMarkers
	}
	return emptyMarkerTable()
}

// getSkippableRanges fills out with the ranges of text whose terminators do
// not end a sentence: quoted text, parenthetical expressions and email
// addresses, sorted by start.
func (l *language) getSkippableRanges(text string, out []skippableRange) []skippableRange {
	out = collectQuoteRanges(text, out[:0])

	for _, m := range parensRegex.FindAllStringIndex(text, -1) {
		out = append(out, newSkippableRange(m[0], m[1], rangeParentheses))
	}

	for _, m := range emailRegex.FindAllStringIndex(text, -1) {
		out = append(out, newSkippableRange(m[0], m[1], rangeEmail))
	}

	// Sort ranges by start position for more efficient lookups.
	slices.SortStableFunc(out, func(a, b skippableRange) int { return cmp.Compare(a.start, b.start) })

	// Cache the mispairing of each quote range for reuse.
	tagQuoteMispairing(text, out)

	return out
}
