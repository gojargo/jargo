package sentencex

import "strings"

// Fronting-phrase detection: does the text before a marker look like an
// adverbial lead?

// wordBeforeMarkerIsCapitalised reports whether the trailing word of prefix,
// once trailing digits, colons and whitespace are stripped, starts with an
// uppercase ASCII letter.
func wordBeforeMarkerIsCapitalised(prefix string) bool {
	trimmed := strings.TrimRightFunc(prefix, func(c rune) bool {
		return isASCIIDigit(c) || c == ':' || isWhitespace(c)
	})
	first, ok := firstRune(lastPieceAfter(trimmed, isWhitespace))
	return ok && isASCIIUpper(first)
}

// hasQuoteOpenerByte reports whether s holds a byte that some quote opener
// begins with. prefixIsPurelyFronting uses it as a fast check.
func hasQuoteOpenerByte(s string) bool {
	firstBytes := quoteOpenerFirstBytes()
	for i := range len(s) {
		if firstBytes[s[i]] {
			return true
		}
	}
	return false
}

// prefixIsPurelyFronting reports whether prefix is purely a fronting
// (adverbial) lead-in, so the marker sits mid-phrase rather than ending a
// sentence, as in "On Jan. 5, at 6 a.m.". Splitting on whitespace, commas,
// semicolons and dashes, every token must be a fronting word, an
// abbreviation, a number or time (digits and :), or capitalized. Spans inside
// paired quotes are skipped. It is false for a language with no fronting
// words.
func prefixIsPurelyFronting(prefix string, l *language) bool {
	fronting := l.frontingWords
	if len(fronting) == 0 {
		return false
	}

	isFronting := func(seg string) bool {
		return segmentIsFronting(seg, fronting, l.abbreviations)
	}

	// Skip the quote scan when no quote opener is present.
	if !hasQuoteOpenerByte(prefix) {
		return isFronting(prefix)
	}

	cursor := 0
	for _, m := range quotesFindAll(prefix) {
		if !isFronting(prefix[cursor:m[0]]) {
			return false
		}
		cursor = m[1]
	}

	return isFronting(prefix[cursor:])
}

// segmentIsFronting reports whether every token of seg, split on whitespace,
// commas, semicolons and dashes, is a fronting word, an abbreviation, a
// number or time (digits and :), or capitalized.
func segmentIsFronting(seg string, fronting, abbrevs map[string]struct{}) bool {
	const (
		emDash = '\u2014'
		enDash = '\u2013'
	)

	tokens := strings.FieldsFunc(seg, func(c rune) bool {
		return isWhitespace(c) || c == ',' || c == ';' || c == emDash || c == enDash
	})
	for _, tok := range tokens {
		bare := strings.TrimRight(tok, ".")
		first, _ := firstRune(bare)
		ok := bare == "" ||
			strings.IndexFunc(bare, func(c rune) bool { return !isASCIIDigit(c) && c != ':' }) < 0 ||
			isUppercase(first) ||
			abbreviationSetContains(fronting, bare) ||
			abbreviationSetContains(abbrevs, bare)
		if !ok {
			return false
		}
	}
	return true
}
