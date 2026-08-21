package context

import (
	"regexp"
	"unicode"
)

// Two notions of "what is markup" live in this file, and they differ on a single
// point: a '<' with no later '>'.
//
//   - stripMarkup treats it as an open tag and swallows the rest of the string.
//     That is correct for a fragment that may have been cut mid-tag, such as a
//     single token from a word-timestamp stream.
//   - stripCompleteMarkup treats it as content. That is correct for a complete
//     text, where a lone '<' is real ("5 < 10", "<3").
//
// Every markup decision in the word-timestamp path routes through one of the
// two, so the callers cannot disagree about which characters are a tag.

// markupRE matches a well-formed markup tag. It is the definition of markup for
// a static text, shared by stripCompleteMarkup and splitMarkupRuns so the two
// cannot disagree about which characters a segment split may treat as a tag.
//
//nolint:gochecknoglobals // compiled once, read-only
var markupRE = regexp.MustCompile(`<[^>]+>`)

// stripCompleteMarkup removes well-formed <...> tags from a complete text. A
// lone '<' with no later '>' is left in place as real content.
func stripCompleteMarkup(text string) string {
	return markupRE.ReplaceAllString(text, "")
}

// iterCleanChars returns the rune indices of runes that lie outside markup.
// Markup is anything between '<' and '>'; an unclosed '<' swallows the rest of
// the string, matching how a streamed word-timestamp token can arrive mid-tag.
func iterCleanChars(runes []rune) []int {
	out := make([]int, 0, len(runes))
	inTag := false
	for i, r := range runes {
		switch {
		case inTag:
			if r == '>' {
				inTag = false
			}
		case r == '<':
			inTag = true
		default:
			out = append(out, i)
		}
	}
	return out
}

// stripMarkup removes markup from a possibly-truncated word-timestamp fragment.
// An unclosed '<' swallows the rest of text.
func stripMarkup(text string) string {
	runes := []rune(text)
	idx := iterCleanChars(runes)
	out := make([]rune, len(idx))
	for i, j := range idx {
		out[i] = runes[j]
	}
	return string(out)
}

// rawLenForCleanChars returns the raw rune offset into text just past its n-th
// markup-stripped rune. It is the inverse of stripMarkup for a prefix: where
// stripMarkup collects every non-markup rune, this finds the raw index one past
// the n-th of them, converting a match measured in markup-stripped space back to
// a raw offset. It returns the rune length of text when text has fewer than n
// non-markup runes.
func rawLenForCleanChars(text string, n int) int {
	if n <= 0 {
		return 0
	}
	runes := []rune(text)
	seen := 0
	for _, i := range iterCleanChars(runes) {
		seen++
		if seen == n {
			return i + 1
		}
	}
	return len(runes)
}

// splitMarkupRuns splits text into alternating runs of tagged and untagged
// words. A word counts as tagged when it overlaps a complete <...> pair; a lone
// '<' is content, not the start of a tag. Consecutive words with the same
// classification form one run, so whitespace inside a tag such as
// <phoneme alphabet="ipa"> never splits the words it spans across runs.
//
//	splitMarkupRuns("I love to count <spell>1234</spell>.")
//	// → ["I love to count ", "<spell>1234</spell>."]
//
// Text with no markup yields a single run, unchanged. It is what lets a tag be
// given a segment of its own rather than turning the sentence around it into one
// all-or-nothing span.
func splitMarkupRuns(text string) []string {
	tagSpans := markupRuneSpans(text)
	if len(tagSpans) == 0 {
		if text == "" {
			return nil
		}
		return []string{text}
	}

	var runs []string
	// runIsTagged is the classification of the run being built; -1 before the
	// first token, so the first token always opens a run.
	runIsTagged := -1
	pos := 0
	for _, token := range tokenizeWS(text) {
		if token == "" {
			continue
		}
		start := pos
		end := pos + len([]rune(token))
		pos = end

		tagged := 0
		for _, sp := range tagSpans {
			if sp[0] < end && start < sp[1] {
				tagged = 1
				break
			}
		}
		if tagged == runIsTagged {
			runs[len(runs)-1] += token
		} else {
			runs = append(runs, token)
			runIsTagged = tagged
		}
	}
	return runs
}

// markupRuneSpans returns the [start, end) rune spans of every well-formed tag
// in text. It reads them off markupRE, the same expression stripCompleteMarkup
// uses, so a segment split can never treat as a tag something the strip would
// leave in place.
func markupRuneSpans(text string) [][2]int {
	byteSpans := markupRE.FindAllStringIndex(text, -1)
	if byteSpans == nil {
		return nil
	}
	// Walk the string once, recording the rune index each byte offset falls on.
	runeAt := make(map[int]int, len(byteSpans)*2)
	n := 0
	for b := range text {
		runeAt[b] = n
		n++
	}
	runeAt[len(text)] = n

	spans := make([][2]int, 0, len(byteSpans))
	for _, bs := range byteSpans {
		spans = append(spans, [2]int{runeAt[bs[0]], runeAt[bs[1]]})
	}
	return spans
}

// tokenizeWS splits text on runs of whitespace, keeping the separators, so
// joining the tokens reconstructs text exactly.
func tokenizeWS(text string) []string {
	runes := []rune(text)
	var tokens []string
	start := 0
	i := 0
	for i < len(runes) {
		if unicode.IsSpace(runes[i]) {
			tokens = append(tokens, string(runes[start:i]))
			j := i
			for j < len(runes) && unicode.IsSpace(runes[j]) {
				j++
			}
			tokens = append(tokens, string(runes[i:j]))
			i = j
			start = j
		} else {
			i++
		}
	}
	tokens = append(tokens, string(runes[start:]))
	return tokens
}
