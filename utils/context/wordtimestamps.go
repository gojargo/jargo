package context

import "strings"

// WordTiming is a single word-timestamp event from a synthesizer: the spoken
// token and its start offset in seconds from the beginning of the synthesis.
type WordTiming struct {
	Word   string
	Offset float64
}

// MergePunctTokens merges punctuation- and space-only tokens into the preceding
// word. Some synthesizers emit spaces and punctuation as separate word-timestamp
// tokens rather than attaching them to the adjacent word; this collapses those
// so downstream consumers always receive words with trailing punctuation already
// attached.
//
// A token is punct/space-only when it has no alphanumeric character after markup
// is stripped. Such a token is appended to the preceding word's text and its
// offset discarded (the preceding word's offset is kept). Leading punct/space
// tokens with no preceding word are dropped. Every returned token is trimmed of
// leading and trailing whitespace.
func MergePunctTokens(words []WordTiming) []WordTiming {
	merged := make([]WordTiming, 0, len(words))
	for _, wt := range words {
		if hasAlnum(stripMarkup(wt.Word)) {
			merged = append(merged, wt)
			continue
		}
		if n := len(merged); n > 0 {
			merged[n-1].Word += wt.Word
		}
		// A leading punct/space token with no preceding word is dropped.
	}
	for i := range merged {
		merged[i].Word = strings.TrimSpace(merged[i].Word)
	}
	return merged
}

// hasAlnum reports whether text contains at least one alphanumeric character.
func hasAlnum(text string) bool {
	for _, r := range text {
		if isAlnum(r) {
			return true
		}
	}
	return false
}
