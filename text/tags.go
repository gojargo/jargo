package text

import "strings"

// StartEndTags is a pair of delimiters marking a run of text: an opening tag and
// the closing tag that ends it.
type StartEndTags struct {
	// Start opens the run.
	Start string
	// End closes it.
	End string
}

// ParseStartEndTags reports which tag pair the text ends inside, and the offset
// a later call should resume scanning from.
//
// Text arrives a piece at a time, so a tag can be split across pieces and the
// buffer has to be rescanned as it grows. Pass the returned offset back with the
// next call, along with the tag returned, and only what is new gets scanned
// again.
//
// The scan starts at index. Inside a tag, it looks for that tag's end; outside
// one, for the earliest start of any pair, and then for that pair's end after
// it. A pair whose start and end are the same, such as {"**", "**"}, is found
// the same way. The offset returned is just past the start of the tag left
// open, or past the last tag that closed.
func ParseStartEndTags(
	text string, tags []StartEndTags, current *StartEndTags, index int,
) (*StartEndTags, int) {
	index = max(index, 0)
	for {
		if current != nil {
			end := find(text, current.End, index)
			if end < 0 {
				return current, index
			}
			index = end + len(current.End)
			current = nil
		}
		position := -1
		for _, tag := range tags {
			if at := find(text, tag.Start, index); at >= 0 && (position < 0 || at < position) {
				position = at
				t := tag
				current = &t
			}
		}
		if current == nil {
			return nil, index
		}
		index = position + len(current.Start)
	}
}

// find is the offset of the first sub in text at or after index, or -1.
func find(text, sub string, index int) int {
	at := strings.Index(from(text, index), sub)
	if at < 0 {
		return -1
	}
	return index + at
}

// from is text from index on, and nothing when the offset is past its end.
//
// The buffer being scanned grows a character at a time, but it also shrinks:
// the aggregator takes a finished sentence off the front of it and leaves the
// offset settled so far where it was. An offset beyond what is left simply has
// nothing to scan, and the buffer regrowing past it puts the two back in step.
func from(text string, index int) string {
	if index >= len(text) {
		return ""
	}
	if index < 0 {
		return text
	}
	return text[index:]
}

// LongestTrailingPartialMatch returns the length of the longest suffix of text
// that is a proper prefix of one of the candidates.
//
// It is what lets a delimiter split across two pieces of text still be
// recognized. Text ending in "<spe" of "<spell>" is the start of a tag that has
// not arrived in full, so holding those characters back until the next piece
// says whether they open a tag keeps the delimiter from being emitted as if it
// were ordinary text.
func LongestTrailingPartialMatch(text string, candidates []string) int {
	longest := 0
	for _, candidate := range candidates {
		maxLen := min(len(candidate)-1, len(text))
		for length := maxLen; length > longest; length-- {
			if text[len(text)-length:] == candidate[:length] {
				longest = length
				break
			}
		}
	}
	return longest
}
