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

// ParseStartEndTags reports which tag pair the text is currently inside, and the
// offset a later call should resume scanning from.
//
// Text arrives a piece at a time, so a tag can be split across pieces and the
// buffer has to be rescanned as it grows. The returned offset is how much of the
// buffer this call settled: pass it back with the next call, along with the tag
// returned, and only what is new gets scanned again.
//
// A nil current means no tag is open, and the scan looks for one to open. A
// non-nil current means one is, and the scan looks only for its closing tag.
func ParseStartEndTags(
	text string, tags []StartEndTags, current *StartEndTags, index int,
) (*StartEndTags, int) {
	// Already inside a tag, so the only thing worth looking for is the end of it.
	if current != nil {
		if strings.Contains(text[index:], current.End) {
			return nil, len(text)
		}
		return current, index
	}

	for _, tag := range tags {
		rest := text[index:]
		starts := strings.Count(rest, tag.Start)
		ends := strings.Count(rest, tag.End)
		switch {
		case starts == 0 && ends == 0:
			// Neither delimiter is here, so nothing has been settled and the
			// same text is scanned again once it has grown.
			return nil, index
		case starts > ends:
			// A run was opened and not closed, so the text from here on belongs
			// to it until the closing tag arrives.
			t := tag
			return &t, len(text)
		case starts == ends:
			// Every run opened here was also closed, so none is left open.
			return nil, len(text)
		}
		// More ends than starts, which says nothing about this pair: the closing
		// tag belongs to a run opened before the text under scan. Try the next.
	}

	return nil, index
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
