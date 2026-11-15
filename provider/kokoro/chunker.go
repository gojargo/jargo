package kokoro

import (
	"strings"
)

// Splitting a phoneme string into batches the model can synthesize in one
// pass. The model context is fixed at maxPhonemeLength phonemes, so long
// inputs are cut at the least disruptive boundary available: sentence, clause,
// word, and only as a last resort mid-word. Every batch is non-empty and
// within the limit. Lengths count runes, one per phoneme.

// sentenceMarks end a sentence: .!? and the ellipsis.
const sentenceMarks = ".!?\u2026"

// clauseMarks end a clause.
const clauseMarks = ",;:"

// boundaryLevel is a place to cut, from least to most disruptive: whitespace
// after a sentence mark, whitespace after a clause mark, any whitespace.
// Punctuation stays with the text before it, which is how the model was
// trained.
type boundaryLevel int

const (
	boundarySentence boundaryLevel = iota
	boundaryClause
	boundaryWord
	boundaryCount
)

// splitAtBoundary splits s at each run of whitespace of the level: after one
// of the level's marks, or anywhere for a word boundary. It returns the pieces
// between the runs.
func splitAtBoundary(s string, level boundaryLevel) []string {
	runes := []rune(s)
	var pieces []string
	start := 0
	for i := 0; i < len(runes); {
		if !isSpace(runes[i]) {
			i++
			continue
		}
		cut := true
		switch level {
		case boundarySentence:
			cut = i > 0 && strings.ContainsRune(sentenceMarks, runes[i-1])
		case boundaryClause:
			cut = i > 0 && strings.ContainsRune(clauseMarks, runes[i-1])
		case boundaryWord, boundaryCount:
		}
		j := i
		for j < len(runes) && isSpace(runes[j]) {
			j++
		}
		if cut {
			pieces = append(pieces, string(runes[start:i]))
			start = j
		}
		i = j
	}
	return append(pieces, string(runes[start:]))
}

// atoms cuts phonemes into pieces no longer than maxLength, at the least
// disruptive boundary from level on.
func atoms(phonemes string, maxLength int, level boundaryLevel) []string {
	runes := []rune(phonemes)
	if len(runes) <= maxLength {
		if len(runes) > 0 {
			return []string{phonemes}
		}
		return nil
	}
	for index := level; index < boundaryCount; index++ {
		pieces := splitAtBoundary(phonemes, index)
		if len(pieces) > 1 {
			var out []string
			for _, piece := range pieces {
				out = append(out, atoms(trimSpace(piece), maxLength, index+1)...)
			}
			return out
		}
	}
	// A single unbroken run longer than the context: slice it rather than
	// dropping the tail.
	var out []string
	for start := 0; start < len(runes); start += maxLength {
		out = append(out, string(runes[start:min(start+maxLength, len(runes))]))
	}
	return out
}

// pack groups consecutive atoms into batches within limit, as index ranges.
func pack(lengths []int, limit int) [][2]int {
	var batches [][2]int
	start, size := 0, 0
	for index, length := range lengths {
		candidate := size + 1 + length
		if index == start {
			candidate = length
		}
		if candidate > limit && index > start {
			batches = append(batches, [2]int{start, index})
			start, size = index, length
		} else {
			size = candidate
		}
	}
	if len(lengths) > 0 {
		batches = append(batches, [2]int{start, len(lengths)})
	}
	return batches
}

// splitPhonemes splits phonemes into evenly sized batches of at most
// maxLength. Filling each batch to the limit would leave a short remainder,
// and a short batch is spoken at a different rate and loudness than its
// neighbors, so the batches are balanced: the smallest limit that still
// needs no extra batch.
func splitPhonemes(phonemes string, maxLength int) []string {
	parts := atoms(trimSpace(phonemes), maxLength, boundarySentence)
	if len(parts) == 0 {
		return nil
	}
	lengths := make([]int, len(parts))
	low := 0
	for i, a := range parts {
		lengths[i] = len([]rune(a))
		low = max(low, lengths[i])
	}
	fewest := len(pack(lengths, maxLength))

	high := maxLength
	for low < high {
		middle := (low + high) / 2
		if len(pack(lengths, middle)) <= fewest {
			high = middle
		} else {
			low = middle + 1
		}
	}

	ranges := pack(lengths, low)
	batches := make([]string, len(ranges))
	for i, r := range ranges {
		batches[i] = strings.Join(parts[r[0]:r[1]], " ")
	}
	return batches
}

// pauseAfter is the seconds of silence a batch ending with this text is
// followed by: sentence after a sentence mark, clause after a clause mark,
// none otherwise.
func pauseAfter(phonemes string, sentence, clause float64) float64 {
	runes := []rune(strings.TrimRightFunc(phonemes, isSpace))
	if len(runes) == 0 {
		// An empty mark is found in any set of marks, the sentence's first.
		return sentence
	}
	mark := runes[len(runes)-1]
	switch {
	case strings.ContainsRune(sentenceMarks, mark):
		return sentence
	case strings.ContainsRune(clauseMarks, mark):
		return clause
	}
	return 0
}
