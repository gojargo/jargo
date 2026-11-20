package context

import (
	"errors"
	"fmt"
	"strings"
)

// ErrCharTimingLength is returned when a batch of character timings has a
// different number of characters and start offsets, which would misalign every
// word assembled after it.
//
//nolint:gochecknoglobals // sentinel error
var ErrCharTimingLength = errors.New("context: character timing length mismatch")

// CharAccumulator assembles per-character timings into whole words. Some
// synthesizers report a start offset for every character (spaces and
// punctuation included) rather than per word, in batches that need not align
// with word boundaries. It splits on spaces, gives each word the offset of its
// first character, and carries a word split across two batches into the next
// one.
//
// The zero value is ready to use. It is not safe for concurrent use; a
// synthesizer drives one per synthesis.
type CharAccumulator struct {
	// AudioTags makes a bracketed audio tag, such as "[short pause]", one word,
	// spaces and all. A synthesizer returning the tags it performs in its
	// character timings sets it, so a tag reported whole matches the text it
	// was sent in.
	AudioTags bool
	// WordEnds makes each word a space completes report where it ends, as its
	// End. A word ends where the space after it ends: that space holds the pause
	// before the next word, or the silence after the last one. The space's
	// length is taken from the durations AddWithDurations is given; without
	// one, the word ends where the space starts.
	WordEnds bool

	word string
	// start is the offset of the carried word's first character, in seconds from
	// the beginning of the synthesis.
	start float64
	// started distinguishes a word beginning at offset zero from no word at all.
	started bool
}

// Add folds one batch in and returns every word it completes. chars and starts
// are parallel: starts[i] is when chars[i] begins, in seconds from the start of
// the synthesis. A word is completed by a space, so the last word of an
// utterance stays buffered until Flush.
func (a *CharAccumulator) Add(chars []string, starts []float64) ([]WordTiming, error) {
	return a.AddWithDurations(chars, starts, nil)
}

// AddWithDurations is Add for a batch that also says how long each character
// lasts: durations[i] is the length of chars[i], in seconds. It may be shorter
// than chars, or nil, and a character without one lasts no time. The durations
// time where each word ends, which is reported when WordEnds is set.
func (a *CharAccumulator) AddWithDurations(chars []string, starts, durations []float64) ([]WordTiming, error) {
	if len(chars) != len(starts) {
		return nil, fmt.Errorf("%w: %d characters, %d offsets", ErrCharTimingLength, len(chars), len(starts))
	}
	var words []WordTiming
	// A word carried over can end inside a tag cut off between batches.
	inTag := a.AudioTags && strings.LastIndex(a.word, "[") > strings.LastIndex(a.word, "]")
	for i, ch := range chars {
		if a.AudioTags && (ch == "[" || ch == "]") {
			inTag = ch == "["
		}
		if strings.TrimSpace(ch) == "" && !inTag {
			if wt, ok := a.Flush(); ok {
				if a.WordEnds {
					wt.End = spaceEnd(starts, durations, i)
				}
				words = append(words, wt)
			}
			continue
		}
		if !a.started {
			a.start = starts[i]
			a.started = true
		}
		a.word += ch
	}
	return words, nil
}

// spaceEnd is where the space at i ends, which is where the word it closes
// ends. A space with no duration ends where it starts.
func spaceEnd(starts, durations []float64, i int) float64 {
	if i < len(durations) {
		return starts[i] + durations[i]
	}
	return starts[i]
}

// Flush returns the word still being assembled, if any, and clears it. Call it
// when the synthesizer reports the utterance is complete: a final word has no
// terminating space to close it.
func (a *CharAccumulator) Flush() (WordTiming, bool) {
	if a.word == "" {
		a.Reset()
		return WordTiming{}, false
	}
	wt := WordTiming{Word: a.word, Offset: a.start}
	a.Reset()
	return wt, true
}

// Reset discards any partially assembled word.
func (a *CharAccumulator) Reset() {
	a.word = ""
	a.start = 0
	a.started = false
}

// CharsAsWords treats each character as its own word, keeping only the ones
// carrying an alphanumeric. Languages written without spaces between words
// (Chinese and Japanese) get per-character timings that cannot be assembled into
// words by splitting on spaces, so they are reported as they arrive.
func CharsAsWords(chars []string, starts []float64) ([]WordTiming, error) {
	if len(chars) != len(starts) {
		return nil, fmt.Errorf("%w: %d characters, %d offsets", ErrCharTimingLength, len(chars), len(starts))
	}
	words := make([]WordTiming, 0, len(chars))
	for i, ch := range chars {
		if containsAlnum(ch) {
			words = append(words, WordTiming{Word: ch, Offset: starts[i]})
		}
	}
	return words, nil
}
