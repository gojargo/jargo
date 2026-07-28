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
	if len(chars) != len(starts) {
		return nil, fmt.Errorf("%w: %d characters, %d offsets", ErrCharTimingLength, len(chars), len(starts))
	}
	var words []WordTiming
	for i, ch := range chars {
		if strings.TrimSpace(ch) == "" {
			if wt, ok := a.Flush(); ok {
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
		if hasAlnum(ch) {
			words = append(words, WordTiming{Word: ch, Offset: starts[i]})
		}
	}
	return words, nil
}
