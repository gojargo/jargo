package align_test

import (
	"errors"
	"math"
	"testing"

	"github.com/gojargo/jargo/text/align"
)

// TestCharAccumulatorAssemblesWords checks the common case: characters split on
// spaces, each word taking the offset of its first character.
func TestCharAccumulatorAssemblesWords(t *testing.T) {
	var acc align.CharAccumulator
	got, err := acc.Add(
		[]string{"H", "i", " ", "y", "o", "u", " "},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
	)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	want := []align.WordTiming{{Word: "Hi", Offset: 0.1}, {Word: "you", Offset: 0.4}}
	assertWords(t, got, want)

	// Both words were closed by a space, so nothing is left buffered.
	if _, ok := acc.Flush(); ok {
		t.Error("Flush() returned a word, want nothing left buffered")
	}
}

// TestCharAccumulatorCarriesWordAcrossBatches checks a word split across two
// batches keeps the offset of its first character, in the earlier batch.
func TestCharAccumulatorCarriesWordAcrossBatches(t *testing.T) {
	var acc align.CharAccumulator

	got, err := acc.Add([]string{"h", "e"}, []float64{0.5, 0.6})
	if err != nil {
		t.Fatalf("first Add: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("first Add returned %+v, want nothing until the word is closed", got)
	}

	got, err = acc.Add([]string{"y", " ", "t", "h", "e", "r", "e"}, []float64{0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3})
	if err != nil {
		t.Fatalf("second Add: %v", err)
	}
	assertWords(t, got, []align.WordTiming{{Word: "hey", Offset: 0.5}})

	// The trailing word had no closing space, so only Flush releases it.
	wt, ok := acc.Flush()
	if !ok {
		t.Fatal("Flush() returned nothing, want the trailing word")
	}
	assertWords(t, []align.WordTiming{wt}, []align.WordTiming{{Word: "there", Offset: 0.9}})
}

// TestCharAccumulatorWordAtZero checks a word starting at offset zero is
// distinguished from no word at all.
func TestCharAccumulatorWordAtZero(t *testing.T) {
	var acc align.CharAccumulator
	if _, err := acc.Add([]string{"h", "i"}, []float64{0.0, 0.1}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	wt, ok := acc.Flush()
	if !ok {
		t.Fatal("Flush() returned nothing, want the buffered word")
	}
	if wt.Word != "hi" || wt.Offset != 0.0 {
		t.Errorf("Flush() = %+v, want hi at offset 0", wt)
	}
}

// TestCharAccumulatorWordEndsRunToTheEndOfTheFollowingSpace checks a word ends
// where the space after it ends, even when it spans two batches, and that no
// end is reported unless asked for.
func TestCharAccumulatorWordEndsRunToTheEndOfTheFollowingSpace(t *testing.T) {
	// batch times each character of text 0.1s after the last from offset, each
	// lasting 0.1s.
	batch := func(text string, offset float64) (chars []string, starts, durations []float64) {
		for i, r := range []rune(text) {
			chars = append(chars, string(r))
			starts = append(starts, offset+float64(i)*0.1)
			durations = append(durations, 0.1)
		}
		return chars, starts, durations
	}
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

	acc := align.CharAccumulator{WordEnds: true}
	first, err := acc.AddWithDurations(batch("Hi the", 0))
	if err != nil {
		t.Fatalf("first AddWithDurations: %v", err)
	}
	if len(first) != 1 || first[0].Word != "Hi" || first[0].Offset != 0 || !near(first[0].End, 0.3) {
		t.Fatalf("first batch = %+v, want Hi from 0 to 0.3", first)
	}

	// "re " starts 0.6s in; its space starts 0.2s later and lasts 0.1s.
	second, err := acc.AddWithDurations(batch("re ", 0.6))
	if err != nil {
		t.Fatalf("second AddWithDurations: %v", err)
	}
	if len(second) != 1 || second[0].Word != "there" || !near(second[0].Offset, 0.3) || !near(second[0].End, 0.9) {
		t.Fatalf("second batch = %+v, want there from 0.3 to 0.9", second)
	}

	var plain align.CharAccumulator
	got, err := plain.AddWithDurations(batch("Hi ", 0))
	if err != nil {
		t.Fatalf("AddWithDurations: %v", err)
	}
	assertWords(t, got, []align.WordTiming{{Word: "Hi", Offset: 0}})
}

// TestCharAccumulatorResetDiscardsPartial checks Reset drops a partial word, so
// a reused accumulator does not leak the previous synthesis into the next.
func TestCharAccumulatorResetDiscardsPartial(t *testing.T) {
	var acc align.CharAccumulator
	if _, err := acc.Add([]string{"h", "i"}, []float64{0.1, 0.2}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	acc.Reset()
	if wt, ok := acc.Flush(); ok {
		t.Errorf("Flush() = %+v, want nothing after Reset", wt)
	}
}

// TestCharAccumulatorTreatsAnyWhitespaceAsABoundary checks tabs and newlines
// close a word too, not just the space character.
func TestCharAccumulatorTreatsAnyWhitespaceAsABoundary(t *testing.T) {
	var acc align.CharAccumulator
	got, err := acc.Add([]string{"a", "\n", "b", "\t"}, []float64{0.1, 0.2, 0.3, 0.4})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	assertWords(t, got, []align.WordTiming{{Word: "a", Offset: 0.1}, {Word: "b", Offset: 0.3}})
}

// TestCharAccumulatorLengthMismatch checks a malformed batch is rejected rather
// than misaligning every word after it.
func TestCharAccumulatorLengthMismatch(t *testing.T) {
	var acc align.CharAccumulator
	_, err := acc.Add([]string{"a", "b"}, []float64{0.1})
	if !errors.Is(err, align.ErrCharTimingLength) {
		t.Errorf("Add() error = %v, want ErrCharTimingLength", err)
	}
}

// TestCharsAsWords checks the spaceless-language path: one word per character,
// with punctuation dropped for want of an alphanumeric.
func TestCharsAsWords(t *testing.T) {
	got, err := align.CharsAsWords([]string{"你", "好", "。", "吗"}, []float64{0.0, 0.2, 0.4, 0.5})
	if err != nil {
		t.Fatalf("CharsAsWords: %v", err)
	}
	assertWords(t, got, []align.WordTiming{
		{Word: "你", Offset: 0.0},
		{Word: "好", Offset: 0.2},
		{Word: "吗", Offset: 0.5},
	})
}

// TestCharsAsWordsLengthMismatch checks the same guard applies here.
func TestCharsAsWordsLengthMismatch(t *testing.T) {
	_, err := align.CharsAsWords([]string{"a"}, nil)
	if !errors.Is(err, align.ErrCharTimingLength) {
		t.Errorf("CharsAsWords() error = %v, want ErrCharTimingLength", err)
	}
}

// assertWords compares two word-timing slices element by element.
func assertWords(t *testing.T, got, want []align.WordTiming) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("word %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}
