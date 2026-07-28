package context_test

import (
	"errors"
	"testing"

	uctx "github.com/gojargo/jargo/utils/context"
)

// TestCharAccumulatorAssemblesWords checks the common case: characters split on
// spaces, each word taking the offset of its first character.
func TestCharAccumulatorAssemblesWords(t *testing.T) {
	var acc uctx.CharAccumulator
	got, err := acc.Add(
		[]string{"H", "i", " ", "y", "o", "u", " "},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
	)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	want := []uctx.WordTiming{{Word: "Hi", Offset: 0.1}, {Word: "you", Offset: 0.4}}
	assertWords(t, got, want)

	// Both words were closed by a space, so nothing is left buffered.
	if _, ok := acc.Flush(); ok {
		t.Error("Flush() returned a word, want nothing left buffered")
	}
}

// TestCharAccumulatorCarriesWordAcrossBatches checks a word split across two
// batches keeps the offset of its first character, in the earlier batch.
func TestCharAccumulatorCarriesWordAcrossBatches(t *testing.T) {
	var acc uctx.CharAccumulator

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
	assertWords(t, got, []uctx.WordTiming{{Word: "hey", Offset: 0.5}})

	// The trailing word had no closing space, so only Flush releases it.
	wt, ok := acc.Flush()
	if !ok {
		t.Fatal("Flush() returned nothing, want the trailing word")
	}
	assertWords(t, []uctx.WordTiming{wt}, []uctx.WordTiming{{Word: "there", Offset: 0.9}})
}

// TestCharAccumulatorWordAtZero checks a word starting at offset zero is
// distinguished from no word at all.
func TestCharAccumulatorWordAtZero(t *testing.T) {
	var acc uctx.CharAccumulator
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

// TestCharAccumulatorResetDiscardsPartial checks Reset drops a partial word, so
// a reused accumulator does not leak the previous synthesis into the next.
func TestCharAccumulatorResetDiscardsPartial(t *testing.T) {
	var acc uctx.CharAccumulator
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
	var acc uctx.CharAccumulator
	got, err := acc.Add([]string{"a", "\n", "b", "\t"}, []float64{0.1, 0.2, 0.3, 0.4})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	assertWords(t, got, []uctx.WordTiming{{Word: "a", Offset: 0.1}, {Word: "b", Offset: 0.3}})
}

// TestCharAccumulatorLengthMismatch checks a malformed batch is rejected rather
// than misaligning every word after it.
func TestCharAccumulatorLengthMismatch(t *testing.T) {
	var acc uctx.CharAccumulator
	_, err := acc.Add([]string{"a", "b"}, []float64{0.1})
	if !errors.Is(err, uctx.ErrCharTimingLength) {
		t.Errorf("Add() error = %v, want ErrCharTimingLength", err)
	}
}

// TestCharsAsWords checks the spaceless-language path: one word per character,
// with punctuation dropped for want of an alphanumeric.
func TestCharsAsWords(t *testing.T) {
	got, err := uctx.CharsAsWords([]string{"你", "好", "。", "吗"}, []float64{0.0, 0.2, 0.4, 0.5})
	if err != nil {
		t.Fatalf("CharsAsWords: %v", err)
	}
	assertWords(t, got, []uctx.WordTiming{
		{Word: "你", Offset: 0.0},
		{Word: "好", Offset: 0.2},
		{Word: "吗", Offset: 0.5},
	})
}

// TestCharsAsWordsLengthMismatch checks the same guard applies here.
func TestCharsAsWordsLengthMismatch(t *testing.T) {
	_, err := uctx.CharsAsWords([]string{"a"}, nil)
	if !errors.Is(err, uctx.ErrCharTimingLength) {
		t.Errorf("CharsAsWords() error = %v, want ErrCharTimingLength", err)
	}
}

// assertWords compares two word-timing slices element by element.
func assertWords(t *testing.T, got, want []uctx.WordTiming) {
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
