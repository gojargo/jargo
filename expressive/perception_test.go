package expressive_test

import (
	"reflect"
	"slices"
	"testing"

	"github.com/gojargo/jargo/expressive"
)

func TestAReadingFormatsLabelsWithConfidencesAndValues(t *testing.T) {
	rate := 4.1
	signals := []expressive.PerceptionSignal{
		{Name: "tone", Labels: []expressive.PerceptionLabel{{"frustrated", 0.86}, {"neutral", 0.10}}},
		{Name: "speech_rate", Value: &rate},
	}

	want := "([tone frustrated=0.86, neutral=0.10],[speech_rate value=4.1])"
	if got := expressive.FormatPerception(signals); got != want {
		t.Errorf("FormatPerception = %q, want %q", got, want)
	}
}

func TestAMeasureIsWrittenInItsShortestForm(t *testing.T) {
	for value, want := range map[float64]string{
		4.0: "4", 0.123456789: "0.123457", 1234567: "1.23457e+06", 100000: "100000", 0.00001: "1e-05",
	} {
		got := expressive.FormatPerception([]expressive.PerceptionSignal{{Name: "m", Value: &value}})
		if got != "([m value="+want+"])" {
			t.Errorf("value %v is written %q, want %q", value, got, want)
		}
	}
}

func tone(labels ...expressive.PerceptionLabel) expressive.PerceptionSignal {
	return expressive.PerceptionSignal{Name: "tone", Labels: labels}
}

func valued(name string, v float64) expressive.PerceptionSignal {
	return expressive.PerceptionSignal{Name: name, Value: &v}
}

// Ported from upstream
// (test_merging_keeps_every_label_with_its_highest_confidence_likeliest_first).
func TestMergingKeepsEveryLabelWithItsHighestConfidenceLikeliestFirst(t *testing.T) {
	earlier := []expressive.PerceptionSignal{tone(
		expressive.PerceptionLabel{Label: "neutral", Confidence: 0.80},
		expressive.PerceptionLabel{Label: "frustrated", Confidence: 0.11})}
	later := []expressive.PerceptionSignal{tone(
		expressive.PerceptionLabel{Label: "frustrated", Confidence: 0.36},
		expressive.PerceptionLabel{Label: "happy", Confidence: 0.30})}
	want := []expressive.PerceptionSignal{tone(
		expressive.PerceptionLabel{Label: "neutral", Confidence: 0.80},
		expressive.PerceptionLabel{Label: "frustrated", Confidence: 0.36},
		expressive.PerceptionLabel{Label: "happy", Confidence: 0.30})}
	if got := expressive.MergePerception(earlier, later); !reflect.DeepEqual(got, want) {
		t.Errorf("merged %+v, want %+v", got, want)
	}
}

// Ported from upstream (test_merging_keeps_the_higher_value_whichever_comes_first).
func TestMergingKeepsTheHigherValueWhicheverComesFirst(t *testing.T) {
	unsure := []expressive.PerceptionSignal{valued("hesitation", 0.6)}
	sure := []expressive.PerceptionSignal{valued("hesitation", 0.2)}
	if got := expressive.MergePerception(sure, unsure); !reflect.DeepEqual(got, unsure) {
		t.Errorf("merged %+v, want %+v", got, unsure)
	}
	if got := expressive.MergePerception(unsure, sure); !reflect.DeepEqual(got, unsure) {
		t.Errorf("merged %+v, want %+v", got, unsure)
	}
}

// Ported from upstream
// (test_merging_keeps_the_signals_only_one_reading_has_in_the_order_first_read).
func TestMergingKeepsTheSignalsOnlyOneReadingHasInTheOrderFirstRead(t *testing.T) {
	earlier := []expressive.PerceptionSignal{
		tone(expressive.PerceptionLabel{Label: "frustrated", Confidence: 0.86}), valued("hesitation", 0.3),
	}
	later := []expressive.PerceptionSignal{
		valued("sarcasm", 0.9), tone(expressive.PerceptionLabel{Label: "neutral", Confidence: 0.81}),
	}
	var names []string
	for _, s := range expressive.MergePerception(earlier, later) {
		names = append(names, s.Name)
	}
	if want := []string{"tone", "hesitation", "sarcasm"}; !slices.Equal(names, want) {
		t.Errorf("signals %v, want %v", names, want)
	}
	if got := expressive.MergePerception(nil, later); !reflect.DeepEqual(got, later) {
		t.Errorf("merged into nothing %+v, want %+v", got, later)
	}
}
