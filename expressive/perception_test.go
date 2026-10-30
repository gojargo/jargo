package expressive_test

import (
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
