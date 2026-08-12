package turn

import (
	"math"
	"testing"
)

// sine builds secs seconds of a tone at freq Hz, sampled at rate.
func sine(freq float64, rate int, secs float64) []float32 {
	n := int(float64(rate) * secs)
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
	}
	return out
}

// rms is the root mean square of a signal, its loudness in one number.
func rms(x []float32) float64 {
	if len(x) == 0 {
		return 0
	}
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(x)))
}

// TestResampleToModelRateFiltersAboveNyquist is the test that says the
// conversion is filtered rather than merely interpolated.
//
// A 12 kHz tone sampled at 48 kHz cannot be represented at 16 kHz: it sits above
// the 8 kHz Nyquist. A filtered conversion discards it and leaves near silence.
// Interpolating without a filter instead folds it down to |12000-16000| = 4 kHz,
// where it becomes a loud tone sitting squarely in the speech band the model
// reads its features from.
func TestResampleToModelRateFiltersAboveNyquist(t *testing.T) {
	const inRate = 48000

	above := sine(12000, inRate, 0.5)
	got, err := resampleToModelRate(above, inRate)
	if err != nil {
		t.Fatalf("resampleToModelRate: %v", err)
	}

	// The tone is gone rather than folded down into the speech band.
	if loud := rms(got); loud > 0.05 {
		t.Errorf("a 12 kHz tone resampled to 16 kHz has RMS %v, want it filtered out (below 0.05)", loud)
	}
}

// TestResampleToModelRateKeepsSpeechBandContent checks the filter takes only
// what it should: a tone well inside the new Nyquist survives the conversion.
func TestResampleToModelRateKeepsSpeechBandContent(t *testing.T) {
	const inRate = 48000

	within := sine(1000, inRate, 0.5)
	got, err := resampleToModelRate(within, inRate)
	if err != nil {
		t.Fatalf("resampleToModelRate: %v", err)
	}

	// A 0.5 amplitude sine has an RMS of about 0.354.
	if loud := rms(got); loud < 0.3 {
		t.Errorf("a 1 kHz tone resampled to 16 kHz has RMS %v, want it kept (about 0.354)", loud)
	}

	// The output runs at a third of the input rate, short by the converter's
	// filter delay. How long that delay is depends on the converter, so the
	// tolerance is generous: what matters is the rate, not the last few samples.
	want := len(within) * melSR / inRate
	if diff := len(got) - want; diff > want/20 || diff < -want/20 {
		t.Errorf("resampled to %d samples, want about %d", len(got), want)
	}
}

// TestResampleToModelRatePassesThroughModelRate checks audio already at the
// model's rate is handed on untouched, so the common path costs nothing.
func TestResampleToModelRatePassesThroughModelRate(t *testing.T) {
	in := sine(1000, melSR, 0.1)

	for _, rate := range []int{melSR, 0} {
		got, err := resampleToModelRate(in, rate)
		if err != nil {
			t.Fatalf("resampleToModelRate at %d: %v", rate, err)
		}
		if len(got) != len(in) {
			t.Errorf("rate %d: got %d samples, want the input's %d", rate, len(got), len(in))
		}
		if len(got) > 0 && got[1] != in[1] {
			t.Errorf("rate %d: the audio was altered", rate)
		}
	}

	// Empty audio is handed back rather than run through a converter.
	if got, err := resampleToModelRate(nil, 48000); err != nil || got != nil {
		t.Errorf("resampleToModelRate(nil) = (%v, %v), want (nil, nil)", got, err)
	}
}

// TestF32ToS16 checks the clamp on the way back to PCM. Sinc interpolation
// overshoots past the range a sample can hold, so the clamp is load-bearing:
// without it an overshoot wraps to the opposite extreme and becomes a click.
func TestF32ToS16(t *testing.T) {
	tests := []struct {
		in   float32
		want int16
	}{
		{in: 0, want: 0},
		{in: 0.5, want: 16384},
		{in: -0.5, want: -16384},
		{in: 1.5, want: 32767},   // overshoot above the range
		{in: -1.5, want: -32768}, // and below it
	}
	for _, tt := range tests {
		if got := f32ToS16(tt.in); got != tt.want {
			t.Errorf("f32ToS16(%v) = %d, want %d", tt.in, got, tt.want)
		}
	}
}
