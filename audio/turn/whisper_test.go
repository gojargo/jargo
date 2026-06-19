package turn

import (
	"math"
	"testing"
)

// refAudio reproduces the deterministic 2-second signal used to generate the
// reference log-mel features from the Python implementation.
func refAudio() []float32 {
	const n = 32000
	audio := make([]float32, n)
	for i := range n {
		t := float64(i) / 16000
		audio[i] = float32(0.3*math.Sin(2*math.Pi*150*t) +
			0.2*math.Sin(2*math.Pi*440*t) +
			0.1*math.Sin(2*math.Pi*900*t))
	}
	return audio
}

func at(features []float32, mel, frame int) float32 {
	return features[mel*nFrames+frame]
}

// TestComputeLogMel checks the pure-Go feature extraction against values
// produced by the reference numpy implementation (Smart Turn v3's
// WhisperFeatureExtractor). The reference runs in float32; jargo runs in
// float64, so a small tolerance is allowed.
func TestComputeLogMel(t *testing.T) {
	features := computeLogMel(refAudio())

	if len(features) != nMels*nFrames {
		t.Fatalf("len(features) = %d, want %d", len(features), nMels*nFrames)
	}

	const tol = 5e-3
	cases := []struct {
		mel, frame int
		want       float32
	}{
		{0, 0, 1.61205},
		{10, 100, 1.588774},
		{40, 400, -0.245753},
		{79, 799, -0.245753},
		{20, 200, 0.928353},
		{60, 600, -0.245753},
		{0, 799, -0.245753},
		{79, 0, 0.361104},
	}
	for _, c := range cases {
		got := at(features, c.mel, c.frame)
		if math.Abs(float64(got-c.want)) > tol {
			t.Errorf("feature[%d,%d] = %v, want %v (tol %v)", c.mel, c.frame, got, c.want, tol)
		}
	}

	// Aggregate statistics over the whole matrix.
	var sum, sumSq, mn, mx float64
	mn, mx = math.Inf(1), math.Inf(-1)
	for _, v := range features {
		f := float64(v)
		sum += f
		sumSq += f * f
		mn = math.Min(mn, f)
		mx = math.Max(mx, f)
	}
	n := float64(len(features))
	mean := sum / n
	std := math.Sqrt(sumSq/n - mean*mean)

	checkStat(t, "mean", mean, -0.13441035, tol)
	checkStat(t, "std", std, 0.38163563, tol)
	checkStat(t, "min", mn, -0.24575278, tol)
	checkStat(t, "max", mx, 1.7542472, tol)
}

func checkStat(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %v, want %v (tol %v)", name, got, want, tol)
	}
}
