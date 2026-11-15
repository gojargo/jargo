package kokoro

import (
	"math"
)

// The silence trimming parameters: a frame is silent when its RMS is topDB
// decibels or more below the loudest frame's, frames of trimFrameLength
// samples taken every trimHopLength.
const (
	trimTopDB       = 60
	trimFrameLength = 2048
	trimHopLength   = 512
	// trimAmin is the floor the powers are clamped to before taking their
	// logarithm, the square of an amplitude floor of 1e-5.
	trimAmin = 1e-10
)

// trim removes the leading and trailing silence of y: the frames whose RMS is
// trimTopDB below the loudest frame's. It returns the trimmed audio and the
// interval of y it covers. A signal of uniform loudness is not trimmed, and a
// signal with no frame above the threshold is trimmed to nothing.
//
// Each frame is centered on its hop, the signal padded with zeros by half a
// frame on either side.
func trim(y []float32) ([]float32, [2]int) {
	nonSilent := frameNonSilent(y)
	first, last := -1, -1
	for i, loud := range nonSilent {
		if loud {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	start, end := 0, 0
	if first >= 0 {
		// The end goes one frame past the last loud one.
		start = first * trimHopLength
		end = min(len(y), (last+1)*trimHopLength)
	}
	return y[start:end], [2]int{start, end}
}

// frameNonSilent reports, for each frame, whether it is louder than the
// threshold below the loudest frame.
func frameNonSilent(y []float32) []bool {
	rms := frameRMS(y)
	var ref float32
	for _, v := range rms {
		ref = max(ref, float32(math.Abs(float64(v))))
	}
	refDB := 10 * math.Log10(max(trimAmin, float64(ref*ref)))
	out := make([]bool, len(rms))
	for i, v := range rms {
		power := v * v
		db := 10*math.Log10(max(trimAmin, float64(power))) - refDB
		out[i] = db > -trimTopDB
	}
	return out
}

// frameRMS is the root-mean-square of each centered frame of y.
func frameRMS(y []float32) []float32 {
	pad := trimFrameLength / 2
	padded := len(y) + 2*pad
	nFrames := 1 + (padded-trimFrameLength)/trimHopLength
	at := func(i int) float32 {
		i -= pad
		if i < 0 || i >= len(y) {
			return 0
		}
		return y[i]
	}
	out := make([]float32, nFrames)
	for f := range nFrames {
		var sum float64
		for i := f * trimHopLength; i < f*trimHopLength+trimFrameLength; i++ {
			v := at(i)
			sum += float64(v * v)
		}
		power := float32(sum / trimFrameLength)
		out[f] = float32(math.Sqrt(float64(power)))
	}
	return out
}
