// Package loudness measures how loud audio is, to ITU-R BS.1770 (the EBU R128
// standard), and normalizes the result to a 0..1 scale.
//
// It is what a voice detector gates on when it is asked for a minimum volume:
// loudness weighted for how the ear actually hears, rather than raw amplitude,
// so a low rumble and a spoken word of the same amplitude are not treated alike.
//
// It is its own package, rather than sitting with the other audio helpers, so
// that audio/vad can use it. The audio package refers to frames, and frames
// refers to audio/vad, so anything audio/vad depends on has to stay clear of
// both.
package loudness

import (
	"encoding/binary"
	"math"
)

// Reference points from the standard. Block loudness is offset by -0.691 dB, a
// gating block is ignored below the absolute threshold, and the relative
// threshold sits this far below the ungated mean.
const (
	loudnessOffset = -0.691
	absoluteGate   = -70.0
	relativeGate   = -10.0
)

// The K-weighting filter pair: a high shelf lifting the treble for the head's
// acoustic response, and a high pass discarding the rumble below speech.
const (
	shelfGainDB = 4.0
	shelfFreq   = 1500.0
	passFreq    = 38.0
	passQ       = 0.5
)

// Volume smoothing. A volume reading is smoothed against the one before it so a
// single loud or quiet chunk does not swing the decision on its own.
const (
	// SmoothingFactor weights a new volume reading against the running one.
	SmoothingFactor = 0.2
	// loudnessFloor and loudnessCeiling are the range volume is normalized from.
	// Samples are measured at their 16-bit magnitudes rather than scaled to
	// unity, which is what puts speech inside this range.
	loudnessFloor   = -20.0
	loudnessCeiling = 80.0
)

// biquad is a second-order section with its coefficients normalized so that
// a0 is 1.
type biquad struct {
	b0, b1, b2 float64
	a1, a2     float64
}

// apply filters x, starting from rest, and returns the result.
func (f biquad) apply(x []float64) []float64 {
	out := make([]float64, len(x))
	var z1, z2 float64
	for i, xn := range x {
		yn := f.b0*xn + z1
		z1 = f.b1*xn - f.a1*yn + z2
		z2 = f.b2*xn - f.a2*yn
		out[i] = yn
	}
	return out
}

// highShelf builds a shelving section lifting everything above fc by gainDB.
func highShelf(gainDB, q, fc float64, rate int) biquad {
	a := math.Pow(10, gainDB/40)
	w0 := 2 * math.Pi * (fc / float64(rate))
	alpha := math.Sin(w0) / (2 * q)
	cos := math.Cos(w0)
	sq := 2 * math.Sqrt(a) * alpha

	b0 := a * ((a + 1) + (a-1)*cos + sq)
	b1 := -2 * a * ((a - 1) + (a+1)*cos)
	b2 := a * ((a + 1) + (a-1)*cos - sq)
	a0 := (a + 1) - (a-1)*cos + sq
	a1 := 2 * ((a - 1) - (a+1)*cos)
	a2 := (a + 1) - (a-1)*cos - sq

	return biquad{b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0}
}

// highPass builds a section discarding everything below fc.
func highPass(q, fc float64, rate int) biquad {
	w0 := 2 * math.Pi * (fc / float64(rate))
	alpha := math.Sin(w0) / (2 * q)
	cos := math.Cos(w0)

	b0 := (1 + cos) / 2
	b1 := -(1 + cos)
	b2 := (1 + cos) / 2
	a0 := 1 + alpha
	a1 := -2 * cos
	a2 := 1 - alpha

	return biquad{b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0}
}

// Meter measures the loudness of audio at one sample rate.
type Meter struct {
	rate      int
	blockSize float64 // gating block length, in seconds
	overlap   float64 // how much of a block the next one repeats
	shelf     biquad
	pass      biquad
}

// NewMeter builds a meter for a sample rate, measuring over blocks of blockSize
// seconds that overlap by three quarters, as the standard specifies.
func NewMeter(rate int, blockSize float64) *Meter {
	return &Meter{
		rate:      rate,
		blockSize: blockSize,
		overlap:   0.75,
		shelf:     highShelf(shelfGainDB, 1/math.Sqrt2, shelfFreq, rate),
		pass:      highPass(passQ, passFreq, rate),
	}
}

// IntegratedLoudness measures the gated loudness of a single channel in LUFS.
// It returns negative infinity when every block falls below the gate, which is
// to say when there is nothing there to measure.
func (m *Meter) IntegratedLoudness(samples []float64) float64 {
	if len(samples) == 0 || m.rate <= 0 || m.blockSize <= 0 {
		return math.Inf(-1)
	}

	// Weight for how the ear hears before measuring anything.
	x := m.pass.apply(m.shelf.apply(samples))

	step := 1 - m.overlap
	total := float64(len(samples)) / float64(m.rate)
	blocks := int(math.Round((total-m.blockSize)/(m.blockSize*step))) + 1
	if blocks < 1 {
		return math.Inf(-1)
	}

	// Mean square of each block.
	power := make([]float64, blocks)
	loud := make([]float64, blocks)
	for j := range power {
		lo := int(m.blockSize * (float64(j) * step) * float64(m.rate))
		hi := min(int(m.blockSize*(float64(j)*step+1)*float64(m.rate)), len(x))
		var sum float64
		for _, v := range x[lo:hi] {
			sum += v * v
		}
		power[j] = sum / (m.blockSize * float64(m.rate))
		loud[j] = loudnessOffset + 10*math.Log10(power[j])
	}

	// Gate against the absolute threshold, then against one relative to what
	// survived it, so a quiet passage does not drag the measurement down.
	mean, ok := gatedMean(power, loud, func(l float64) bool { return l >= absoluteGate })
	if !ok {
		return math.Inf(-1)
	}
	relative := loudnessOffset + 10*math.Log10(mean) + relativeGate

	mean, ok = gatedMean(power, loud, func(l float64) bool {
		return l > relative && l > absoluteGate
	})
	if !ok {
		return math.Inf(-1)
	}
	return loudnessOffset + 10*math.Log10(mean)
}

// gatedMean averages the power of the blocks whose loudness keeps, reporting
// false when none do.
func gatedMean(power, loud []float64, keep func(float64) bool) (float64, bool) {
	var sum float64
	n := 0
	for j, l := range loud {
		if keep(l) {
			sum += power[j]
			n++
		}
	}
	if n == 0 {
		return 0, false
	}
	return sum / float64(n), true
}

// Volume measures how loud a chunk of 16-bit mono PCM is, on a 0..1 scale where
// 0 is silent. The whole chunk is measured as one gating block.
func Volume(pcm []byte, sampleRate int) float64 {
	n := len(pcm) / 2
	if n == 0 || sampleRate <= 0 {
		return 0
	}
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = float64(int16(binary.LittleEndian.Uint16(pcm[2*i:])))
	}
	m := NewMeter(sampleRate, float64(n)/float64(sampleRate))
	return Normalize(m.IntegratedLoudness(samples), loudnessFloor, loudnessCeiling)
}

// Normalize maps a value from a range onto 0..1, clamped to it.
func Normalize(value, low, high float64) float64 {
	if high == low {
		return 0
	}
	n := (value - low) / (high - low)
	switch {
	case math.IsNaN(n) || n < 0:
		return 0
	case n > 1:
		return 1
	}
	return n
}

// Smooth blends a new reading into the running one, weighting the new reading by
// factor.
func Smooth(value, previous, factor float64) float64 {
	return previous + factor*(value-previous)
}
