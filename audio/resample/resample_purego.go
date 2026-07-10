//go:build !libsoxr

package resample

import (
	"encoding/binary"
	"fmt"

	gore "github.com/gojargo/go-resample"
)

// converter is the go-resample algorithm used by the default build.
// SincMediumQuality is transparent for virtually all real-world audio at
// moderate CPU cost — the closest match to libsoxr's HQ preset.
const converter = gore.SincMediumQuality

// Resampler converts a stream of interleaved S16LE PCM from one sample rate to
// another using the pure-Go github.com/gojargo/go-resample converter (no cgo).
// This is the default build; `-tags libsoxr` swaps in libsoxr (see
// resample_soxr.go). Create one per audio stream with New; it is not safe for
// concurrent use. Close is a no-op kept for API parity with the libsoxr build.
type Resampler struct {
	inRate   int
	outRate  int
	channels int
	ratio    float64
	r        *gore.Resampler // nil when inRate == outRate (passthrough)

	inF  []float32 // reused S16LE->float32 decode scratch
	outF []float32 // reused resampler output scratch
}

// New returns a Resampler from inRate to outRate for the given channel count.
// When inRate equals outRate the Resampler passes audio through unchanged and
// allocates no converter.
func New(inRate, outRate, channels int) (*Resampler, error) {
	if channels < 1 {
		channels = 1
	}
	r := &Resampler{inRate: inRate, outRate: outRate, channels: channels}
	if inRate == outRate {
		return r, nil
	}
	conv, err := gore.New(converter, channels)
	if err != nil {
		return nil, fmt.Errorf("resample: new %d->%d ch=%d: %w", inRate, outRate, channels, err)
	}
	r.r = conv
	r.ratio = float64(outRate) / float64(inRate)
	return r, nil
}

// Process resamples one buffer of interleaved S16LE PCM and returns the
// resampled audio. When the input and output rates match it returns the input
// unchanged; otherwise the returned slice is freshly allocated and owned by the
// caller. Because the converter has filter delay, the first calls of a stream
// emit slightly fewer frames than the rate ratio implies, which later calls
// make up.
func (r *Resampler) Process(in []byte) []byte {
	if r.r == nil {
		return in
	}
	bytesPerFrame := r.channels * 2
	inFrames := len(in) / bytesPerFrame
	if inFrames == 0 {
		return nil
	}
	inSamples := inFrames * r.channels

	// Decode interleaved S16LE -> float32 in [-1, 1).
	if cap(r.inF) < inSamples {
		r.inF = make([]float32, inSamples)
	}
	inF := r.inF[:inSamples]
	for i := range inF {
		inF[i] = float32(int16(binary.LittleEndian.Uint16(in[i*2:]))) / 32768
	}

	// The converter emits about inFrames*outRate/inRate frames per call; the
	// margin covers rounding and filter delay so a single call drains all of the
	// output the input can yield (Process consumes every input frame).
	outFrames := inFrames*r.outRate/r.inRate + 64
	outSamples := outFrames * r.channels
	if cap(r.outF) < outSamples {
		r.outF = make([]float32, outSamples)
	}
	outF := r.outF[:outSamples]

	d := gore.Data{
		In:           inF,
		InputFrames:  inFrames,
		Out:          outF,
		OutputFrames: outFrames,
		Ratio:        r.ratio,
	}
	if err := r.r.Process(&d); err != nil {
		return nil
	}

	// Encode float32 -> interleaved S16LE with rounding and clamping.
	n := d.OutputFramesGen * r.channels
	out := make([]byte, n*2)
	for i := range n {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(f32ToS16(outF[i])))
	}
	return out
}

// f32ToS16 converts a normalized float sample to S16 with rounding and
// clamping. Sinc interpolation can overshoot past [-1, 1), so the clamp is
// load-bearing, not defensive.
func f32ToS16(f float32) int16 {
	v := float64(f) * 32768
	if v >= 0 {
		v += 0.5
	} else {
		v -= 0.5
	}
	switch {
	case v > 32767:
		return 32767
	case v < -32768:
		return -32768
	default:
		return int16(v)
	}
}

// Close releases converter resources. The pure-Go converter holds none, so this
// is a no-op; it is safe to call more than once and exists for API parity with
// the libsoxr build.
func (r *Resampler) Close() {
	r.r = nil
}
