package rnnoise

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/gojargo/jargo/frames"
)

// stubDenoiser stands in for librnnoise for the rest of the test with a
// denoiser that passes each 480-sample frame through unchanged, counting the
// frames it is given. The filter around it is the real one, so what is tested
// is the resampling to 48 kHz and back.
func stubDenoiser(t *testing.T) *int {
	t.Helper()
	prevCreate, prevDestroy := rnnoiseCreate, rnnoiseDestroy
	prevFrameSize, prevProcess := rnnoiseGetFrameSize, rnnoiseProcessFrame
	t.Cleanup(func() {
		rnnoiseCreate, rnnoiseDestroy = prevCreate, prevDestroy
		rnnoiseGetFrameSize, rnnoiseProcessFrame = prevFrameSize, prevProcess
	})

	const frameSize = 480
	var frames int
	rnnoiseCreate = func(uintptr) uintptr { return 1 }
	rnnoiseDestroy = func(uintptr) {}
	rnnoiseGetFrameSize = func() int32 { return frameSize }
	rnnoiseProcessFrame = func(_ uintptr, out, in *float32) float32 {
		// Like the C function it stands in for, it is handed pointers to
		// frame-sized buffers.
		copy(unsafe.Slice(out, frameSize), unsafe.Slice(in, frameSize)) //nolint:gosec // frame-sized buffers, as above

		frames++
		return 0.99
	}
	return &frames
}

// peakFrequency returns the frequency of the strongest bin of the discrete
// Fourier transform of samples taken at rate. Each bin's power is computed with
// the Goertzel recurrence.
func peakFrequency(samples []float64, rate int) float64 {
	n := len(samples)
	best, bestBin := -1.0, 0
	for k := 0; k <= n/2; k++ {
		w := 2 * math.Pi * float64(k) / float64(n)
		coeff := 2 * math.Cos(w)
		var s1, s2 float64
		for _, x := range samples {
			s1, s2 = x+coeff*s1-s2, s1
		}
		if p := s1*s1 + s2*s2 - coeff*s1*s2; p > best {
			best, bestBin = p, k
		}
	}
	return float64(bestBin) * float64(rate) / float64(n)
}

// Audio at 16 kHz goes up to the denoiser's 48 kHz and back down: about as
// much comes out as went in, and a 440 Hz tone keeps its pitch.
func TestResamplingTo48kHzAndBackKeepsLengthAndPitch(t *testing.T) {
	processed := stubDenoiser(t)
	const rate = 16000

	// One second of a full-scale 440 Hz tone.
	in := make([]byte, 2*rate)
	for i := range rate {
		v := int16(math.Sin(2*math.Pi*440*float64(i)/rate) * 32767)
		binary.LittleEndian.PutUint16(in[2*i:], uint16(v))
	}

	// Built as New builds it, without loading librnnoise: the stubs stand in
	// for it, and loading it would bind the real functions over them.
	f := &filter{filtering: true}
	ctx := t.Context()
	if err := f.Start(ctx, rate); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := f.ProcessFrame(ctx, frames.NewFilterEnableFrame(true)); err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}

	// 10 ms chunks, as a transport delivers them.
	var out []byte
	for i := 0; i < len(in); i += 320 {
		chunk, err := f.Filter(ctx, in[i:min(i+320, len(in))])
		if err != nil {
			t.Fatalf("Filter: %v", err)
		}
		out = append(out, chunk...)
	}
	if err := f.Stop(ctx); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	expectedFrames := len(in) / 2 * rnnoiseRate / rate / 480
	if *processed < expectedFrames-5 {
		t.Errorf("denoiser was given %d frames, want at least %d", *processed, expectedFrames-5)
	}

	// Within 0.2 s of the input, allowing for the resamplers' latency.
	tolerance := int(0.2 * rate * 2)
	if len(out) < len(in)-tolerance || len(out) > len(in)+tolerance {
		t.Fatalf("output is %d bytes, want %d within %d", len(out), len(in), tolerance)
	}

	samples := make([]float64, len(out)/2)
	for i := range samples {
		samples[i] = float64(int16(binary.LittleEndian.Uint16(out[2*i:])))
	}
	segment := samples[len(samples)/4 : 3*len(samples)/4]
	if peak := peakFrequency(segment, rate); math.Abs(peak-440) >= 50 {
		t.Errorf("peak frequency %.2f Hz, want 440 Hz within 50", peak)
	}
}
