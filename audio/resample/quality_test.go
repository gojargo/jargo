package resample_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gojargo/jargo/audio/resample"
)

// minSINAD is the transparency floor in dB: a resampled pure tone must stay this
// far above its own distortion + noise. Both backends clear it with margin
// (pure-Go ~91 dB, libsoxr ~86 dB); the guard catches a regression that would
// add the audible harmonics/aliasing heard as a "metallic" voice.
const minSINAD = 60.0

// encodeS16 encodes float samples in [-1,1) to interleaved S16LE bytes (mono).
func encodeS16(samples []float64) []byte {
	b := make([]byte, len(samples)*2)
	for i, v := range samples {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(v*32767)))
	}
	return b
}

// decodeS16 decodes S16LE bytes to float64 in [-1,1].
func decodeS16(b []byte) []float64 {
	out := make([]float64, len(b)/2)
	for i := range out {
		out[i] = float64(int16(binary.LittleEndian.Uint16(b[i*2:]))) / 32768
	}
	return out
}

// sinad fits a pure sinusoid of frequency f (Hz, at rate fs) to x by least
// squares and returns 10*log10(signal / residual) in dB. A clean resampled tone
// scores high; harmonic/aliasing distortion drives the residual up and the
// score down.
func sinad(x []float64, f, fs float64) float64 {
	w := 2 * math.Pi * f / fs
	var cc, ss, cs, xc, xs float64
	for n, v := range x {
		c, s := math.Cos(w*float64(n)), math.Sin(w*float64(n))
		cc += c * c
		ss += s * s
		cs += c * s
		xc += v * c
		xs += v * s
	}
	det := cc*ss - cs*cs
	a := (xc*ss - xs*cs) / det
	b := (cc*xs - cs*xc) / det
	var sig, res float64
	for n, v := range x {
		fit := a*math.Cos(w*float64(n)) + b*math.Sin(w*float64(n))
		sig += fit * fit
		res += (v - fit) * (v - fit)
	}
	if res == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(sig/res)
}

// TestTransparency upsamples a pure 1 kHz tone 24k->48k (the TTS->WebRTC path)
// both in one call and as small streaming chunks, and asserts each stays above
// the transparency floor. Streaming must match one-shot: a per-chunk boundary
// discontinuity would drop the SINAD and sound metallic. Runs against whichever
// backend is built (default pure-Go, or libsoxr with -tags libsoxr).
func TestTransparency(t *testing.T) {
	const (
		inRate, outRate = 24000, 48000
		freq            = 1000.0
		inN             = 24000 // 1 s
	)
	in := make([]float64, inN)
	for i := range in {
		in[i] = 0.5 * math.Sin(2*math.Pi*freq*float64(i)/inRate)
	}
	inBytes := encodeS16(in)

	// steadySINAD measures the middle half of out, skipping filter-delay edges.
	steadySINAD := func(out []byte) float64 {
		o := decodeS16(out)
		return sinad(o[len(o)/4:len(o)*3/4], freq, outRate)
	}

	r1, err := resample.New(inRate, outRate, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer r1.Close()
	if got := steadySINAD(r1.Process(inBytes)); got < minSINAD {
		t.Errorf("one-shot SINAD = %.1f dB, want >= %.0f dB", got, minSINAD)
	}

	for _, chunk := range []int{480, 320, 160, 64} {
		r, err := resample.New(inRate, outRate, 1)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		var acc []byte
		for off := 0; off < len(inBytes); off += chunk * 2 {
			end := min(off+chunk*2, len(inBytes))
			acc = append(acc, r.Process(inBytes[off:end])...)
		}
		r.Close()
		if got := steadySINAD(acc); got < minSINAD {
			t.Errorf("streaming chunk=%d SINAD = %.1f dB, want >= %.0f dB", chunk, got, minSINAD)
		}
	}
}
