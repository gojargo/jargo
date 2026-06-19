package resample_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gojargo/jargo/audio/resample"
)

func sine(freq float64, rate, n int) []byte {
	b := make([]byte, n*2)
	for i := range n {
		v := math.Sin(2*math.Pi*freq*float64(i)/float64(rate)) * 0.5 * math.MaxInt16
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(v)))
	}
	return b
}

func TestIdentityWhenRatesMatch(t *testing.T) {
	r, err := resample.New(48000, 48000, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer r.Close()
	in := sine(440, 48000, 100)
	out := r.Process(in)
	if len(out) != len(in) {
		t.Fatalf("identity changed length: %d -> %d", len(in), len(out))
	}
}

// streamFrames runs `buffers` equal-sized buffers through r and returns the
// total output frames. Streaming a long signal makes libsoxr's fixed filter
// delay negligible relative to the total, so the output-to-input frame ratio
// converges to outRate/inRate even though any single call is off by the delay.
func streamFrames(r *resample.Resampler, freq float64, inRate, framesPerBuffer, buffers int) int {
	total := 0
	for range buffers {
		out := r.Process(sine(freq, inRate, framesPerBuffer))
		total += len(out) / 2
	}
	return total
}

// assertRatio checks got is within 2% of inFrames*outRate/inRate, tolerating
// libsoxr's filter delay.
func assertRatio(t *testing.T, got, inFrames, inRate, outRate int) {
	t.Helper()
	want := float64(inFrames) * float64(outRate) / float64(inRate)
	if math.Abs(float64(got)-want) > 0.02*want {
		t.Fatalf("got %d output frames, want ~%.0f (±2%%)", got, want)
	}
}

func TestUpsample24kTo48k(t *testing.T) {
	r, err := resample.New(24000, 48000, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer r.Close()
	// 500 * 480 = 240000 input frames @ 24k -> ~480000 frames @ 48k. The stream
	// is long enough that soxr's fixed filter delay is well under the tolerance.
	got := streamFrames(r, 440, 24000, 480, 500)
	assertRatio(t, got, 240000, 24000, 48000)
}

func TestDownsample48kTo16k(t *testing.T) {
	r, err := resample.New(48000, 16000, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer r.Close()
	// 500 * 960 = 480000 input frames @ 48k -> ~160000 frames @ 16k.
	got := streamFrames(r, 440, 48000, 960, 500)
	assertRatio(t, got, 480000, 48000, 16000)
}

func TestCloseIsIdempotent(t *testing.T) {
	r, err := resample.New(24000, 48000, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	r.Close()
	r.Close() // must not panic
}
