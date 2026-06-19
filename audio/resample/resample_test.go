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
	r := resample.New(48000, 48000, 1)
	in := sine(440, 48000, 100)
	out := r.Process(in)
	if len(out) != len(in) {
		t.Fatalf("identity changed length: %d -> %d", len(in), len(out))
	}
}

func TestUpsample24kTo48kDoublesLength(t *testing.T) {
	r := resample.New(24000, 48000, 1)
	// Feed 480 input frames (20 ms @ 24k) -> expect ~960 output frames (20 ms @ 48k).
	in := sine(440, 24000, 480)
	out := r.Process(in)
	got := len(out) / 2
	if got < 955 || got > 965 {
		t.Fatalf("24k->48k of 480 frames produced %d frames, want ~960", got)
	}
}

func TestStreamingContinuityRoughlyDoubles(t *testing.T) {
	r := resample.New(24000, 48000, 1)
	total := 0
	// Stream ten 480-frame buffers (4800 input frames @ 24k); the total output
	// should be ~2x, about 9600 frames @ 48k.
	for range 10 {
		out := r.Process(sine(440, 24000, 480))
		total += len(out) / 2
	}
	if total < 9550 || total > 9650 {
		t.Fatalf("streamed 4800 input frames -> %d output frames, want ~9600", total)
	}
}

func TestDownsample48kTo16k(t *testing.T) {
	r := resample.New(48000, 16000, 1)
	in := sine(440, 48000, 960) // 20 ms @ 48k -> ~320 frames @ 16k
	out := r.Process(in)
	got := len(out) / 2
	if got < 316 || got > 324 {
		t.Fatalf("48k->16k of 960 frames produced %d frames, want ~320", got)
	}
}
