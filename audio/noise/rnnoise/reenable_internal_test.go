package rnnoise

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
)

// sine is n samples of a 440 Hz tone at rate, as 16-bit PCM.
func sine(n, rate int) []byte {
	b := make([]byte, 2*n)
	for i := range n {
		v := int16(math.Sin(2*math.Pi*440*float64(i)/float64(rate)) * 8000)
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return b
}

// TestReenablingDiscardsWhatTheResamplersHeld checks switching filtering back
// on starts the conversion afresh: what the resamplers held from before it was
// switched off is not spliced into the new audio. It needs no denoiser, only
// the resamplers around it.
func TestReenablingDiscardsWhatTheResamplersHeld(t *testing.T) {
	cfg := resample.Config{}
	newResampler := func(from, to int) *resample.Resampler {
		r, err := resample.NewWithConfig(from, to, 1, cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(r.Close)
		return r
	}
	f := &filter{filtering: true, up: newResampler(16000, rnnoiseRate), down: newResampler(rnnoiseRate, 16000)}

	// Audio converted before filtering is switched off leaves a tail held.
	f.up.Process(sine(1000, 16000))
	f.down.Process(sine(3000, rnnoiseRate))
	if err := f.ProcessFrame(t.Context(), frames.NewFilterEnableFrame(false)); err != nil {
		t.Fatal(err)
	}
	if err := f.ProcessFrame(t.Context(), frames.NewFilterEnableFrame(true)); err != nil {
		t.Fatal(err)
	}

	next := sine(320, 16000)
	fresh := newResampler(16000, rnnoiseRate)
	if got, want := f.up.Process(next), fresh.Process(next); !bytes.Equal(got, want) {
		t.Errorf("after re-enabling, the input resampler gave %d bytes unlike a fresh one's %d", len(got), len(want))
	}
}
