package dtmf_test

import (
	"encoding/binary"
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"github.com/gojargo/jargo/audio/dtmf"
	"github.com/gojargo/jargo/frames"
)

const sampleRate = 8000

// goertzel reports how much energy the signal carries at freq. It is the
// standard way a receiver detects one DTMF frequency, so measuring the tone the
// way a receiver would is what says it is the right tone.
func goertzel(pcm []byte, freq float64, rate int) float64 {
	n := len(pcm) / 2
	k := 2 * math.Pi * freq / float64(rate)
	var s0, s1, s2 float64
	coeff := 2 * math.Cos(k)
	for i := range n {
		sample := float64(int16(binary.LittleEndian.Uint16(pcm[i*2:]))) / math.MaxInt16
		s0 = sample + coeff*s1 - s2
		s2, s1 = s1, s0
	}
	return cmplx.Abs(complex(s1-s2*math.Cos(k), -s2*math.Sin(k))) / float64(n)
}

// TestToneCarriesBothFrequencies checks each key sounds the two frequencies that
// name it on the keypad grid, and neither of the two that name its neighbors. A
// receiver identifies the key by that pair, so a tone missing one of them, or
// carrying a third, is a key nobody can read.
func TestToneCarriesBothFrequencies(t *testing.T) {
	cases := map[frames.KeypadEntry][2]float64{
		frames.KeypadOne:   {697, 1209},
		frames.KeypadFive:  {770, 1336},
		frames.KeypadNine:  {852, 1477},
		frames.KeypadZero:  {941, 1336},
		frames.KeypadStar:  {941, 1209},
		frames.KeypadPound: {941, 1477},
		frames.KeypadD:     {941, 1633},
	}
	for button, want := range cases {
		pcm, err := dtmf.Tone(button, sampleRate)
		if err != nil {
			t.Fatalf("Tone(%q): %v", button, err)
		}
		for _, freq := range want {
			if got := goertzel(pcm, freq, sampleRate); got < 0.1 {
				t.Errorf("key %q carries %.0f Hz at %.4f, want it sounded", button, freq, got)
			}
		}
		// A frequency from another row or column must not be there.
		for _, absent := range []float64{697, 770, 852, 941, 1209, 1336, 1477, 1633} {
			if absent == want[0] || absent == want[1] {
				continue
			}
			if got := goertzel(pcm, absent, sampleRate); got > 0.02 {
				t.Errorf("key %q carries %.0f Hz at %.4f, want it silent", button, absent, got)
			}
		}
	}
}

// TestToneLength checks the tone lasts long enough to be registered. A receiver
// sampling too short a burst does not hear the key at all.
func TestToneLength(t *testing.T) {
	for _, rate := range []int{8000, 16000, 48000} {
		pcm, err := dtmf.Tone(frames.KeypadSeven, rate)
		if err != nil {
			t.Fatalf("Tone: %v", err)
		}
		wantSamples := rate / 2 // 500ms
		if got := len(pcm) / 2; got != wantSamples {
			t.Errorf("rate %d produced %d samples, want %d", rate, got, wantSamples)
		}
	}
}

// TestToneDoesNotClip checks the two summed tones stay inside the sample range.
// A tone that clips is distorted, and distortion spreads energy onto
// frequencies the receiver reads as a different key.
func TestToneDoesNotClip(t *testing.T) {
	pcm, err := dtmf.Tone(frames.KeypadFive, sampleRate)
	if err != nil {
		t.Fatalf("Tone: %v", err)
	}
	for i := range len(pcm) / 2 {
		v := int16(binary.LittleEndian.Uint16(pcm[i*2:]))
		if v == math.MaxInt16 || v == math.MinInt16 {
			t.Fatalf("sample %d reached the rail at %d, so the tone clips", i, v)
		}
	}
}

// TestUnknownKey checks a key that is not on a keypad is refused rather than
// sounded as silence, which nobody could tell from a tone that was not heard.
func TestUnknownKey(t *testing.T) {
	if _, err := dtmf.Tone(frames.KeypadEntry("X"), sampleRate); !errors.Is(err, dtmf.ErrUnknownKey) {
		t.Errorf("Tone(X) error = %v, want ErrUnknownKey", err)
	}
	if _, err := dtmf.Tone(frames.KeypadOne, 0); err == nil {
		t.Error("Tone at rate 0 = nil, want an error")
	}
	if !dtmf.Key(frames.KeypadA) {
		t.Error("Key(A) = false: A to D are part of the signaling standard")
	}
	if dtmf.Key(frames.KeypadEntry("X")) {
		t.Error("Key(X) = true, want false")
	}
}
