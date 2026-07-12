package onset

import (
	"encoding/binary"
	"math"
	"testing"
)

// pcm builds little-endian 16-bit mono PCM from normalized samples.
func pcm(samples []float64) []byte {
	out := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(s*32767)))
	}
	return out
}

// tone appends n samples of a sine at freq/amp to dst.
func tone(dst []float64, n, sampleRate int, freq, amp float64) []float64 {
	for i := range n {
		dst = append(dst, amp*math.Sin(2*math.Pi*freq*float64(i)/float64(sampleRate)))
	}
	return dst
}

func TestDetectLeadingSilence(t *testing.T) {
	const rate = 16000
	silence := 1600 // 100 ms
	var sig []float64
	sig = append(sig, make([]float64, silence)...)
	sig = tone(sig, rate/2, rate, 440, 0.3) // 500 ms of speech-level tone

	got := Detect(pcm(sig), rate, 1)
	if got < 0 {
		t.Fatalf("expected an onset, got none")
	}
	// Onset should land near the silence/tone boundary, within one analysis
	// window's slack on either side.
	if got < silence-160 || got > silence+320 {
		t.Fatalf("onset = %d samples, want ~%d (%.0f ms off)", got, silence,
			float64(got-silence)/rate*1000)
	}
}

func TestDetectAllSilence(t *testing.T) {
	if got := Detect(pcm(make([]float64, 16000)), 16000, 1); got != -1 {
		t.Fatalf("all-silence onset = %d, want -1", got)
	}
}

func TestDetectEmptyAndTooShort(t *testing.T) {
	if got := Detect(nil, 16000, 1); got != -1 {
		t.Fatalf("empty onset = %d, want -1", got)
	}
	// Fewer samples than one analysis window.
	if got := Detect(pcm(make([]float64, 10)), 16000, 1); got != -1 {
		t.Fatalf("too-short onset = %d, want -1", got)
	}
}

func TestDetectImmediateSpeech(t *testing.T) {
	// No leading silence: onset should be at (or very near) zero.
	sig := tone(nil, 16000/2, 16000, 440, 0.3)
	got := Detect(pcm(sig), 16000, 1)
	if got < 0 || got > 160 {
		t.Fatalf("immediate-speech onset = %d, want ~0", got)
	}
}
