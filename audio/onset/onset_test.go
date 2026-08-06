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

// silence appends n silent samples to dst.
func silence(dst []float64, n int) []float64 {
	return append(dst, make([]float64, n)...)
}

// blip appends one sample at a realistic noise-floor level, the shape of the
// stray sample a TTS service can leave in its leading padding.
func blip(dst []float64) []float64 {
	return append(dst, 2370.0/32768.0)
}

// TestDetectRejectsAnIsolatedBlip covers the reason detection works on energy
// rather than on how loud any one sample is. A lone sample at the noise floor
// averages out across its window, so it never reads as the response starting.
func TestDetectRejectsAnIsolatedBlip(t *testing.T) {
	const rate = 16000
	var sig []float64
	sig = silence(sig, rate/5) // 200 ms
	sig = blip(sig)
	sig = silence(sig, rate/5)

	if got := Detect(pcm(sig), rate, 1); got != -1 {
		t.Errorf("a single noise-floor sample was heard as the response starting at %d, want none", got)
	}
}

// TestDetectRejectsABriefLoudBurst covers the other half of that rejection. A
// transient can be loud enough to clear the gate and still not be speech, so the
// energy has to hold above it for minVoicedMs before an onset counts.
func TestDetectRejectsABriefLoudBurst(t *testing.T) {
	const rate = 16000
	var sig []float64
	sig = silence(sig, rate/5)
	sig = tone(sig, rate/100, rate, 220, 8000.0/32768.0) // 10 ms, well under the minimum
	sig = silence(sig, rate/5)

	if got := Detect(pcm(sig), rate, 1); got != -1 {
		t.Errorf("a 10 ms burst was heard as the response starting at %d, want none", got)
	}
}

// TestDetectLocksOntoSpeechAfterABlip covers the two together: a blip early in
// the padding must not anchor the measurement, so the onset reported is the
// sustained audio that follows it rather than the noise that came first.
func TestDetectLocksOntoSpeechAfterABlip(t *testing.T) {
	const rate = 16000
	var sig []float64
	sig = silence(sig, rate/10) // 100 ms
	sig = blip(sig)
	sig = silence(sig, rate*15/100) // 150 ms, so the tone starts at ~250 ms
	sig = tone(sig, rate*3/10, rate, 220, 5000.0/32768.0)

	const wantAt = rate * 25 / 100 // 250 ms
	got := Detect(pcm(sig), rate, 1)
	if got < 0 {
		t.Fatal("expected an onset at the sustained tone, got none")
	}
	// Upstream allows 20 ms of slack here; the same window applies.
	if slack := rate * 2 / 100; got < wantAt-slack || got > wantAt+slack {
		t.Errorf("onset = %d samples (%.0f ms), want ~%d (250 ms): the blip anchored the measurement",
			got, float64(got)/rate*1000, wantAt)
	}
}

// TestDetectStereo covers interleaved input, which is downmixed before anything
// is measured. Reading a stereo buffer as mono would halve every position.
func TestDetectStereo(t *testing.T) {
	const rate = 16000
	var mono []float64
	mono = silence(mono, rate/10) // 100 ms
	mono = tone(mono, rate*3/10, rate, 220, 5000.0/32768.0)

	// Both channels carry the same signal, so the downmix reproduces it.
	stereo := make([]float64, 0, len(mono)*2)
	for _, s := range mono {
		stereo = append(stereo, s, s)
	}

	const wantAt = rate / 10
	got := Detect(pcm(stereo), rate, 2)
	if got < 0 {
		t.Fatal("expected an onset in the stereo buffer, got none")
	}
	if slack := rate * 2 / 100; got < wantAt-slack || got > wantAt+slack {
		t.Errorf("onset = %d samples (%.0f ms), want ~%d (100 ms): the channels were not downmixed",
			got, float64(got)/rate*1000, wantAt)
	}
}
