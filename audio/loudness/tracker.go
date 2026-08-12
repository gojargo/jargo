package loudness

import "math"

// windowSecs is the span of audio a Tracker measures. Loudness is measured over
// a BS.1770 gating block, which is 400 ms. The audio chunks flowing through a
// pipeline are shorter than that, so they accumulate into a rolling window of
// this size.
const windowSecs = 0.4

// Tracker measures the volume of an audio stream over a rolling window.
//
// Audio is fed in chunks of any size with Update, which retains the most recent
// windowSecs of it, and Volume measures that window. Volume reads 0 until the
// window holds enough audio for loudness to be measurable, and the window is
// discarded if the sample rate changes.
//
// Measuring costs a few hundred microseconds and grows with the sample rate, so
// it happens on read and the result is cached until more audio arrives. A caller
// that reports volume less often than it receives audio pays only for the reads.
//
// Audio is expected to be mono; interleaved channels would be measured as if
// they were consecutive samples.
//
// The zero value is ready to use.
type Tracker struct {
	sampleRate     int
	windowNumBytes int
	buf            []byte
	volume         float64
	// stale is set once the window holds audio that has not been measured yet.
	stale bool
}

// Volume is the volume of the audio in the rolling window, between 0 (quiet)
// and 1 (loud). It reads 0 until the window holds a measurable amount of audio.
func (t *Tracker) Volume() float64 {
	if t.stale {
		t.volume = Volume(t.buf, t.sampleRate)
		t.stale = false
	}
	return t.volume
}

// Update adds audio to the rolling window. Changing the sample rate discards
// what the window held, since it was measured against a different one.
func (t *Tracker) Update(audio []byte, sampleRate int) {
	if sampleRate != t.sampleRate {
		t.sampleRate = sampleRate
		// Rounded up so the window is never a sample short of a gating block,
		// which loudness rejects.
		t.windowNumBytes = int(math.Ceil(windowSecs*float64(sampleRate))) * 2
		t.Reset()
	}

	t.buf = append(t.buf, audio...)
	if len(t.buf) > t.windowNumBytes {
		t.buf = t.buf[len(t.buf)-t.windowNumBytes:]
	}
	if len(t.buf) == t.windowNumBytes {
		t.stale = true
	}
}

// Reset clears the rolling window and the tracked volume.
func (t *Tracker) Reset() {
	t.buf = t.buf[:0]
	t.volume = 0
	t.stale = false
}
