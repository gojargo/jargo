// Package dtmf synthesizes DTMF (touch-tone) keypad tones and aggregates
// received keypresses. Tone generates the dual-tone PCM for a key so a bot can
// navigate an IVR; Aggregator collects the keys a caller presses into a string.
package dtmf

import (
	"math"

	"github.com/gojargo/jargo/frames"
)

// tones maps each keypad entry to its low and high DTMF frequencies in Hz. The
// standard keypad is a grid of four low (row) and four high (column) tones.
//
//nolint:gochecknoglobals // frequency table
var tones = map[frames.KeypadEntry][2]float64{
	frames.KeypadOne: {697, 1209}, frames.KeypadTwo: {697, 1336},
	frames.KeypadThree: {697, 1477}, frames.KeypadA: {697, 1633},
	frames.KeypadFour: {770, 1209}, frames.KeypadFive: {770, 1336},
	frames.KeypadSix: {770, 1477}, frames.KeypadB: {770, 1633},
	frames.KeypadSeven: {852, 1209}, frames.KeypadEight: {852, 1336},
	frames.KeypadNine: {852, 1477}, frames.KeypadC: {852, 1633},
	frames.KeypadStar: {941, 1209}, frames.KeypadZero: {941, 1336},
	frames.KeypadPound: {941, 1477}, frames.KeypadD: {941, 1633},
}

// amplitude scales each tone; two summed sines at 0.3 stay within the 16-bit
// range with headroom.
const amplitude = 0.3

// Tone synthesizes durationMs of the DTMF dual tone for button as 16-bit mono
// little-endian PCM at sampleRate. An unknown button or a non-positive duration
// or sample rate returns nil.
func Tone(button frames.KeypadEntry, durationMs, sampleRate int) []byte {
	f, ok := tones[button]
	if !ok || durationMs <= 0 || sampleRate <= 0 {
		return nil
	}
	n := sampleRate * durationMs / 1000
	out := make([]byte, n*2)
	for i := range n {
		t := float64(i) / float64(sampleRate)
		v := amplitude * (math.Sin(2*math.Pi*f[0]*t) + math.Sin(2*math.Pi*f[1]*t)) / 2
		s := int16(v * 32767)
		out[2*i] = byte(s)
		out[2*i+1] = byte(uint16(s) >> 8)
	}
	return out
}
