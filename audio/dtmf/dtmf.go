// Package dtmf generates the dual-tone multi-frequency signals a telephone
// keypad produces, as 16-bit mono PCM. A transport that cannot signal a keypress
// natively sends the tone as audio instead, which is what carries a keypress
// down a call that only has an audio path.
package dtmf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/gojargo/jargo/frames"
)

// ErrUnknownKey is returned for a key that is not on a telephone keypad and so
// has no tone pair.
//
//nolint:gochecknoglobals // sentinel error
var ErrUnknownKey = errors.New("dtmf: not a keypad key")

// ErrSampleRate is returned when the requested rate cannot carry a tone.
//
//nolint:gochecknoglobals // sentinel error
var ErrSampleRate = errors.New("dtmf: unusable sample rate")

// ToneDuration is how long one keypress occupies the audio stream: the tone
// itself and the pause that closes it. A keypress is a tone, not an instant, and
// a receiver that samples too short a burst does not register it.
const ToneDuration = 500 * time.Millisecond

// toneOn is how much of ToneDuration actually sounds. The rest is silence, and
// it is not padding: a receiver tells one keypress from the next by the gap
// between them, so a run of keys sounded back to back with no gap reads as
// fewer keys than were pressed. Sounding the tone for the whole of ToneDuration
// is what produces that run.
const toneOn = 300 * time.Millisecond

// tonePair is the two frequencies sounded together for one key. A keypad is a
// grid, and a key is named by the row tone and the column tone that cross at it,
// which is what lets a receiver tell a keypress from speech or noise that
// happens to land on one frequency.
type tonePair struct{ low, high float64 }

// pairs maps each key to the two frequencies that name it. The four rows are
// 697, 770, 852 and 941 Hz; the four columns are 1209, 1336, 1477 and 1633 Hz.
// The fourth column (A to D) is not on a consumer keypad but is part of the
// signaling standard.
//
//nolint:gochecknoglobals // read-only table
var pairs = map[frames.KeypadEntry]tonePair{
	frames.KeypadOne: {697, 1209}, frames.KeypadTwo: {697, 1336},
	frames.KeypadThree: {697, 1477}, frames.KeypadA: {697, 1633},

	frames.KeypadFour: {770, 1209}, frames.KeypadFive: {770, 1336},
	frames.KeypadSix: {770, 1477}, frames.KeypadB: {770, 1633},

	frames.KeypadSeven: {852, 1209}, frames.KeypadEight: {852, 1336},
	frames.KeypadNine: {852, 1477}, frames.KeypadC: {852, 1633},

	frames.KeypadStar: {941, 1209}, frames.KeypadZero: {941, 1336},
	frames.KeypadPound: {941, 1477}, frames.KeypadD: {941, 1633},
}

// amplitude is how loud each of the two tones is, as a fraction of full scale.
// The two are summed, so the pair peaks at twice this, around a eighth of full
// scale.
//
// It is deliberately quiet. A keypress rides a call alongside speech, and the
// level a receiver expects is the one the network has always carried, roughly
// -18 dBFS at the peak. A tone sounded near full scale is not easier to detect:
// it is loud enough to be shaved by the compander on a PSTN leg, and the
// distortion that produces spreads energy onto the frequencies that name other
// keys.
const amplitude = 0.0625

// Tone renders one key as 16-bit little-endian mono PCM at sampleRate, lasting
// ToneDuration: toneOn of the two frequencies sounded together, then silence for
// the rest. A key that is not on a keypad is an error rather than silence, since
// silence would be indistinguishable from a tone nobody heard.
func Tone(button frames.KeypadEntry, sampleRate int) ([]byte, error) {
	pair, ok := pairs[button]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownKey, button)
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("%w: %d", ErrSampleRate, sampleRate)
	}

	samples := int(float64(sampleRate) * ToneDuration.Seconds())
	sounded := int(float64(sampleRate) * toneOn.Seconds())
	pcm := make([]byte, samples*2)
	for i := range sounded {
		t := float64(i) / float64(sampleRate)
		v := amplitude * (math.Sin(2*math.Pi*pair.low*t) + math.Sin(2*math.Pi*pair.high*t))
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(int16(v*math.MaxInt16))) //nolint:gosec // bounded by amplitude
	}
	return pcm, nil
}

// Key reports whether button is a key this package can sound.
func Key(button frames.KeypadEntry) bool {
	_, ok := pairs[button]
	return ok
}
