// Package rimepcm converts the audio Rime's services are given to what they take:
// signed 16-bit little-endian mono PCM at 16 kHz. Speech-to-text and the Prism
// conversations share it.
package rimepcm

import (
	"encoding/binary"
	"slices"
)

// WireRate is the rate Rime takes audio at.
const WireRate = 16000

// Format is headerless signed 16-bit little-endian PCM input, interleaved when
// stereo.
type Format struct {
	// SampleRate is the input frames per second: 8000, 16000, 24000 or 48000.
	SampleRate int
	// Channels is 1 for mono or 2 for interleaved stereo.
	Channels int
}

// Valid reports whether Rime takes audio in this format.
func (f Format) Valid() bool {
	return slices.Contains([]int{8000, 16000, 24000, 48000}, f.SampleRate) && (f.Channels == 1 || f.Channels == 2)
}

// RateConverter converts 16-bit mono PCM from one rate to another by linear
// interpolation between neighboring samples, carrying its state from one chunk
// to the next so a stream converts as one. It is the conversion Rime's own
// client applies to audio it is given at another rate than 16 kHz, and like it
// adds no delay: the first sample out is the first sample in.
type RateConverter struct {
	inRate, outRate int64
	// d is the position of the next output sample between prev and cur, in
	// units of 1/outRate of an input sample; negative means another input
	// sample is needed first.
	d         int64
	prev, cur int64
}

// NewRateConverter converts from inRate to outRate.
func NewRateConverter(inRate, outRate int) *RateConverter {
	g := int64(gcd(inRate, outRate))
	c := &RateConverter{inRate: int64(inRate) / g, outRate: int64(outRate) / g}
	c.d = -c.outRate
	return c
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// Process converts a chunk of whole samples.
func (c *RateConverter) Process(pcm []byte) []byte {
	out := make([]byte, 0, len(pcm)*int(c.outRate)/int(c.inRate)+4)
	for i := 0; ; {
		for c.d < 0 {
			if i+1 >= len(pcm) {
				return out
			}
			c.prev = c.cur
			// Scaled to 32 bits, as the conversion is carried out at that width.
			c.cur = int64(int16(binary.LittleEndian.Uint16(pcm[i:]))) << 16
			i += 2
			c.d += c.outRate
		}
		for c.d >= 0 {
			// Truncated toward zero, then scaled back down with an arithmetic
			// shift, as the 32-bit conversion does.
			v := (c.prev*c.d + c.cur*(c.outRate-c.d)) / c.outRate
			out = binary.LittleEndian.AppendUint16(out, uint16(int16(v>>16)))
			c.d -= c.inRate
		}
	}
}

// InputConverter converts complete frames in one format to 16 kHz mono. Each
// stream of audio owns one, which carries the resampling state from one chunk
// to the next.
type InputConverter struct {
	format Format
	rate   *RateConverter
}

// NewInputConverter converts audio in format.
func NewInputConverter(format Format) *InputConverter {
	c := &InputConverter{format: format}
	if format.SampleRate != WireRate {
		c.rate = NewRateConverter(format.SampleRate, WireRate)
	}
	return c
}

// Format is the format the converter takes.
func (c *InputConverter) Format() Format { return c.format }

// Clone snapshots the resampling state, so a send that is given up can leave
// the stream where it was.
func (c *InputConverter) Clone() *InputConverter {
	clone := &InputConverter{format: c.format}
	if c.rate != nil {
		rate := *c.rate
		clone.rate = &rate
	}
	return clone
}

// Process converts a chunk of whole frames.
func (c *InputConverter) Process(data []byte) []byte {
	if c.format.Channels == 2 {
		data = toMono(data)
	}
	if c.rate != nil {
		data = c.rate.Process(data)
	}
	return data
}

// toMono averages the two channels of interleaved stereo, rounding down.
func toMono(stereo []byte) []byte {
	out := make([]byte, 0, len(stereo)/2)
	for i := 0; i+3 < len(stereo); i += 4 {
		left := int32(int16(binary.LittleEndian.Uint16(stereo[i:])))
		right := int32(int16(binary.LittleEndian.Uint16(stereo[i+2:])))
		out = binary.LittleEndian.AppendUint16(out, uint16(int16((left+right)>>1)))
	}
	return out
}
