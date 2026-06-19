// Package resample converts S16LE PCM audio between sample rates. It is used to
// reconcile service sample rates (for example 24 kHz TTS audio) with the 48 kHz
// the WebRTC/Opus output path needs.
//
// The resampler uses linear interpolation and is stateful, carrying the last
// sample across calls so a continuous stream resamples without clicks at chunk
// boundaries. Pure Go, no cgo.
package resample

import "encoding/binary"

// Resampler converts a stream of S16LE PCM from one sample rate to another.
// Create one per audio stream; it is not safe for concurrent use.
type Resampler struct {
	inRate   int
	outRate  int
	channels int

	ratio   float64 // input frames advanced per output frame
	prev    []int16 // last input frame of the previous call, per channel
	hasPrev bool
	pos     float64 // fractional input position of the next output frame
}

// New returns a Resampler from inRate to outRate for the given channel count.
func New(inRate, outRate, channels int) *Resampler {
	if channels < 1 {
		channels = 1
	}
	return &Resampler{
		inRate:   inRate,
		outRate:  outRate,
		channels: channels,
		ratio:    float64(inRate) / float64(outRate),
		prev:     make([]int16, channels),
	}
}

// Process resamples one buffer of interleaved S16LE PCM and returns the
// resampled audio. When the input and output rates match it returns the input
// unchanged.
func (r *Resampler) Process(in []byte) []byte {
	if r.inRate == r.outRate {
		return in
	}
	ch := r.channels
	src := bytesToInt16(in)
	frames := len(src) / ch
	if frames == 0 {
		return nil
	}

	// Work in a frame index space where index 0 is the carried-over previous
	// frame and indices 1..frames are this buffer's frames. get reads a channel
	// sample at a work index.
	get := func(idx, c int) float64 {
		if idx == 0 {
			return float64(r.prev[c])
		}
		return float64(src[(idx-1)*ch+c])
	}

	if !r.hasPrev {
		// First call: seed prev with the first frame and start output at index
		// 1, so the first output frame equals the first input frame.
		copy(r.prev, src[:ch])
		r.hasPrev = true
		r.pos = 1
	}

	maxIdx := frames // highest valid work index (the last input frame)
	var out []int16
	for {
		i := int(r.pos)
		if i+1 > maxIdx {
			break // need both work[i] and work[i+1]
		}
		frac := r.pos - float64(i)
		for c := range ch {
			l := get(i, c)
			rr := get(i+1, c)
			out = append(out, int16(l+(rr-l)*frac))
		}
		r.pos += r.ratio
	}

	// The last input frame becomes the carried previous frame; rebase pos so it
	// sits at index 0 for the next call.
	copy(r.prev, src[(frames-1)*ch:])
	r.pos -= float64(frames)
	if r.pos < 0 {
		r.pos = 0
	}
	return int16ToBytes(out)
}

func bytesToInt16(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}
	return out
}

func int16ToBytes(s []int16) []byte {
	out := make([]byte, len(s)*2)
	for i, v := range s {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}
	return out
}
