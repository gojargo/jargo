package resample

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// The polyphase windowed-sinc filter both conversions run: a Kaiser-windowed
// sinc low-pass, split into one filter per phase of the rate ratio.
//
// A filter is fixed once made, so any number of streams can share one. A
// stream keeps the recent input of each channel and returns, on every call,
// exactly the audio its input amounts to at the output rate, carrying any
// fraction of a sample into the next call: equal input chunks come out as equal
// output chunks, the first one included. The filter delays the audio by a few
// milliseconds instead of holding output back. A whole clip, converted in one
// pass, comes out lined up with its input, taken to be silent before and after.
//
// Lengths are in frames: one sample for every channel.

const (
	// sincAttenuationDB is how much the filter attenuates what the output rate
	// cannot carry.
	sincAttenuationDB = 90.0
	// sincPassband is how much of the lower rate's band the filter keeps flat.
	sincPassband = 0.9
	// sincMaxTaps is the largest filter, in taps across all phases, a pair of
	// rates may need.
	sincMaxTaps = 1 << 22
)

// Errors a filter is refused with.
var (
	// ErrBadRate reports a sample rate that is not positive.
	ErrBadRate = errors.New("resample: sample rates must be positive")
	// ErrTooLarge reports a pair of rates whose filter would be too large.
	ErrTooLarge = errors.New("resample: the filter for these rates is too large")
)

// sincPhase is one phase's taps, reversed. A symmetric phase is computed on
// mirrored pairs of samples, holding only its first half.
type sincPhase struct {
	fold bool
	// len is the taps, padded to a multiple of 8, or, folded, the taps spanned.
	len int
	// hp is, folded, the taps of the first half, padded to a multiple of 8.
	hp   int
	taps []float32
}

// sincFilter holds everything a pair of rates needs.
type sincFilter struct {
	inRate, outRate int
	passthrough     bool
	up, down, width int
	// center is the middle tap, on the input upsampled by up.
	center    int64
	stepWhole int64
	stepFrac  int
	phases    []sincPhase
}

// newSincFilter designs the filter between two rates.
func newSincFilter(inRate, outRate int) (*sincFilter, error) {
	if inRate <= 0 || outRate <= 0 {
		return nil, fmt.Errorf("%w: %d -> %d", ErrBadRate, inRate, outRate)
	}
	f := &sincFilter{inRate: inRate, outRate: outRate}
	if inRate == outRate {
		f.passthrough = true
		return f, nil
	}
	divisor := gcd(inRate, outRate)
	f.up = outRate / divisor
	f.down = inRate / divisor
	f.stepWhole = int64(f.down / f.up)
	f.stepFrac = f.down % f.up

	upRate := float64(inRate) * float64(f.up)
	nyquist := float64(min(inRate, outRate)) / 2
	transition := nyquist * (1 - sincPassband)
	estimate := (sincAttenuationDB-7.95)/(2.285*2*math.Pi*transition/upRate) + 1
	whole := math.Ceil(estimate)
	if whole+float64(f.up) > sincMaxTaps {
		return nil, fmt.Errorf("%w: %d -> %d", ErrTooLarge, inRate, outRate)
	}
	// An odd number of taps puts the filter's middle, where each output sits,
	// on a whole upsampled sample. Zeros pad it to whole phases.
	taps := int64(whole) | 1
	f.width = int((taps + int64(f.up) - 1) / int64(f.up))
	length := int64(f.width) * int64(f.up)
	f.center = (taps - 1) / 2
	h := f.designTaps(length, taps)

	// Each output's phase is stepFrac past the last one's.
	f.phases = make([]sincPhase, f.up)
	row := make([]float32, f.width)
	for k := range f.up {
		p := int(int64(k) * int64(f.stepFrac) % int64(f.up))
		for j := range f.width {
			row[j] = float32(h[int64(f.width-1-j)*int64(f.up)+int64(p)])
		}
		f.phases[p] = newSincPhase(row, f.width)
	}
	return f, nil
}

// designTaps is a Kaiser-windowed sinc low-pass with taps coefficients, padded
// with zeros to length, to be split into one reversed filter per phase: phase
// p, tap j is h[(width - 1 - j) * up + p].
func (f *sincFilter) designTaps(length, taps int64) []float64 {
	upRate := float64(f.inRate) * float64(f.up)
	nyquist := float64(min(f.inRate, f.outRate)) / 2
	transition := nyquist * (1 - sincPassband)
	cutoff := nyquist - transition/2
	beta := 0.1102 * (sincAttenuationDB - 8.7)
	i0Beta := besselI0(beta)
	alpha := float64(taps-1) / 2

	h := make([]float64, length)
	sum := 0.0
	for k := range taps {
		x := 2 * cutoff / upRate * (float64(k) - alpha)
		sinc := 1.0
		if x != 0 {
			sinc = math.Sin(math.Pi*x) / (math.Pi * x)
		}
		ratio := 0.0
		if alpha > 0 {
			ratio = (float64(k) - alpha) / alpha
		}
		window := besselI0(beta*math.Sqrt(math.Max(0, 1-ratio*ratio))) / i0Beta
		h[k] = sinc * window
		sum += h[k]
	}
	scale := float64(f.up) / sum
	for k := range taps {
		h[k] *= scale
	}
	return h
}

// newSincPhase stores one phase's taps, folded when they are symmetric.
func newSincPhase(taps []float32, width int) sincPhase {
	var peak float32
	for _, t := range taps[:width] {
		peak = max(peak, float32(math.Abs(float64(t))))
	}
	symmetric := width >= 16
	for j := 0; j < width/2 && symmetric; j++ {
		if float32(math.Abs(float64(taps[j]-taps[width-1-j]))) > 1e-6*peak {
			symmetric = false
		}
	}
	if symmetric {
		half := (width + 1) / 2
		ph := sincPhase{fold: true, len: width, hp: round8(half)}
		ph.taps = make([]float32, ph.hp)
		copy(ph.taps, taps[:half])
		// The middle tap of an odd filter pairs with itself.
		if width%2 == 1 {
			ph.taps[half-1] *= 0.5
		}
		return ph
	}
	ph := sincPhase{len: round8(width)}
	ph.taps = make([]float32, ph.len)
	copy(ph.taps, taps[:width])
	return ph
}

// delay is how many input frames it takes for the last input to reach the
// output.
func (f *sincFilter) delay() int {
	if f.passthrough {
		return 0
	}
	return int((f.center + int64(f.up) - 1) / int64(f.up))
}

// resampleCount is how many output frames a clip of inFrames comes out as: its
// length at the output rate, rounded to the nearest frame.
func (f *sincFilter) resampleCount(inFrames int) int {
	if f.passthrough {
		return inFrames
	}
	return int((int64(inFrames)*2*int64(f.up) + int64(f.down)) / (2 * int64(f.down)))
}

// resample converts a whole clip of interleaved S16LE frames. Output frame n is
// the clip at input frame n * inRate / outRate.
func (f *sincFilter) resample(pcm []byte, channels int) []byte {
	inFrames := len(pcm) / (2 * channels)
	if f.passthrough {
		return pcm[:inFrames*2*channels]
	}
	count := f.resampleCount(inFrames)
	if count == 0 {
		return nil
	}
	// Silence before the clip fills the first window, and silence after it the
	// windows that reach past its end. The padding lets the last dot product
	// read whole blocks of taps.
	hist := f.width - 1
	reach := int((f.center+int64(count-1)*int64(f.down))/int64(f.up)) + f.width
	n := max(reach, hist+inFrames) + 8
	buf := make([]float32, n)
	out := make([]int16, count*channels)
	for c := range channels {
		clear(buf)
		loadChannel(buf[hist:], pcm, inFrames, channels, c)
		f.filterChannel(buf, f.center, count, channels, out[c:])
	}
	return int16sToBytes(out)
}

// filterChannel filters one channel's buffer into every stride-th sample of out.
// Output n sits at n * down on the input upsampled by up; offset is where the
// first output sits relative to the first input there.
func (f *sincFilter) filterChannel(b []float32, offset int64, count, stride int, out []int16) {
	start := offset / int64(f.up)
	phase := int(offset % int64(f.up))
	for n := range count {
		ph := &f.phases[phase]
		w := b[start:]
		var v float32
		if ph.fold {
			v = foldGeneric(w, ph.len, ph.taps, ph.hp)
		} else {
			v = dotGeneric(w, ph.taps, ph.len)
		}
		out[n*stride] = toInt16(v)
		start += f.stepWhole
		phase += f.stepFrac
		if phase >= f.up {
			phase -= f.up
			start++
		}
	}
}

// dotGeneric is the dot product of a window with a phase's taps. n is a
// multiple of 8.
func dotGeneric(a, t []float32, n int) float32 {
	var s0, s1, s2, s3 float32
	for j := 0; j < n; j += 4 {
		s0 += float32(a[j] * t[j])
		s1 += float32(a[j+1] * t[j+1])
		s2 += float32(a[j+2] * t[j+2])
		s3 += float32(a[j+3] * t[j+3])
	}
	return (s0 + s1) + (s2 + s3)
}

// foldGeneric is the dot product of a window with a symmetric phase, computed
// on mirrored pairs of samples. hp is a multiple of 8.
func foldGeneric(w []float32, span int, t []float32, hp int) float32 {
	var s0, s1 float32
	for j := 0; j < hp; j += 2 {
		s0 += float32(t[j] * (w[j] + w[span-1-j]))
		s1 += float32(t[j+1] * (w[j+1] + w[span-2-j]))
	}
	return s0 + s1
}

// sincStream is a streaming conversion between one pair of rates. It is not
// safe for concurrent use.
type sincStream struct {
	f        *sincFilter
	channels int
	// bufs is one buffer per channel: width - 1 samples of history, the input,
	// then padding.
	bufs [][]float32
	// pending reports that audio has gone in since the last reset.
	pending bool
	offset  int64
}

func newSincStream(f *sincFilter, channels int) *sincStream {
	return &sincStream{f: f, channels: channels, bufs: make([][]float32, channels)}
}

// outputCount is how many output frames the next call with inFrames input
// frames returns.
func (r *sincStream) outputCount(inFrames int) int {
	f := r.f
	if f.passthrough {
		return inFrames
	}
	total := int64(inFrames)*int64(f.up) - r.offset
	if total <= 0 {
		return 0
	}
	return int((total + int64(f.down) - 1) / int64(f.down))
}

// process converts a chunk of interleaved S16LE frames, returning exactly the
// audio it amounts to at the output rate.
func (r *sincStream) process(pcm []byte) []byte {
	f := r.f
	inFrames := len(pcm) / (2 * r.channels)
	if f.passthrough {
		return pcm[:inFrames*2*r.channels]
	}
	if inFrames == 0 {
		return nil
	}
	hist := f.width - 1
	const tail = 8
	need := hist + inFrames + tail
	for c := range r.channels {
		b := r.bufs[c]
		if len(b) < need {
			grown := make([]float32, need)
			copy(grown, b[:min(len(b), hist)])
			b = grown
		}
		b = b[:need]
		loadChannel(b[hist:], pcm, inFrames, r.channels, c)
		clear(b[hist+inFrames:])
		r.bufs[c] = b
	}

	count := r.outputCount(inFrames)
	out := make([]int16, count*r.channels)
	for c := range r.channels {
		b := r.bufs[c]
		f.filterChannel(b, r.offset, count, r.channels, out[c:])
		copy(b, b[inFrames:inFrames+hist])
	}
	r.offset += int64(count)*int64(f.down) - int64(inFrames)*int64(f.up)
	r.pending = true
	return int16sToBytes(out)
}

// flush returns the audio the filter is delaying and resets for the next
// stream. It is empty when no audio has gone in since the last reset.
func (r *sincStream) flush() []byte {
	delay := 0
	if r.pending {
		delay = r.f.delay()
	}
	out := r.process(make([]byte, delay*2*r.channels))
	r.reset()
	return out
}

// reset discards the audio held in the filter.
func (r *sincStream) reset() {
	hist := r.f.width - 1
	for _, b := range r.bufs {
		if len(b) >= hist && hist > 0 {
			clear(b[:hist])
		}
	}
	r.offset = 0
	r.pending = false
}

// loadChannel copies one channel of interleaved S16LE frames into b as floats.
func loadChannel(b []float32, pcm []byte, frames, channels, c int) {
	for i := range frames {
		b[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*(i*channels+c):])))
	}
}

// toInt16 rounds to the nearest sample, ties to even, and clips.
func toInt16(v float32) int16 {
	r := math.RoundToEven(float64(v))
	return int16(max(math.MinInt16, min(math.MaxInt16, r)))
}

// int16sToBytes encodes samples as S16LE.
func int16sToBytes(s []int16) []byte {
	out := make([]byte, 2*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(v))
	}
	return out
}

// besselI0 is the modified Bessel function of the first kind, order zero.
func besselI0(x float64) float64 {
	sum, term, q := 1.0, 1.0, x*x/4
	for k := 1; k < 500; k++ {
		term *= q / (float64(k) * float64(k))
		sum += term
		if term < sum*1e-17 {
			break
		}
	}
	return sum
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func round8(n int) int { return (n + 7) &^ 7 }
