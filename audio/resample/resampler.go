package resample

import (
	"fmt"
	"sync"
)

// filters holds the filter of every pair of rates in use, shared by every
// stream between them: a filter is fixed once made, and designing one is the
// costly part of opening a stream.
//
//nolint:gochecknoglobals // a process-wide cache of immutable filters
var filters sync.Map

// filterFor returns the shared filter between two rates.
func filterFor(inRate, outRate int) (*sincFilter, error) {
	key := [2]int{inRate, outRate}
	if f, ok := filters.Load(key); ok {
		return f.(*sincFilter), nil //nolint:forcetypeassert // only filters are stored
	}
	f, err := newSincFilter(inRate, outRate)
	if err != nil {
		return nil, err
	}
	actual, _ := filters.LoadOrStore(key, f)
	return actual.(*sincFilter), nil //nolint:forcetypeassert // only filters are stored
}

// Resampler converts a stream of interleaved S16LE PCM from one sample rate to
// another with a polyphase windowed-sinc filter that keeps the lower rate's band
// flat to 90% of its Nyquist frequency and attenuates aliasing and images by
// about 90 dB.
//
// Its output keeps pace with its input: every call returns exactly the audio its
// input amounts to at the output rate, so equal input chunks come out as equal
// output chunks, the first one included. The filter delays the audio by a few
// milliseconds instead of holding output back; Flush emits those milliseconds
// where a stream ends, and Clear drops them where a stream is abandoned.
//
// Create one per audio stream with New; it is not safe for concurrent use.
type Resampler struct {
	inRate   int
	outRate  int
	channels int
	s        *sincStream // nil when inRate == outRate (passthrough)

	idle idleClock
}

// New returns a Resampler from inRate to outRate for the given channel count.
// When inRate equals outRate the Resampler passes audio through unchanged.
func New(inRate, outRate, channels int) (*Resampler, error) {
	return NewWithConfig(inRate, outRate, channels, Config{})
}

// NewWithConfig returns a Resampler configured by cfg. See New.
func NewWithConfig(inRate, outRate, channels int, cfg Config) (*Resampler, error) {
	if channels < 1 {
		channels = 1
	}
	r := &Resampler{inRate: inRate, outRate: outRate, channels: channels, idle: newIdleClock(cfg)}
	if inRate == outRate {
		return r, nil
	}
	f, err := filterFor(inRate, outRate)
	if err != nil {
		return nil, fmt.Errorf("resample: new %d->%d ch=%d: %w", inRate, outRate, channels, err)
	}
	r.s = newSincStream(f, channels)
	return r, nil
}

// Delay is how many input frames it takes for the last input to reach the
// output: the audio the filter is holding, which Flush emits.
func (r *Resampler) Delay() int {
	if r.s == nil {
		return 0
	}
	return r.s.f.delay()
}

// Clear discards the audio held in the filter, so the next chunk is converted as
// the start of a fresh signal rather than the continuation of the last one.
//
// It is the counterpart of Flush: use Clear for a stream that was abandoned, so
// its leftover tail does not bleed into the next one, and Flush for one that
// ended, so the tail is heard before the next one starts.
func (r *Resampler) Clear() {
	if r.s != nil {
		r.s.reset()
	}
	r.idle.reset()
}

// Flush emits the audio the filter is still delaying and resets for the next
// stream. Call it when a stream has ended; the returned audio belongs at the end
// of that stream. It returns nothing when no audio has gone in since the
// resampler was created, flushed or cleared, and when the rates match.
func (r *Resampler) Flush() []byte {
	if r.s == nil {
		return nil
	}
	r.idle.reset()
	return r.s.flush()
}

// Process resamples one buffer of interleaved S16LE PCM and returns exactly the
// audio it amounts to at the output rate, carrying any fraction of a frame into
// the next call. When the input and output rates match it returns the input
// unchanged; otherwise the returned slice is freshly allocated and owned by the
// caller. Use Resample instead for a buffer that is complete on its own.
//
// A gap longer than a configured idle window clears the audio held in the
// filter first, so the tail of what came before is not mixed into what comes
// next.
func (r *Resampler) Process(in []byte) []byte {
	if r.s == nil {
		return in
	}
	if r.idle.stale() {
		r.s.reset()
	}
	if len(in)/(2*r.channels) == 0 {
		return nil
	}
	return r.s.process(in)
}

// resampleAll converts a complete buffer in one pass. It backs Resample; see
// there.
func resampleAll(pcm []byte, inRate, outRate, channels int) ([]byte, error) {
	if len(pcm)/(2*channels) == 0 {
		return nil, nil
	}
	f, err := filterFor(inRate, outRate)
	if err != nil {
		return nil, fmt.Errorf("resample: %d->%d ch=%d: %w", inRate, outRate, channels, err)
	}
	return f.resample(pcm, channels), nil
}

// Close releases the stream. The filter it used stays shared with the other
// streams between the same rates. It is safe to call more than once.
func (r *Resampler) Close() {
	r.s = nil
}
