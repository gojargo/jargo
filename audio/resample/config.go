package resample

import "time"

// Quality selects the conversion filter, trading audio quality against CPU cost.
// The zero value is the highest quality, which is what a pipeline gets unless it
// asks for something cheaper.
//
// The names are the five standard SoX Resampler recipes. Both builds understand
// all five: the libsoxr build passes them straight through, and the pure-Go
// build maps them onto the converters its library offers (see converterFor).
type Quality int

const (
	// QualityVHQ is very high quality. It is the zero value, and so the default.
	QualityVHQ Quality = iota
	// QualityHQ is high quality: a shorter filter than VHQ, and audibly
	// transparent for speech.
	QualityHQ
	// QualityMQ is medium quality.
	QualityMQ
	// QualityLQ is low quality.
	QualityLQ
	// QualityQQ is "quick": interpolation rather than a windowed filter, for
	// paths that care more about cost than about the stopband.
	QualityQQ
)

// String returns the recipe name.
func (q Quality) String() string {
	switch q {
	case QualityVHQ:
		return "VHQ"
	case QualityHQ:
		return "HQ"
	case QualityMQ:
		return "MQ"
	case QualityLQ:
		return "LQ"
	case QualityQQ:
		return "QQ"
	default:
		return "unknown"
	}
}

// DefaultClearAfter is how long a Resampler may sit idle before the next chunk
// is treated as the start of a fresh signal rather than the continuation of the
// last one.
const DefaultClearAfter = 200 * time.Millisecond

// Config configures a Resampler.
type Config struct {
	// Quality selects the conversion filter; the zero value is QualityVHQ.
	Quality Quality
	// ClearAfter is how long the resampler may sit idle before its filter
	// history is discarded. A resampler carries the tail of the audio it last
	// saw so that a continuous stream converts cleanly across chunk boundaries,
	// but after a gap that tail is no longer what came before: it is the end of
	// the previous utterance bleeding into the start of the next one, which is
	// heard as a click. 0 uses DefaultClearAfter; a negative value never clears,
	// which is what a telephony leg wants, since its chunks arrive at irregular
	// intervals that are gaps in delivery rather than gaps in the audio.
	ClearAfter time.Duration
}

// clearAfter returns the idle window the config asks for, resolving the zero
// value to the default and reporting whether clearing is wanted at all.
func (c Config) clearAfter() (time.Duration, bool) {
	switch {
	case c.ClearAfter < 0:
		return 0, false
	case c.ClearAfter == 0:
		return DefaultClearAfter, true
	default:
		return c.ClearAfter, true
	}
}

// idleClock decides when a stream has been idle long enough that the filter
// history should be dropped. It measures the gap between successive calls, not
// the age of the audio, because that is the only thing a resampler can see.
type idleClock struct {
	after   time.Duration
	enabled bool
	last    time.Time
	// now is the clock, replaced in tests. nil means time.Now.
	now func() time.Time
}

func newIdleClock(cfg Config) idleClock {
	after, enabled := cfg.clearAfter()
	return idleClock{after: after, enabled: enabled}
}

func (c *idleClock) time() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}

// stale reports whether the gap since the previous call exceeds the idle window,
// and records this call as the most recent one. The first call is never stale:
// there is no history to drop yet.
func (c *idleClock) stale() bool {
	now := c.time()
	idle := c.enabled && !c.last.IsZero() && now.Sub(c.last) > c.after
	c.last = now
	return idle
}

// reset forgets the last call, so the next one starts a fresh idle window.
func (c *idleClock) reset() { c.last = time.Time{} }
