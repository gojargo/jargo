package resample

import (
	"testing"
	"time"
)

func TestClearAfterResolvesTheZeroValue(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		want    time.Duration
		enabled bool
	}{
		{"zero never clears", Config{}, 0, false},
		{"explicit window", Config{ClearAfter: time.Second}, time.Second, true},
		{"negative never clears", Config{ClearAfter: -1}, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, enabled := tt.cfg.clearAfter()
			if got != tt.want || enabled != tt.enabled {
				t.Errorf("clearAfter() = (%v, %v), want (%v, %v)", got, enabled, tt.want, tt.enabled)
			}
		})
	}
}

// TestIdleClockGoesStaleAfterAGap drives the clock by hand: the audio carries no
// timestamps, so the only thing a resampler can measure is the gap between the
// calls it receives.
func TestIdleClockGoesStaleAfterAGap(t *testing.T) {
	now := time.Unix(0, 0)
	c := newIdleClock(Config{ClearAfter: 200 * time.Millisecond})
	c.now = func() time.Time { return now }

	// The first call has no history behind it, so there is nothing to drop.
	if c.stale() {
		t.Error("the first call should never be stale")
	}

	// A chunk arriving inside the window continues the same signal.
	now = now.Add(20 * time.Millisecond)
	if c.stale() {
		t.Error("a chunk inside the idle window should not be stale")
	}

	// A gap past the window means the next chunk starts a fresh signal.
	now = now.Add(201 * time.Millisecond)
	if !c.stale() {
		t.Error("a gap longer than the idle window should be stale")
	}

	// Having cleared, the stream continues from there.
	now = now.Add(20 * time.Millisecond)
	if c.stale() {
		t.Error("the chunk after a clear should not be stale again")
	}
}

func TestIdleClockNeverStaleWhenDisabled(t *testing.T) {
	// The default config keeps the history across a pause, as an explicit
	// never does: a pause is not the end of the stream, and dropping the held
	// audio would cut a gap into it.
	for _, cfg := range []Config{{}, {ClearAfter: -1}} {
		now := time.Unix(0, 0)
		c := newIdleClock(cfg)
		c.now = func() time.Time { return now }

		c.stale()
		now = now.Add(time.Hour)
		if c.stale() {
			t.Errorf("ClearAfter %v: a gap was stale, want the history kept", cfg.ClearAfter)
		}
	}
}

func TestIdleClockResetStartsAFreshWindow(t *testing.T) {
	now := time.Unix(0, 0)
	c := newIdleClock(Config{})
	c.now = func() time.Time { return now }

	c.stale()
	c.reset()
	now = now.Add(time.Hour)
	if c.stale() {
		t.Error("after reset the next call is the first one again, so it cannot be stale")
	}
}
