package clock_test

import (
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
)

// TestTimeBeforeStart pins the documented contract: an unstarted clock reports
// zero rather than the time since the epoch, so a frame timestamped before the
// pipeline starts is recognizably unset instead of absurdly large.
func TestTimeBeforeStart(t *testing.T) {
	if got := clock.NewSystem().Time(); got != 0 {
		t.Errorf("Time() = %v, want 0 before Start", got)
	}
}

func TestTimeAdvancesAfterStart(t *testing.T) {
	c := clock.NewSystem()
	c.Start()

	time.Sleep(10 * time.Millisecond)
	first := c.Time()
	if first <= 0 {
		t.Fatalf("Time() = %v, want a positive elapsed time", first)
	}

	time.Sleep(10 * time.Millisecond)
	if second := c.Time(); second <= first {
		t.Errorf("Time() = %v, want it to advance past %v", second, first)
	}
}

// TestRestartResetsReference checks a second Start moves the reference point, so
// a reused clock measures the new session rather than accumulating.
func TestRestartResetsReference(t *testing.T) {
	c := clock.NewSystem()
	c.Start()
	time.Sleep(20 * time.Millisecond)

	before := c.Time()
	c.Start()
	if after := c.Time(); after >= before {
		t.Errorf("Time() = %v after restart, want less than %v", after, before)
	}
}

// TestConcurrentUse checks the clock is safe to read while the pipeline starts
// it, which is how it is used: one starter, many readers.
func TestConcurrentUse(t *testing.T) {
	c := clock.NewSystem()
	var wg sync.WaitGroup

	wg.Go(c.Start)
	for range 8 {
		wg.Go(func() {
			for range 50 {
				_ = c.Time()
			}
		})
	}
	wg.Wait()
}

// System must satisfy the Clock interface a pipeline depends on.
var _ clock.Clock = (*clock.System)(nil)
