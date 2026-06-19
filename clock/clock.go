// Package clock provides the timing source a pipeline uses for presentation
// timestamps and elapsed-time measurements.
package clock

import (
	"sync"
	"time"
)

// Clock is a timing source for a pipeline. Start fixes the reference point and
// Time reports the elapsed time since then. The unit and reference are defined
// by the implementation.
type Clock interface {
	// Start fixes the clock's reference point. It must be called before Time.
	Start()
	// Time is the elapsed time since Start. It is 0 before Start is called.
	Time() time.Duration
}

// System is a monotonic Clock backed by the host's monotonic time, so it is
// unaffected by wall-clock adjustments. The zero value is not usable; build one
// with NewSystem. It is safe for concurrent use.
type System struct {
	mu      sync.RWMutex
	start   time.Time
	started bool
}

// NewSystem returns a System clock that has not been started yet.
func NewSystem() *System { return &System{} }

// Start records the current monotonic time as the reference for Time.
func (c *System) Start() {
	c.mu.Lock()
	c.start = time.Now()
	c.started = true
	c.mu.Unlock()
}

// Time returns the elapsed time since Start, or 0 if the clock has not been
// started yet.
func (c *System) Time() time.Duration {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if !c.started {
		return 0
	}
	return time.Since(c.start)
}
