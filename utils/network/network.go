// Package network holds the retry helpers shared by services that keep a
// connection open to a provider for the length of a call. Such a connection
// drops for ordinary reasons (a network blip, a server recycling), so a service
// reconnects rather than failing the call. These helpers decide how long to wait
// between attempts, and when waiting longer has stopped being the answer.
package network

import (
	"math"
	"time"
)

const (
	// DefaultMinWait is the shortest wait between two retry attempts.
	DefaultMinWait = 4 * time.Second
	// DefaultMaxWait is the longest wait between two retry attempts.
	DefaultMaxWait = 10 * time.Second
	// DefaultMultiplier scales the doubling. One means the raw powers of two.
	DefaultMultiplier = 1.0
)

// ExponentialBackoffTime is how long to wait before a retry, doubling with each
// attempt. attempt counts from one, and the wait is two to the power of
// attempt-1 seconds scaled by multiplier, then clamped to minWait..maxWait. A
// multiplier that is not a number yields maxWait, the most cautious answer
// available.
func ExponentialBackoffTime(attempt int, minWait, maxWait time.Duration, multiplier float64) time.Duration {
	seconds := math.Pow(2, float64(attempt-1)) * multiplier
	if math.IsNaN(seconds) {
		return maxWait
	}
	seconds = math.Max(0, math.Min(seconds, maxWait.Seconds()))
	return max(minWait, time.Duration(seconds*float64(time.Second)))
}

const (
	// DefaultMinStableDuration is how long a connection must survive before
	// QuickFailureTracker counts it as stable.
	DefaultMinStableDuration = 5 * time.Second
	// DefaultMaxConsecutiveFailures is how many quick failures in a row
	// QuickFailureTracker tolerates before telling the caller to give up.
	DefaultMaxConsecutiveFailures = 3
)

// QuickFailureConfig configures a QuickFailureTracker. A zero field takes its
// default.
type QuickFailureConfig struct {
	// MinStableDuration is how long a connection must survive to count as
	// stable rather than as a quick failure.
	MinStableDuration time.Duration
	// MaxConsecutiveFailures is how many quick failures in a row end the
	// retrying.
	MaxConsecutiveFailures int
}

// QuickFailureResult is what recording one failed connection revealed.
type QuickFailureResult struct {
	// QuickFailure reports whether the connection lasted less than the
	// tracker's MinStableDuration.
	QuickFailure bool
	// GiveUp reports whether MaxConsecutiveFailures quick failures have now
	// happened in a row, so the caller should stop retrying rather than wait
	// longer and try again.
	GiveUp bool
}

// QuickFailureTracker spots a connection that keeps failing the moment it is
// established. Backing off cannot help there: every attempt completes the
// handshake and then fails straight away, which is what a server does when it
// rejects the credentials after the upgrade rather than before it. Waiting
// longer between attempts changes nothing, so the tracker is what says to stop.
//
// Report how long each failed connection lasted with Record. Once
// MaxConsecutiveFailures of them in a row each lasted less than
// MinStableDuration, the result says to give up.
//
// A tracker is not safe for concurrent use. Call it from the goroutine that owns
// the connection.
type QuickFailureTracker struct {
	minStable   time.Duration
	maxFailures int
	count       int
}

// NewQuickFailureTracker builds a tracker from cfg.
func NewQuickFailureTracker(cfg QuickFailureConfig) *QuickFailureTracker {
	t := &QuickFailureTracker{
		minStable:   cfg.MinStableDuration,
		maxFailures: cfg.MaxConsecutiveFailures,
	}
	if t.minStable <= 0 {
		t.minStable = DefaultMinStableDuration
	}
	if t.maxFailures <= 0 {
		t.maxFailures = DefaultMaxConsecutiveFailures
	}
	return t
}

// Record notes a failed connection that lasted d, lengthening the streak when it
// was a quick failure and ending the streak when it was not.
func (t *QuickFailureTracker) Record(d time.Duration) QuickFailureResult {
	quick := d < t.minStable
	if quick {
		t.count++
	} else {
		t.count = 0
	}
	return QuickFailureResult{
		QuickFailure: quick,
		GiveUp:       quick && t.count >= t.maxFailures,
	}
}

// Reset ends the streak, for a connection that is starting fresh.
func (t *QuickFailureTracker) Reset() { t.count = 0 }

// Count is how many quick failures have happened in a row.
func (t *QuickFailureTracker) Count() int { return t.count }

// MaxConsecutiveFailures is how many quick failures in a row end the retrying.
func (t *QuickFailureTracker) MaxConsecutiveFailures() int { return t.maxFailures }
