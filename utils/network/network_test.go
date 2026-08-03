package network_test

import (
	"math"
	"testing"
	"time"

	"github.com/gojargo/jargo/utils/network"
)

func TestExponentialBackoffTime(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		minWait    time.Duration
		maxWait    time.Duration
		multiplier float64
		// wantSeconds is the wait for attempts one to six, in seconds.
		wantSeconds []int
	}{
		{
			name:    "the floor holds the first attempts",
			minWait: 4 * time.Second, maxWait: 10 * time.Second, multiplier: 1,
			wantSeconds: []int{4, 4, 4, 8, 10, 10},
		},
		{
			name:    "doubling from a low floor",
			minWait: time.Second, maxWait: 10 * time.Second, multiplier: 1,
			wantSeconds: []int{1, 2, 4, 8, 10, 10},
		},
		{
			name:    "the multiplier scales the doubling",
			minWait: time.Second, maxWait: 20 * time.Second, multiplier: 2,
			wantSeconds: []int{2, 4, 8, 16, 20, 20},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			for i, seconds := range tc.wantSeconds {
				attempt := i + 1
				want := time.Duration(seconds) * time.Second
				got := network.ExponentialBackoffTime(attempt, tc.minWait, tc.maxWait, tc.multiplier)
				if got != want {
					t.Errorf("attempt %d: got %v, want %v", attempt, got, want)
				}
			}
		})
	}
}

// TestExponentialBackoffTimeSaturates guards the far end of the curve, where an
// attempt number large enough to overflow the doubling must still clamp to
// maxWait rather than wrap into a nonsense wait.
func TestExponentialBackoffTimeSaturates(t *testing.T) {
	t.Parallel()

	for _, attempt := range []int{64, 1024, math.MaxInt32} {
		got := network.ExponentialBackoffTime(attempt, 4*time.Second, 10*time.Second, 1)
		if got != 10*time.Second {
			t.Errorf("attempt %d: got %v, want %v", attempt, got, 10*time.Second)
		}
	}
}

func TestQuickFailureTrackerGivesUpAfterThreeQuickFailures(t *testing.T) {
	t.Parallel()

	tracker := network.NewQuickFailureTracker(network.QuickFailureConfig{})

	for attempt := 1; attempt <= 2; attempt++ {
		got := tracker.Record(time.Second)
		if !got.QuickFailure {
			t.Errorf("attempt %d: got a stable connection, want a quick failure", attempt)
		}
		if got.GiveUp {
			t.Errorf("attempt %d: gave up before the streak was long enough", attempt)
		}
	}

	got := tracker.Record(time.Second)
	if !got.GiveUp {
		t.Error("third quick failure did not end the retrying")
	}
	if tracker.Count() != 3 {
		t.Errorf("streak: got %d, want 3", tracker.Count())
	}
}

func TestQuickFailureTrackerStableConnectionEndsTheStreak(t *testing.T) {
	t.Parallel()

	tracker := network.NewQuickFailureTracker(network.QuickFailureConfig{})
	tracker.Record(time.Second)
	tracker.Record(time.Second)

	stable := tracker.Record(10 * time.Second)
	if stable.QuickFailure {
		t.Error("a connection that outlived the stable duration counted as a quick failure")
	}
	if tracker.Count() != 0 {
		t.Errorf("streak after a stable connection: got %d, want 0", tracker.Count())
	}

	// The streak restarts, so it takes three more quick failures to give up.
	for attempt := 1; attempt <= 2; attempt++ {
		if got := tracker.Record(time.Second); got.GiveUp {
			t.Errorf("attempt %d after the reset: gave up too early", attempt)
		}
	}
	if got := tracker.Record(time.Second); !got.GiveUp {
		t.Error("third quick failure after the reset did not end the retrying")
	}
}

func TestQuickFailureTrackerReset(t *testing.T) {
	t.Parallel()

	tracker := network.NewQuickFailureTracker(network.QuickFailureConfig{})
	tracker.Record(time.Second)
	tracker.Record(time.Second)
	tracker.Reset()

	if tracker.Count() != 0 {
		t.Errorf("streak after Reset: got %d, want 0", tracker.Count())
	}
	if got := tracker.Record(time.Second); got.GiveUp {
		t.Error("gave up on the first quick failure after a reset")
	}
}

func TestQuickFailureTrackerConfig(t *testing.T) {
	t.Parallel()

	tracker := network.NewQuickFailureTracker(network.QuickFailureConfig{
		MinStableDuration:      time.Second,
		MaxConsecutiveFailures: 2,
	})
	if tracker.MaxConsecutiveFailures() != 2 {
		t.Errorf("MaxConsecutiveFailures: got %d, want 2", tracker.MaxConsecutiveFailures())
	}
	// Two seconds is a quick failure under the default, stable under this one.
	if got := tracker.Record(2 * time.Second); got.QuickFailure {
		t.Error("a connection longer than the configured stable duration counted as a quick failure")
	}
	tracker.Record(time.Millisecond)
	if got := tracker.Record(time.Millisecond); !got.GiveUp {
		t.Error("the configured failure ceiling did not end the retrying")
	}
}
