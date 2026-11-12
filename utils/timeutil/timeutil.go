// Package timeutil converts between the units time is counted in across the
// framework.
package timeutil

import (
	"math"
	"time"
)

// SecondsToNanoseconds converts seconds to nanoseconds, rounded to the nearest
// one: the float product can land just below the exact count (1.001 s gives
// 1000999999.9999999), and truncating it would put the time a nanosecond early.
func SecondsToNanoseconds(seconds float64) int64 {
	return int64(math.RoundToEven(seconds * 1e9))
}

// SecondsToDuration is SecondsToNanoseconds as a time.Duration.
func SecondsToDuration(seconds float64) time.Duration {
	return time.Duration(SecondsToNanoseconds(seconds))
}
