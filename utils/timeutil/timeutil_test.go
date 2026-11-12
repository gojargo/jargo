package timeutil_test

import (
	"testing"

	"github.com/gojargo/jargo/utils/timeutil"
)

// test_whole_and_zero: whole seconds convert exactly.
func TestSecondsToNanosecondsWholeAndZero(t *testing.T) {
	if got := timeutil.SecondsToNanoseconds(0); got != 0 {
		t.Errorf("0 s = %d ns", got)
	}
	if got := timeutil.SecondsToNanoseconds(2); got != 2_000_000_000 {
		t.Errorf("2 s = %d ns", got)
	}
}

// test_rounds_float_error: float products such as 1.001 * 1e9 land just below
// the exact value.
func TestSecondsToNanosecondsRoundsFloatError(t *testing.T) {
	for secs, want := range map[float64]int64{1.001: 1_001_000_000, 0.57: 570_000_000, 4.35: 4_350_000_000} {
		if got := timeutil.SecondsToNanoseconds(secs); got != want {
			t.Errorf("%v s = %d ns, want %d", secs, got, want)
		}
	}
}

// test_every_millisecond_is_exact: no millisecond lands a nanosecond off.
func TestSecondsToNanosecondsEveryMillisecondIsExact(t *testing.T) {
	for ms := range int64(10_000) {
		if got := timeutil.SecondsToNanoseconds(float64(ms) / 1000); got != ms*1_000_000 {
			t.Fatalf("%d ms = %d ns, want %d", ms, got, ms*1_000_000)
		}
	}
}
