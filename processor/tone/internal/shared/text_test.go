package shared_test

import (
	"slices"
	"testing"

	"github.com/gojargo/jargo/processor/tone/internal/shared"
)

func TestStringRulesFollowPython(t *testing.T) {
	// Python: "\x1c a 　b\x1f".strip() == "a 　b"; .split() == ["a", "b"].
	if got := shared.Strip("\x1c a 　b\x1f"); got != "a 　b" {
		t.Errorf("Strip = %q", got)
	}
	if got := shared.Fields("\x1c a 　b\x1f"); !slices.Equal(got, []string{"a", "b"}) {
		t.Errorf("Fields = %q", got)
	}
	// Python: "²".isalnum(), "٣".isalnum() and "一".isalnum() are True, "_" is not.
	for r, want := range map[rune]bool{'²': true, '٣': true, '一': true, '_': false, 'é': true} {
		if shared.IsAlnum(r) != want {
			t.Errorf("IsAlnum(%q) = %v, want %v", r, !want, want)
		}
	}
}

func TestRoundFollowsPython(t *testing.T) {
	// Python: round(0.12345, 4) == 0.1235 (0.12345 is just above the tie),
	// round(2.675, 2) == 2.67 (just below), round(0.125, 2) == 0.12 (an exact
	// tie, to even), round(0.71, 2) == 0.71.
	for _, c := range []struct {
		x      float64
		digits int
		want   float64
	}{
		{0.12345, 4, 0.1235},
		{2.675, 2, 2.67},
		{0.125, 2, 0.12},
		{0.71, 2, 0.71},
		{0.99999, 4, 1},
	} {
		if got := shared.Round(c.x, c.digits); got != c.want {
			t.Errorf("Round(%v, %d) = %v, want %v", c.x, c.digits, got, c.want)
		}
	}
}
