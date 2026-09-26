package sentencex

import "testing"

// These tests cover detector policy that the end-to-end corpus in
// testdata/lists.txt cannot tell apart: cases where the detector must return
// nothing (so segmentation falls back to the terminators) and cases that test
// which family wins. The corpus sees the segments after the whole pipeline,
// which can match the terminator-driven output even when the detector wrongly
// fires, so these assert on the detector itself.

func TestDetectListItemsSilent(t *testing.T) {
	tests := []struct {
		name string
		text string
	}{
		// A bare-dot closer inline is ambiguous with a wrapped prose sentence
		// ending in "...Foo 1.".
		{"inline numeric dot", "1. The first item. 2. The second item."},
		// A bare-dot closer and a lowercase letter collide with initials
		// (e. e. cummings) and date or abbreviation shapes.
		{"inline letter dot", "a. The first item b. The second item"},
		{"inline roman dot", "ii. The first item iii. The second item"},
		{"inline year parens padded", "Examples include 'Sonar Tari' ( 1894 ), 'Chitra' ( 1896 ), " +
			"and 'Katha O Kahini' ( 1900 )."},
		{"inline year parens", "Works (1894), (1896), and (1900) are notable."},
		// e. e. has no content between the markers, so the gap pruning drops
		// the second, leaving fewer than two siblings.
		{"ee cummings", "From\ne. e. cummings, with love."},
		{"uppercase letter dot", "Reviewed by\nA. Smith and\nB. Jones."},
		{"lone lowercase letter dot", "The answer is a.\nFollow up later."},
		{"lone numeric after wrap", "The total was\n1. Two hundred dollars exactly."},
		{"e.g. at line start", "e.g. one\ne.g. two\n"},
		{"marker-only lines", "*\n*\n*\n"},
		{"plain prose", "Hello world. This is a test. Three sentences here."},
		{"empty paragraph", ""},
	}
	for _, tt := range tests {
		if starts := detectListItems(tt.text); len(starts) != 0 {
			t.Errorf("%s: got %v, want none", tt.name, starts)
		}
	}
}

func TestDetectListItemsUnicodeBulletWithDecoration(t *testing.T) {
	// Bullets must win over the numeric decoration.
	s := "• 9. The first item • 10. The second item"
	starts := detectListItems(s)
	if len(starts) != 2 {
		t.Fatalf("got %v, want two starts", starts)
	}
	if starts[0] != 0 {
		t.Errorf("first start = %d, want 0", starts[0])
	}
	if got := s[starts[1] : starts[1]+3]; got != "•" {
		t.Errorf("second start opens %q, want •", got)
	}
}
