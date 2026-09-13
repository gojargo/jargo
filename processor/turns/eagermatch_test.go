package turns_test

import (
	"testing"

	"github.com/gojargo/jargo/processor/turns"
)

// TestExactMatch covers the strict policy: only two identical transcripts keep
// the speculative reply.
func TestExactMatch(t *testing.T) {
	var p turns.ExactMatch

	if !p.Matches("book a flight", "book a flight") {
		t.Error("identical transcripts did not match")
	}
	if p.Matches("book a flight", "Book a flight.") {
		t.Error("a formatted transcript matched, want the strict policy to refuse it")
	}
	if p.Matches("book a flight", "book a flight tomorrow") {
		t.Error("a different transcript matched")
	}
}

// TestNormalizedMatch covers the default policy: a service that capitalizes and
// punctuates the committed transcript has not said anything different, but one
// that heard another word has.
func TestNormalizedMatch(t *testing.T) {
	var p turns.NormalizedMatch

	for _, c := range []struct{ eager, final string }{
		{"book a flight", "Book a flight."},
		{"book  a flight", "book a flight"},
		{"its ready", "It's ready!"},
	} {
		if !p.Matches(c.eager, c.final) {
			t.Errorf("%q did not match %q, want formatting ignored", c.eager, c.final)
		}
	}

	for _, c := range []struct{ eager, final string }{
		{"book a flight", "book a flight tomorrow"},
		{"i want to cancel", "I want to reschedule."},
	} {
		if p.Matches(c.eager, c.final) {
			t.Errorf("%q matched %q, want the difference caught", c.eager, c.final)
		}
	}
}
