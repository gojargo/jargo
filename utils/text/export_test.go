package text

import "testing"

// StubMatchEndOfSentence has the aggregators find their boundaries with fn for
// the rest of the test.
func StubMatchEndOfSentence(tb testing.TB, fn func(text, language string) int) {
	tb.Helper()
	prev := matchEndOfSentence
	matchEndOfSentence = fn
	tb.Cleanup(func() { matchEndOfSentence = prev })
}
