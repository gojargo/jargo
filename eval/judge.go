package eval

import "context"

// Judge decides whether a bot reply satisfies a natural-language criterion — it
// backs a scenario's `judge:` assertion. Implementations typically run a
// one-shot LLM inference; the harness treats the judge as optional, so a
// scenario without any `judge:` assertions needs none.
type Judge interface {
	// Evaluate reports whether reply meets criterion, with a short reason to
	// surface when it does not.
	Evaluate(ctx context.Context, criterion, reply string) (pass bool, reason string, err error)
}
