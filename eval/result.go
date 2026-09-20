package eval

import (
	"fmt"
	"strings"
	"time"
)

// Failure is one unmet expectation in a scenario run.
type Failure struct {
	// Turn is the 1-based turn number.
	Turn int
	// Expectation is the 1-based expectation index within the turn.
	Expectation int
	// Event is the event name that was expected.
	Event string
	// Reason explains what went wrong.
	Reason string
}

// String renders a failure as a single line.
func (f Failure) String() string {
	return fmt.Sprintf("turn %d expectation %d (%s): %s", f.Turn, f.Expectation, f.Event, f.Reason)
}

// The statuses a turn resolves to.
const (
	// TurnPassed means every expectation of the turn was met.
	TurnPassed = "passed"
	// TurnFailed means at least one was not.
	TurnFailed = "failed"
)

// ExpectationResult is what one expectation of a turn resolved to.
type ExpectationResult struct {
	// Expectation is the 1-based index of the expectation within the turn.
	Expectation int
	// Event is the event name the expectation waited for.
	Event string
	// Passed reports whether it was satisfied.
	Passed bool
	// Matched is what it matched, in short, when it passed: a function call's
	// name and arguments, the text of a reply or a transcript. It is empty for
	// an event carrying no text, and on a failure, whose reason is in Failures.
	Matched string
}

// TurnResult is what one turn of a scenario resolved to.
//
// It is what a run is read back from once it is over: the failures alone say
// what went wrong, and these say what happened, a passing run included. A turn
// stops at an expectation nothing arrived for, so it lists its expectations up
// to that one and no further.
type TurnResult struct {
	// Turn is the 1-based turn number.
	Turn int
	// Status is TurnPassed or TurnFailed.
	Status string
	// Expectations is what each of the turn's expectations resolved to, in
	// order, up to the one that timed out.
	Expectations []ExpectationResult
	// Duration is how long the turn took.
	Duration time.Duration
}

// Result is the outcome of running one scenario.
type Result struct {
	// Scenario is the scenario's name.
	Scenario string
	// Failures lists every unmet expectation; empty means the scenario passed.
	Failures []Failure
	// Turns is what each turn resolved to, in order, and what each of its
	// expectations matched. A scenario stopped early lists the turns it reached.
	Turns []TurnResult
	// Duration is how long the run took.
	Duration time.Duration
	// Events is every event the bot emitted, in order, whether or not a scenario
	// asserted on it. It is what a failure is read against: what the bot actually
	// did, rather than only what it was expected to do.
	Events []Event
	// DebugLog is a timestamped trace of the harness's own decisions: events as
	// they arrived, what each expectation waited for, what the judge said. It is
	// what makes a run that failed once and passed the next time diagnosable.
	DebugLog []string
}

// Passed reports whether every expectation was met.
func (r Result) Passed() bool { return len(r.Failures) == 0 }

// String renders a human-readable summary of the run.
func (r Result) String() string {
	if r.Passed() {
		return "PASS " + r.Scenario
	}
	var b strings.Builder
	fmt.Fprintf(&b, "FAIL %s (%d failure(s))", r.Scenario, len(r.Failures))
	for _, f := range r.Failures {
		fmt.Fprintf(&b, "\n  - %s", f)
	}
	return b.String()
}
