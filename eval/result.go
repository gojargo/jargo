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

// Result is the outcome of running one scenario.
type Result struct {
	// Scenario is the scenario's name.
	Scenario string
	// Failures lists every unmet expectation; empty means the scenario passed.
	Failures []Failure
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
	// Skipped, when set, says why the scenario was not run. Such a result neither
	// passed nor failed.
	Skipped string
}

// Passed reports whether the scenario ran and every expectation was met.
func (r Result) Passed() bool { return r.Skipped == "" && len(r.Failures) == 0 }

// String renders a human-readable summary of the run.
func (r Result) String() string {
	if r.Skipped != "" {
		return fmt.Sprintf("SKIP %s (%s)", r.Scenario, r.Skipped)
	}
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
