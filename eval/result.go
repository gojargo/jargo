package eval

import (
	"fmt"
	"strings"
	"time"
)

// The kinds a failure can be, the stable key for grouping failures across runs.
// Each says how an assertion failed, so a repeated suite can report "10x timeout
// on turn 3" without parsing free-text reasons.
const (
	// FailureTimeout means no event of the expected type arrived within the
	// budget.
	FailureTimeout = "timeout"
	// FailureJudgeNo means the judge rejected the reply, or a function call's
	// criterion.
	FailureJudgeNo = "judge_no"
	// FailureJudgeContinue means the judge never accepted the reply before the
	// budget ran out.
	FailureJudgeContinue = "judge_continue"
	// FailureNoJudge means the scenario uses `eval:` but no judge could be
	// built.
	FailureNoJudge = "no_judge"
	// FailureNoContent means the matched event carried no text to judge.
	FailureNoContent = "no_content"
	// FailureTextMismatch means text_contains was not in the event's text.
	FailureTextMismatch = "text_mismatch"
	// FailureTextPresent means text_excludes was found in the event's text.
	FailureTextPresent = "text_present"
	// FailureTextUnmatched means a text_matches pattern did not match the
	// event's text.
	FailureTextUnmatched = "text_unmatched"
	// FailureMarkerMismatch means the marker meant something other than the
	// expectation asked for.
	FailureMarkerMismatch = "marker_mismatch"
	// FailureMarkerFormat means the response did not follow the marker protocol
	// the expectation checks.
	FailureMarkerFormat = "marker_format"
	// FailureMissingFunctionCall means an expected function call never arrived.
	FailureMissingFunctionCall = "missing_function_call"
	// FailureFunctionArgsMismatch means the call arrived with other arguments.
	FailureFunctionArgsMismatch = "function_args_mismatch"
	// FailureUnexpectedEvent means an absent expectation saw the event it
	// forbade.
	FailureUnexpectedEvent = "unexpected_event"
	// FailureSendAfterTimeout means a turn's send_after event never fired.
	FailureSendAfterTimeout = "send_after_timeout"
	// FailureConnectFailed means the harness never connected to the bot.
	FailureConnectFailed = "connect_failed"
	// FailureClosedBeforeReady means the bot closed the connection before
	// bot-ready.
	FailureClosedBeforeReady = "closed_before_ready"
	// FailureHandshakeTimeout means the bot connected but never sent bot-ready.
	FailureHandshakeTimeout = "handshake_timeout"
	// FailureError means the harness itself failed (sending to the bot, the
	// user TTS, the judge), not the bot.
	FailureError = "error"
)

// The event names a failure of the run itself carries, rather than one of an
// expectation.
const (
	// RunEventConnect is a run that never connected to the bot.
	RunEventConnect = "<connect>"
	// RunEventBotReady is a run whose bot never became ready.
	RunEventBotReady = "<bot-ready>"
	// RunEventError is a run the harness itself failed.
	RunEventError = "<error>"
)

// Failure is one failed assertion in a scenario run, or a failure of the run
// itself.
type Failure struct {
	// Turn is the 1-based turn number, or 0 for a failure before any turn.
	Turn int
	// Expectation is the 1-based expectation index within the turn, or 0 for a
	// turn-level failure (a send_after that never fired) or a failure of the run
	// itself.
	Expectation int
	// Event is the event name that was expected, or one of the RunEvent names
	// for a failure of the run itself.
	Event string
	// Reason explains what went wrong. It is free text and differs on every run,
	// often the judge's prose.
	Reason string
	// Kind is how the assertion failed, one of the Failure kinds. It says which
	// assertion gave way, not what that means about the bot, and is what failures
	// are grouped by across many runs.
	Kind string
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
	// TurnNotRun means the run ended before reaching the turn. It is distinct
	// from a pass: a run that stops at its first failure leaves its later turns
	// undriven, and counting those as passes would inflate any rate computed
	// from the result.
	TurnNotRun = "not_run"
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
// what went wrong, and these say what happened, a passing run included. A turn's
// expectations share one deadline and stop at the first to time out, so the turn
// is the unit a run is scored by.
type TurnResult struct {
	// Turn is the 1-based turn number.
	Turn int
	// Status is TurnPassed, TurnFailed, or TurnNotRun when the run ended before
	// reaching the turn (see Scenario.StopOnFailure).
	Status string
	// Failures is the turn's failed assertions, in order; empty unless Status is
	// TurnFailed.
	Failures []Failure
	// Expectations is what each of the turn's expectations resolved to, in
	// order, up to the one that timed out.
	Expectations []ExpectationResult
	// Duration is how long the turn took; zero when it was not run.
	Duration time.Duration
}

// Result is the outcome of running one scenario.
type Result struct {
	// Scenario is the scenario's name.
	Scenario string
	// Failures lists every failure; empty means the scenario passed. It is the
	// turns' failures flattened, plus any that belong to no turn, such as a run
	// that never connected.
	Failures []Failure
	// Turns is one record per scenario turn, in order: what it resolved to and
	// what each of its expectations matched. A turn the run never reached is
	// TurnNotRun, which is what a per-turn pass rate is computed from without the
	// scenario file for a denominator.
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
