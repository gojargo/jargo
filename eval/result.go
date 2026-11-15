package eval

import (
	"fmt"
	"slices"
	"strings"
	"sync"
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
	// FailureBotEnded means the bot ended the call before a turn's send_after
	// or expectation was met.
	FailureBotEnded = "bot_ended"
	// FailureConnectFailed means the harness never connected to the bot.
	FailureConnectFailed = "connect_failed"
	// FailureClosedBeforeReady means the bot closed the connection before
	// bot-ready.
	FailureClosedBeforeReady = "closed_before_ready"
	// FailureHandshakeTimeout means the bot connected but never sent bot-ready.
	FailureHandshakeTimeout = "handshake_timeout"
	// FailureAudioUnsupported means bot-ready said the transport is off for
	// audio the run uses.
	FailureAudioUnsupported = "audio_unsupported"
	// FailureJudgeNoVerdict means a judge call failed, so it decided nothing.
	FailureJudgeNoVerdict = "judge_no_verdict"
	// FailureError means the harness itself failed (a sub-pipeline, the judge,
	// sending to the bot), not the bot.
	FailureError = "error"
)

// FailureKinds are the categories of AssertionFailure.Kind.
//
//nolint:gochecknoglobals // fixed list, read only
var FailureKinds = []string{
	FailureTimeout, FailureJudgeNo, FailureJudgeContinue, FailureNoJudge, FailureNoContent,
	FailureTextMismatch, FailureTextPresent, FailureTextUnmatched, FailureMissingFunctionCall,
	FailureFunctionArgsMismatch, FailureUnexpectedEvent, FailureSendAfterTimeout, FailureBotEnded,
	FailureConnectFailed, FailureClosedBeforeReady, FailureHandshakeTimeout, FailureAudioUnsupported,
	FailureJudgeNoVerdict, FailureError,
}

// The event names a failure of the run itself carries, rather than one of an
// expectation.
const (
	// RunEventConnect is a run that never connected to the bot.
	RunEventConnect = "<connect>"
	// RunEventBotReady is a run whose bot never became ready.
	RunEventBotReady = "<bot-ready>"
	// RunEventHarness is a run the harness itself failed.
	RunEventHarness = "<harness>"
)

// AssertionFailure is a single failed assertion within an eval, or a failure
// of the run itself.
type AssertionFailure struct {
	// TurnIndex is the index of the turn that failed, from 0, or -1 for a
	// failure of the run before any turn.
	TurnIndex int
	// ExpectationIndex is the index of the expectation within the turn, from
	// 0, or -1 for a turn-level failure (a send_after that never fired) or a
	// failure of the run itself.
	ExpectationIndex int
	// EventName is the expectation's event name, or one of the RunEvent names
	// for a failure of the run itself.
	EventName string
	// Reason explains what went wrong. It is free text and differs on every
	// run, often the judge's prose.
	Reason string
	// Kind is how the assertion failed, one of FailureKinds. It says which
	// assertion gave way, not what that means about the bot, and is what
	// failures are grouped by across many runs.
	Kind string
}

// Failure is an assertion failure.
//
// Deprecated: Use AssertionFailure instead.
type Failure = AssertionFailure

// String renders a failure as a single line.
func (f AssertionFailure) String() string {
	return fmt.Sprintf("turn %d expectation %d (%s): %s", f.TurnIndex, f.ExpectationIndex, f.EventName, f.Reason)
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

// TurnStatuses are the values ScriptTurnResult.Status takes.
//
//nolint:gochecknoglobals // fixed list, read only
var TurnStatuses = []string{TurnPassed, TurnFailed, TurnNotRun}

// ExpectationResult is what one expectation of a turn resolved to, pass or
// fail.
type ExpectationResult struct {
	// ExpectationIndex is the index of the expectation within the turn, from 0.
	ExpectationIndex int
	// EventName is the expectation's event name.
	EventName string
	// Passed reports whether the expectation was satisfied.
	Passed bool
	// Matched is what it matched, in short, when it passed: the marker of an
	// llm_marker, a function call's signature, the text of a reply or a
	// transcript. It is empty for an event carrying no text, and on a failure,
	// whose reason is in the turn's failures.
	Matched string
}

// ScriptTurnResult is the outcome of one turn within a scenario run.
//
// A turn's expectations share one deadline and stop at the first to time out,
// so the turn is the unit a run is scored by.
type ScriptTurnResult struct {
	// TurnIndex is the index of the turn in the scenario, from 0.
	TurnIndex int
	// Status is one of TurnStatuses. TurnNotRun means the run ended before
	// reaching this turn (see ScriptScenario.StopOnFailure).
	Status string
	// Failures are the turn's failed assertions, in order; empty unless Status
	// is TurnFailed.
	Failures []AssertionFailure
	// Expectations are what each expectation resolved to, in order, up to the
	// one that timed out; what a passed run matched is only recorded here.
	Expectations []ExpectationResult
	// DurationMS is the wall-clock time the turn took, in milliseconds; 0 when
	// the turn was not run.
	DurationMS int
}

// TurnResult is the outcome of one turn of a scripted scenario.
//
// Deprecated: Use ScriptTurnResult instead.
type TurnResult = ScriptTurnResult

// ScriptResult is the outcome of running a scripted scenario.
type ScriptResult struct {
	// ScenarioName is the name of the scenario that was run.
	ScenarioName string
	// Failures are the assertions that failed, in order. It is the turns'
	// failures flattened, plus any that belong to no turn, such as a run that
	// never connected.
	Failures []AssertionFailure
	// Turns is one record per scenario turn, in order: what a per-turn pass
	// rate is computed from, without needing the scenario file for a
	// denominator.
	Turns []ScriptTurnResult
	// DurationMS is the wall-clock time the run took, in milliseconds.
	DurationMS int
	// EventsSeen is every event observed, for diagnostics, whether or not the
	// scenario asserted on it: what a failure is read against.
	EventsSeen []map[string]any
	// DebugLog is a timestamped trace of the harness's own decisions (events
	// received, audio transcribed, matcher progress), for diagnosing a flaky
	// run.
	DebugLog []string
	// Skipped, when set, means the scenario was not run, and says why. Such a
	// result is neither passed nor failed.
	Skipped string
}

// Result is the outcome of running a scripted scenario.
//
// Deprecated: Use ScriptResult instead.
type Result = ScriptResult

// Passed reports whether every assertion passed and the scenario was run.
func (r ScriptResult) Passed() bool { return len(r.Failures) == 0 && r.Skipped == "" }

// String renders a human-readable summary of the run.
func (r ScriptResult) String() string {
	switch {
	case r.Skipped != "":
		return fmt.Sprintf("SKIP %s (%s)", r.ScenarioName, r.Skipped)
	case r.Passed():
		return "PASS " + r.ScenarioName
	}
	var b strings.Builder
	fmt.Fprintf(&b, "FAIL %s (%d failure(s))", r.ScenarioName, len(r.Failures))
	for _, f := range r.Failures {
		fmt.Fprintf(&b, "\n  - %s", f)
	}
	return b.String()
}

// How a simulation run came to an end, for SimulationResult.EndedBy.
const (
	// SimulationEndedByEndCall means the persona called its end_call tool.
	SimulationEndedByEndCall = "end_call"
	// SimulationEndedByBot means the bot ended the call (it closed the
	// connection).
	SimulationEndedByBot = "bot"
	// SimulationEndedByMaxTurns means the persona's turn cap was reached.
	SimulationEndedByMaxTurns = "max_turns"
	// SimulationEndedByMaxDuration means the run's wall-clock cap was reached.
	SimulationEndedByMaxDuration = "max_duration"
	// SimulationEndedBySilence means neither side did anything for
	// max_silence_s.
	SimulationEndedBySilence = "silence"
	// SimulationEndedByError means the run did not complete (see
	// SimulationResult.Error).
	SimulationEndedByError = "error"
)

// SimulationEndings are the values SimulationResult.EndedBy takes.
//
//nolint:gochecknoglobals // fixed list, read only
var SimulationEndings = []string{
	SimulationEndedByEndCall, SimulationEndedByBot, SimulationEndedByMaxTurns,
	SimulationEndedByMaxDuration, SimulationEndedBySilence, SimulationEndedByError,
}

// SimulationTurnVerdict is the judge's verdict on one bot turn, for a per-turn
// metric.
type SimulationTurnVerdict struct {
	// Turn is the bot turn, from 1, counting the turns in which the bot said
	// something; it indexes the assistant messages of the run's conversation.
	Turn int
	// Passed reports whether the turn satisfied the criterion.
	Passed bool
	// Reason is the explainer's one-sentence reason for the verdict, or the
	// classifier's probabilities when nothing explained it.
	Reason string
	// Verdict is the judge's answer: yes, no, or none when it gave no verdict
	// on the turn, which counts as a no.
	Verdict string
	// Confidence is how sure the judge is of the verdict, from 0 to 1; nil when
	// it gave none.
	Confidence *float64
	// Probabilities is the probability of each outcome the classifier was
	// offered; empty when it gave no verdict.
	Probabilities map[string]float64
}

// SimulationMetricScore is one quality metric's outcome for a simulation run.
type SimulationMetricScore struct {
	// Name is the metric's name, from the simulation file.
	Name string
	// Score is the share of the bot's turns the judge answered yes for, in
	// 0..1; nil when there was no turn to judge.
	Score *float64
	// Passed reports whether the metric let the run pass: its score reached its
	// MinScore, or it has none.
	Passed bool
	// Reason is what the score rests on: the turns that fell short and why, or
	// that every turn passed.
	Reason string
	// MinScore is the score the metric needed, or nil when it only reports.
	MinScore *float64
	// Verdicts are the judge's verdict on each bot turn, in order; empty for a
	// measured metric.
	Verdicts []SimulationTurnVerdict
	// Value is what a measured metric measured, in its unit; nil for a judged
	// one, or when there was nothing to measure.
	Value *float64
	// FailureKind is how the metric failed, for grouping across runs: judge_no
	// when the judge rejected a turn, judge_no_verdict when it only left turns
	// unanswered, out_of_range for a measure outside its bounds,
	// function_calls for a call list that did not match, error when a required
	// measurement was unavailable; empty when it passed.
	FailureKind string
	// Criterion is the criterion each bot turn was judged on, as the
	// simulation worded it; empty for a measured metric.
	Criterion string
}

// SimulationResult is the outcome of one run of a simulation.
type SimulationResult struct {
	// SimulationName is the name of the simulation that was run.
	SimulationName string
	// Succeeded reports whether the judge decided the goal was achieved.
	Succeeded bool
	// Reason is the judge's justification for Succeeded, or the error.
	Reason string
	// Error, when set, means the run did not complete (a failed connect, a
	// harness error); Succeeded is then false and the run is neither a goal
	// success nor a goal failure.
	Error string
	// FailureKind is what kind of failure Error is, one of FailureKinds;
	// empty when no failure ended the run.
	FailureKind string
	// Metrics are the quality metrics' outcomes.
	Metrics []SimulationMetricScore
	// Messages is the conversation, with the persona's turns as user messages
	// and the bot's as assistant; the assistant messages are the bot turns the
	// metrics scored.
	Messages []map[string]any
	// Conversation is the conversation as a conversation file holds it: the
	// messages with the bot's function calls in place.
	Conversation []map[string]any
	// Turns is how many turns the persona took.
	Turns int
	// EndedBy is how the run ended, one of SimulationEndings.
	EndedBy string
	// EndCall is the persona's own end_call claim (success, reason) when it
	// made one; advisory, the judge decides Succeeded.
	EndCall map[string]any
	// DurationMS is the wall-clock time the run took, in milliseconds.
	DurationMS int
	// EventsSeen is every event observed, for diagnostics.
	EventsSeen []map[string]any
	// DebugLog is a timestamped trace of the harness's own decisions.
	DebugLog []string
	// Confidence is how sure the judge is of its goal verdict, from 0 to 1; nil
	// when it gave none.
	Confidence *float64
	// Probabilities is the probability of each answer to the goal question;
	// empty when the judge gave none.
	Probabilities map[string]float64
	// Success is the goal the conversation was judged on, the simulation's
	// success as it was worded.
	Success string
}

// Passed reports whether the run completed, achieved its goal, and no metric
// fell short.
func (r *SimulationResult) Passed() bool {
	if r.Error != "" || !r.Succeeded {
		return false
	}
	for _, m := range r.Metrics {
		if !m.Passed {
			return false
		}
	}
	return true
}

// Failure is why the run did not pass, or "" when it did: the error, the goal,
// or the first failed metric.
func (r *SimulationResult) Failure() string {
	if r.Error != "" {
		return r.Error
	}
	if !r.Succeeded {
		return "goal not met: " + r.Reason
	}
	for _, m := range r.Metrics {
		if m.Passed {
			continue
		}
		if m.MinScore == nil {
			return m.Name + ": " + m.Reason
		}
		score := "unscored"
		if m.Score != nil {
			score = fmt.Sprintf("%.2f", *m.Score)
		}
		return fmt.Sprintf("%s %s below %.2f: %s", m.Name, score, *m.MinScore, m.Reason)
	}
	return ""
}

// The statuses of a ScriptTurnProgress.
const (
	// StatusTurn heads a turn, carrying its input as the event.
	StatusTurn = "turn"
	// StatusMatched means the expectation was met.
	StatusMatched = "matched"
	// StatusFailed means the event arrived but did not satisfy the expectation.
	StatusFailed = "failed"
	// StatusTimeout means nothing of the expectation's kind arrived at all
	// before its budget expired.
	StatusTimeout = "timeout"
)

// ScriptTurnProgress is a progress record emitted while a turn runs, for a
// caller reporting progress live.
type ScriptTurnProgress struct {
	// TurnIndex is the turn being run, from 0.
	TurnIndex int
	// ExpectationIndex is the index of the expectation, from 0, or -1 for a
	// turn-level record (the turn header, or a send_after that never fired).
	ExpectationIndex int
	// EventName is the expectation's event, or the turn's input for a turn
	// header.
	EventName string
	// Status is StatusTurn, StatusMatched, StatusFailed or StatusTimeout.
	Status string
	// Detail is optional extra text: the failure reason, the user utterance.
	Detail string
}

// Progress is a progress record emitted while a turn runs.
//
// Deprecated: Use ScriptTurnProgress instead.
type Progress = ScriptTurnProgress

// SimulationProgress is a progress record emitted while a simulation runs.
type SimulationProgress struct {
	// Status is "bot" for a response the bot finished, "user" for a turn the
	// persona spoke, or "ended" once the conversation is over.
	Status string
	// Text is what was said; for "ended", how the conversation ended (one of
	// SimulationEndings).
	Text string
	// Turn is the persona's turn count so far.
	Turn int
}

// Trace is a timestamped, turn-tagged log of the harness's own decisions, kept
// as a result's DebugLog.
//
// The tag is the turn the harness is processing (`[--]` before the first). An
// event that lands while a turn waits on send_after is tagged with that turn
// even though it is the previous turn's output; the send: lines mark the
// boundary. A Trace is safe for concurrent use.
type Trace struct {
	mu    sync.Mutex
	lines []string
	// turn is the index of the turn being processed; -1 outside any turn.
	turn int
	t0   time.Time
}

// NewTrace returns an empty trace, outside any turn; Start anchors its clock.
func NewTrace() *Trace {
	return &Trace{turn: -1}
}

// Start anchors the trace's timestamps at now, the start of the run.
func (t *Trace) Start() {
	t.mu.Lock()
	t.t0 = time.Now()
	t.mu.Unlock()
}

// SetTurn sets the turn being processed, from 0, or -1 outside any turn.
func (t *Trace) SetTurn(turn int) {
	t.mu.Lock()
	t.turn = turn
	t.mu.Unlock()
}

// Turn is the turn being processed, or -1 outside any turn.
func (t *Trace) Turn() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.turn
}

// Log appends one line, stamped with the seconds since Start and the turn.
func (t *Trace) Log(msg string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	var elapsed float64
	if !t.t0.IsZero() {
		elapsed = time.Since(t.t0).Seconds()
	}
	tag := "--"
	if t.turn >= 0 {
		tag = fmt.Sprintf("t%d", t.turn)
	}
	t.lines = append(t.lines, fmt.Sprintf("%8.3f  [%3s]  %s", elapsed, tag, msg))
}

// Lines are the lines logged so far.
func (t *Trace) Lines() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return slices.Clone(t.lines)
}
