package eval_test

import (
	"testing"

	"github.com/gojargo/jargo/eval"
)

// Tests for the entry points a bot's own test file calls. They stand the bot up
// on a loopback WebSocket, play the scenario, and report each unmet expectation
// through the testing.T they were handed, which is what makes an eval scenario
// an ordinary `go test` failure.

// echoScenario is met by the fake bot, which echoes what it was told.
const echoScenario = `
name: echo
turns:
  - user: "hello world"
    expect:
      - event: llm_response
        text_contains: "hello world"
`

// TestRun checks the plain entry point: load the scenario, host the bot, and
// report nothing when every expectation is met.
func TestRun(t *testing.T) {
	eval.Run(t, writeScenario(t, echoScenario), buildFakeBot)
}

// TestRunWithJudge checks the judge entry point. A scenario with no `judge:`
// assertion needs none, which is why nil is allowed.
func TestRunWithJudge(t *testing.T) {
	eval.RunWithJudge(t, writeScenario(t, echoScenario), buildFakeBot, nil)
}

// TestRunWith checks the explicit-options entry point with its zero value, the
// form the other two delegate to.
func TestRunWith(t *testing.T) {
	eval.RunWith(t, writeScenario(t, echoScenario), buildFakeBot, eval.Options{})
}

// TestRunWithReportsProgress checks the progress callback fires as the run goes,
// for a caller reporting a long scenario as it happens rather than at the end.
func TestRunWithReportsProgress(t *testing.T) {
	var updates []eval.Progress
	eval.RunWith(t, writeScenario(t, echoScenario), buildFakeBot, eval.Options{
		OnProgress: func(p eval.Progress) { updates = append(updates, p) },
	})

	if len(updates) == 0 {
		t.Fatal("the run reported no progress")
	}
}

// TestResultString checks the one-line summary a runner prints, and the
// expanded form that lists what actually went wrong.
func TestResultString(t *testing.T) {
	passed := eval.Result{Scenario: "echo"}
	if got := passed.String(); got != "PASS echo" {
		t.Errorf("String() = %q, want %q", got, "PASS echo")
	}
	if !passed.Passed() {
		t.Error("a result with no failures does not report as passed")
	}

	failed := eval.Result{
		Scenario: "weather",
		Failures: []eval.Failure{
			{Turn: 1, Expectation: 2, Event: "function_call", Reason: "never arrived"},
			{Turn: 3, Expectation: 1, Event: "llm_response", Reason: "text did not match"},
		},
	}
	if failed.Passed() {
		t.Error("a result with failures reports as passed")
	}

	want := "FAIL weather (2 failure(s))" +
		"\n  - turn 1 expectation 2 (function_call): never arrived" +
		"\n  - turn 3 expectation 1 (llm_response): text did not match"
	if got := failed.String(); got != want {
		t.Errorf("String() =\n%q\nwant\n%q", got, want)
	}
}

// TestFailureString checks a single unmet expectation renders on one line, which
// is what reaches the test output through t.Errorf.
func TestFailureString(t *testing.T) {
	f := eval.Failure{Turn: 2, Expectation: 0, Event: "llm_started", Reason: "timed out"}
	want := "turn 2 expectation 0 (llm_started): timed out"
	if got := f.String(); got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
}

// TestJudgeVerdictPassed checks only a definite yes passes: a judge that cannot
// decide has not agreed, and must not be read as if it had.
func TestJudgeVerdictPassed(t *testing.T) {
	tests := []struct {
		verdict string
		want    bool
	}{
		{verdict: eval.VerdictYes, want: true},
		{verdict: eval.VerdictNo},
		{verdict: eval.VerdictContinue},
		{verdict: ""},
	}
	for _, tt := range tests {
		got := eval.JudgeVerdict{Verdict: tt.verdict}.Passed()
		if got != tt.want {
			t.Errorf("Passed() with verdict %q = %v, want %v", tt.verdict, got, tt.want)
		}
	}
}
