package eval_test

import (
	"regexp"
	"testing"

	"github.com/gojargo/jargo/eval"
)

func TestSimulationResultPassedNeedsSuccessAndNoError(t *testing.T) {
	if !(&eval.SimulationResult{SimulationName: "s", Succeeded: true}).Passed() {
		t.Error("a succeeded run should pass")
	}
	if (&eval.SimulationResult{SimulationName: "s", Succeeded: false}).Passed() {
		t.Error("a run that missed its goal should not pass")
	}
	if (&eval.SimulationResult{SimulationName: "s", Succeeded: true, Error: "boom"}).Passed() {
		t.Error("a run that errored should not pass")
	}
}

// Failure says why a run did not pass: the error, the goal, or the first metric
// that fell short.
func TestSimulationResultFailure(t *testing.T) {
	half, one := 0.5, 1.0
	for _, tc := range []struct {
		name string
		res  eval.SimulationResult
		want string
	}{
		{"passed", eval.SimulationResult{Succeeded: true}, ""},
		{"error", eval.SimulationResult{Succeeded: true, Error: "failed to connect"}, "failed to connect"},
		{"goal", eval.SimulationResult{Reason: "no booking"}, "goal not met: no booking"},
		{"metric below", eval.SimulationResult{Succeeded: true, Metrics: []eval.SimulationMetricScore{
			{Name: "politeness", Score: &half, MinScore: &one, Reason: "turn 1: because stayed polite"},
		}}, "politeness 0.50 below 1.00: turn 1: because stayed polite"},
		{"metric unscored", eval.SimulationResult{Succeeded: true, Metrics: []eval.SimulationMetricScore{
			{Name: "politeness", MinScore: &one, Reason: "no bot turn to judge"},
		}}, "politeness unscored below 1.00: no bot turn to judge"},
		{"measured", eval.SimulationResult{Succeeded: true, Metrics: []eval.SimulationMetricScore{
			{Name: "booking", Reason: "expected book_table(party_size=2)"},
		}}, "booking: expected book_table(party_size=2)"},
	} {
		if got := tc.res.Failure(); got != tc.want {
			t.Errorf("%s: Failure() = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// The trace stamps each line with the seconds since the run began and the turn
// it was logged in, `--` outside any turn.
func TestTraceTagsTheTurn(t *testing.T) {
	trace := eval.NewTrace()
	trace.Start()
	trace.Log("run: start")
	trace.SetTurn(0)
	trace.Log("send: hi")
	lines := trace.Lines()
	if len(lines) != 2 {
		t.Fatalf("lines = %q", lines)
	}
	if !regexp.MustCompile(`^ +\d+\.\d{3}  \[ --\]  run: start$`).MatchString(lines[0]) {
		t.Errorf("line = %q", lines[0])
	}
	if !regexp.MustCompile(`^ +\d+\.\d{3}  \[ t0\]  send: hi$`).MatchString(lines[1]) {
		t.Errorf("line = %q", lines[1])
	}
}
