package eval_test

import (
	"context"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// outputSettings reports the LLM output settings that reach the bot's pipeline.
type outputSettings struct {
	*processor.Base
	seen chan<- bool
}

func (o *outputSettings) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := o.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if c, ok := f.(*frames.LLMConfigureOutputFrame); ok {
		o.seen <- c.SkipTTS
	}
	return o.PushFrame(ctx, f, dir)
}

// TestHandlerConfiguresEachConnectionsOutput covers a handler serving several
// connections in a row: each gets the output setting its own URL asks for, an
// audio connection after a text one included, whether it says so or not.
func TestHandlerConfiguresEachConnectionsOutput(t *testing.T) {
	for _, audioPath := range []string{"/", "/?skip_tts=false"} {
		t.Run(audioPath, func(t *testing.T) {
			seen := make(chan bool, 4)
			srv := httptest.NewServer(eval.Handler(func(in, out processor.Processor) *pipeline.Worker {
				o := &outputSettings{seen: seen}
				o.Base = processor.New("OutputSettings", o)
				return pipeline.NewWorker(pipeline.New(in, o, out), pipeline.WorkerConfig{})
			}))
			defer srv.Close()

			var observed []bool
			for _, path := range []string{"/?skip_tts=true", audioPath, "/?skip_tts=true"} {
				ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
				conn, resp, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(srv.URL, "http")+path, nil)
				if resp != nil && resp.Body != nil {
					_ = resp.Body.Close()
				}
				if err != nil {
					cancel()
					t.Fatalf("dial %s: %v", path, err)
				}
				select {
				case skipTTS := <-seen:
					observed = append(observed, skipTTS)
				case <-ctx.Done():
					t.Fatalf("%s: the bot's output was never configured", path)
				}
				_ = conn.CloseNow()
				cancel()
			}

			// Each connection must see its own session's output setting.
			if want := []bool{true, false, true}; !slices.Equal(observed, want) {
				t.Fatalf("skip_tts per connection = %v, want %v", observed, want)
			}
		})
	}
}

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
	passed := eval.ScriptResult{ScenarioName: "echo"}
	if got := passed.String(); got != "PASS echo" {
		t.Errorf("String() = %q, want %q", got, "PASS echo")
	}
	if !passed.Passed() {
		t.Error("a result with no failures does not report as passed")
	}

	failed := eval.ScriptResult{
		ScenarioName: "weather",
		Failures: []eval.AssertionFailure{
			{TurnIndex: 0, ExpectationIndex: 1, EventName: "function_call", Reason: "never arrived"},
			{TurnIndex: 2, ExpectationIndex: 0, EventName: "llm_response", Reason: "text did not match"},
		},
	}
	if failed.Passed() {
		t.Error("a result with failures reports as passed")
	}

	want := "FAIL weather (2 failure(s))" +
		"\n  - turn 0 expectation 1 (function_call): never arrived" +
		"\n  - turn 2 expectation 0 (llm_response): text did not match"
	if got := failed.String(); got != want {
		t.Errorf("String() =\n%q\nwant\n%q", got, want)
	}

	skipped := eval.ScriptResult{ScenarioName: "audio", Skipped: "asserts response in text modality"}
	if skipped.Passed() {
		t.Error("a skipped result reports as passed")
	}
	if got := skipped.String(); got != "SKIP audio (asserts response in text modality)" {
		t.Errorf("String() = %q", got)
	}
}

// TestFailureString checks a single unmet expectation renders on one line, which
// is what reaches the test output through t.Errorf. A turn-level failure, a
// send_after that never fired, is at expectation -1, and a failure of the run
// before any turn at turn -1.
func TestFailureString(t *testing.T) {
	for _, tc := range []struct {
		f    eval.AssertionFailure
		want string
	}{
		{
			eval.AssertionFailure{TurnIndex: 1, ExpectationIndex: -1, EventName: "llm_started", Reason: "timed out"},
			"turn 1 expectation -1 (llm_started): timed out",
		},
		{
			eval.AssertionFailure{TurnIndex: -1, ExpectationIndex: -1, EventName: eval.RunEventHarness, Reason: "boom"},
			"turn -1 expectation -1 (<harness>): boom",
		},
	} {
		if got := tc.f.String(); got != tc.want {
			t.Errorf("String() = %q, want %q", got, tc.want)
		}
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
