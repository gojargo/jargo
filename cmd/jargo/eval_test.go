package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/coder/websocket"
	llmclassifier "github.com/gojargo/jargo/classifier/llm"
	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/spf13/pflag"
)

// echoLLM answers each turn by echoing the user's last message, so the CLI can
// be tested against a real (if trivial) bot.
type echoLLM struct{ *processor.Base }

func newEchoLLM() *echoLLM {
	e := &echoLLM{}
	e.Base = processor.New("EchoLLM", e)
	return e
}

func (e *echoLLM) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	cf, ok := f.(*frames.LLMContextFrame)
	if !ok {
		return e.PushFrame(ctx, f, dir)
	}
	msgs := cf.Context.Messages()
	last := ""
	if len(msgs) > 0 {
		last = msgs[len(msgs)-1].Text
	}
	_ = e.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	_ = e.PushFrame(ctx, frames.NewLLMTextFrame("echo: "+last), processor.Downstream)
	return e.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

func echoBot(in, out processor.Processor) *pipeline.Worker {
	agg := aggregators.New(frames.NewLLMContext("test"))
	rtviProc := rtvi.NewProcessor()
	return pipeline.NewWorker(pipeline.New(
		rtviProc, in, agg.User(), newEchoLLM(), out, agg.Assistant(),
	), pipeline.WorkerConfig{
		// The observer reports pipeline events; the processor carries them.
		Observers: []pipeline.Observer{rtvi.NewObserver(rtviProc)},
	})
}

func writeScenario(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "scenario.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestEvalRunAgainstBot(t *testing.T) {
	srv := httptest.NewServer(eval.Handler(echoBot))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	path := writeScenario(t, `
name: cli
turns:
  - user: "ping"
    expect:
      - event: llm_response
        text_contains: "echo: ping"
`)

	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs([]string{"eval", "run", path, "--bot-url", wsURL})
	cmd.SetOut(&out)
	cmd.SetErr(&out)

	if err := cmd.Execute(); err != nil {
		t.Fatalf("eval run failed: %v\noutput:\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "PASS cli") {
		t.Fatalf("expected PASS in output, got:\n%s", out.String())
	}
}

func TestEvalSuiteAgainstBots(t *testing.T) {
	srv := httptest.NewServer(eval.Handler(echoBot))
	defer srv.Close()
	ws := "ws" + strings.TrimPrefix(srv.URL, "http")

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "s.yaml"), []byte(`name: s
turns:
  - user: "ping"
    expect:
      - event: llm_response
        text_contains: "echo: ping"
`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "m.yaml"), fmt.Appendf(nil, `suite:
  - bot_url: %s
    scenarios: [s.yaml]
`, ws), 0o600); err != nil {
		t.Fatal(err)
	}

	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs([]string{"eval", "suite", filepath.Join(dir, "m.yaml")})
	cmd.SetOut(&out)
	cmd.SetErr(&out)

	if err := cmd.Execute(); err != nil {
		t.Fatalf("eval suite failed: %v\noutput:\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "1/1 scenarios passed") {
		t.Fatalf("expected summary in output, got:\n%s", out.String())
	}
}

func TestBuildJudge(t *testing.T) {
	if buildJudge(judgeOptions{}) != nil {
		t.Fatal("no --judge-model should yield no judge")
	}
	if buildJudge(judgeOptions{model: "gpt-4o-mini", explainer: true, explainBelow: 0.75}) == nil {
		t.Fatal("a --judge-model should yield a judge")
	}
}

// judgeEndpoint is an OpenAI-compatible endpoint answering each request with the
// next queued reply, and recording every request body.
type judgeEndpoint struct {
	*httptest.Server

	mu       sync.Mutex
	replies  []string
	requests []string
}

func newJudgeEndpoint(t *testing.T, replies ...string) *judgeEndpoint {
	t.Helper()
	e := &judgeEndpoint{replies: replies}
	e.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		e.mu.Lock()
		e.requests = append(e.requests, string(body))
		reply := ""
		if len(e.replies) > 0 {
			reply, e.replies = e.replies[0], e.replies[1:]
		}
		e.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": reply}}},
		})
	}))
	t.Cleanup(e.Close)
	return e
}

func (e *judgeEndpoint) bodies() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.requests)
}

// judgeFromFlags is the judge the --judge-* flags build, over endpoint.
func judgeFromFlags(t *testing.T, endpoint *judgeEndpoint, flags ...string) eval.Judge {
	t.Helper()
	fs := pflag.NewFlagSet("judge", pflag.ContinueOnError)
	get := addJudgeFlags(fs)
	args := append([]string{"--judge-model", "judge-model", "--judge-url", endpoint.URL, "--judge-key", "k"}, flags...)
	if err := fs.Parse(args); err != nil {
		t.Fatal(err)
	}
	judge := get()
	if judge == nil {
		t.Fatal("a --judge-model should yield a judge")
	}
	return judge
}

const (
	judgedNo      = `{"verdict": {"choice": "no", "probabilities": {"no": 0.9}}}`
	explainPrompt = "You explain the verdicts a judge gave"
)

// The --judge-* flags configure the judge: by default it classifies and
// explains with the one model, --judge-explainer=false leaves the reasons to
// the probabilities, and --judge-allow-continue=false judges a reply yes or no.
func TestJudgeFlags(t *testing.T) {
	t.Run("the default judge classifies and explains with one LLM", func(t *testing.T) {
		endpoint := newJudgeEndpoint(t, judgedNo, `{"reason": "it never mentions the weather"}`)
		judge := judgeFromFlags(t, endpoint)
		judge.AddAssistantMessage("Hello there.")
		v := judge.Evaluate(t.Context(), "describes the weather")
		if v.Verdict != eval.VerdictNo || v.Reason != "it never mentions the weather" {
			t.Fatalf("verdict = %+v", v)
		}
		bodies := endpoint.bodies()
		if len(bodies) != 2 || !strings.Contains(bodies[1], explainPrompt) {
			t.Fatalf("requests = %v, want a classification then an explanation", bodies)
		}
		for _, body := range bodies {
			if !strings.Contains(body, `"model":"judge-model"`) {
				t.Errorf("request %s is not to the judge's model", body)
			}
		}
		evalJudge, ok := judge.(*eval.EvalJudge)
		if !ok {
			t.Fatalf("judge = %T, want an EvalJudge", judge)
		}
		if _, ok := evalJudge.Classifier().(*llmclassifier.Classifier); !ok {
			t.Errorf("classifier = %T, want the judge's model behind an LLM classifier", evalJudge.Classifier())
		}
	})

	t.Run("explainer false builds no explainer", func(t *testing.T) {
		endpoint := newJudgeEndpoint(t, judgedNo)
		judge := judgeFromFlags(t, endpoint, "--judge-explainer=false")
		judge.AddAssistantMessage("Hello there.")
		v := judge.Evaluate(t.Context(), "describes the weather")
		if !strings.HasPrefix(v.Reason, "P(") || len(endpoint.bodies()) != 1 {
			t.Fatalf("verdict = %+v after %d requests, want the probabilities alone", v, len(endpoint.bodies()))
		}
	})

	t.Run("allow continue false is read from the flag", func(t *testing.T) {
		for flag, wantContinue := range map[string]bool{
			"--judge-allow-continue=true":  true,
			"--judge-allow-continue=false": false,
		} {
			endpoint := newJudgeEndpoint(t, judgedNo, `{"reason": "no"}`)
			judge := judgeFromFlags(t, endpoint, flag)
			judge.AddAssistantMessage("Hello there.")
			judge.Evaluate(t.Context(), "describes the weather")
			if got := strings.Contains(endpoint.bodies()[0], "- continue:"); got != wantContinue {
				t.Errorf("%s: continue offered = %v, want %v", flag, got, wantContinue)
			}
		}
	})

	t.Run("explain below is read from the flag", func(t *testing.T) {
		unsureYes := `{"verdict": {"choice": "yes", "probabilities": {"yes": 0.6, "no": 0.4}}}`
		endpoint := newJudgeEndpoint(t, unsureYes, `{"reason": "close enough"}`)
		judge := judgeFromFlags(t, endpoint, "--judge-explain-below", "0.5")
		judge.AddAssistantMessage("Sunny-ish.")
		if v := judge.Evaluate(t.Context(), "gives the weather"); !v.Passed() || len(endpoint.bodies()) != 1 {
			t.Fatalf("verdict = %+v after %d requests, want a yes surer than 0.5 left unexplained",
				v, len(endpoint.bodies()))
		}
	})
}

func TestEvalRunRequiresBotURL(t *testing.T) {
	path := writeScenario(t, "name: x\nturns:\n  - user: hi\n    expect:\n      - event: llm_started\n")

	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs([]string{"eval", "run", path})
	cmd.SetOut(&out)
	cmd.SetErr(&out)

	if err := cmd.Execute(); err == nil {
		t.Fatal("expected an error when --bot-url is missing")
	}
}

// TestEvalRunExpandsADirectory checks a folder of scenarios can be played by
// naming the folder, in name order, taking both YAML suffixes and leaving
// everything else alone.
func TestEvalRunExpandsADirectory(t *testing.T) {
	srv := httptest.NewServer(eval.Handler(echoBot))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	dir := t.TempDir()
	scenario := func(name string) string {
		return "name: " + name + "\nturns:\n  - user: \"ping\"\n    expect:\n" +
			"      - event: llm_response\n        text_contains: \"echo: ping\"\n"
	}
	for file, name := range map[string]string{
		"zeta.yaml":  "zeta",
		"alpha.yaml": "alpha",
		"beta.yml":   "beta",
	} {
		if err := os.WriteFile(filepath.Join(dir, file), []byte(scenario(name)), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("not a scenario"), 0o600); err != nil {
		t.Fatal(err)
	}

	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs([]string{"eval", "run", dir, "--bot-url", wsURL})
	cmd.SetOut(&out)
	cmd.SetErr(&out)

	if err := cmd.Execute(); err != nil {
		t.Fatalf("eval run failed: %v\noutput:\n%s", err, out.String())
	}
	got := out.String()
	alpha, beta, zeta := strings.Index(got, "PASS alpha"), strings.Index(got, "PASS beta"), strings.Index(got, "PASS zeta")
	if alpha < 0 || beta < 0 || zeta < 0 {
		t.Fatalf("want all three scenarios played, got:\n%s", got)
	}
	if alpha >= beta || beta >= zeta {
		t.Errorf("scenarios played out of name order:\n%s", got)
	}
}

// TestEvalRunRejectsADirectoryWithNoScenarios checks a folder holding nothing to
// play is refused: a run that plays nothing reads like a run that passed.
func TestEvalRunRejectsADirectoryWithNoScenarios(t *testing.T) {
	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs([]string{"eval", "run", t.TempDir(), "--bot-url", "ws://example.test"})
	cmd.SetOut(&out)
	cmd.SetErr(&out)

	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "no .yaml or .yml scenario files found") {
		t.Fatalf("err = %v, want a complaint that the directory holds no scenarios", err)
	}
}

// TestEvalRunRecordsAScenarioThatWillNotLoad checks a broken scenario fails on
// its own rather than ending the run: the others are still worth playing, and it
// shows in the tally like any other failure.
func TestEvalRunRecordsAScenarioThatWillNotLoad(t *testing.T) {
	srv := httptest.NewServer(eval.Handler(echoBot))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	dir := t.TempDir()
	broken := filepath.Join(dir, "broken.yaml")
	if err := os.WriteFile(broken, []byte("name: [unclosed\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	good := filepath.Join(dir, "working.yaml")
	body := "name: working\nturns:\n  - user: \"ping\"\n    expect:\n" +
		"      - event: llm_response\n        text_contains: \"echo: ping\"\n"
	if err := os.WriteFile(good, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}

	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs([]string{"eval", "run", dir, "--bot-url", wsURL})
	cmd.SetOut(&out)
	cmd.SetErr(&out)

	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "1 of 2") {
		t.Fatalf("err = %v, want one of the two scenarios counted as failed", err)
	}
	got := out.String()
	if !strings.Contains(got, "FAIL broken") {
		t.Errorf("want the broken scenario reported by name, got:\n%s", got)
	}
	if !strings.Contains(got, "PASS working") {
		t.Errorf("want the working scenario still played, got:\n%s", got)
	}
}

// TestExpandScenarioPathsLeavesIncludeFragmentsOut checks the blocks a
// directory's scenarios include, which have no name, are not played as
// scenarios, while a file that does not parse is still taken so the run
// reports it instead of hiding it.
func TestExpandScenarioPathsLeavesIncludeFragmentsOut(t *testing.T) {
	dir := t.TempDir()
	for file, body := range map[string]string{
		"alpha.yaml":      "name: alpha\nscenarios: [{name: alpha, turns: []}]\n",
		"judge_text.yaml": "modality: text\n",
		"simulator.yaml":  "service: openai\n",
		"broken.yaml":     "name: [broken\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, file), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	paths, err := expandScenarioPaths([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{filepath.Join(dir, "alpha.yaml"), filepath.Join(dir, "broken.yaml")}
	if !slices.Equal(paths, want) {
		t.Fatalf("paths = %v, want %v", paths, want)
	}
}

// TestEvalRunPlaysEachScenarioOfAGroupFile checks a file holding several
// scenarios is one run per scenario, each named under the file.
func TestEvalRunPlaysEachScenarioOfAGroupFile(t *testing.T) {
	srv := httptest.NewServer(eval.Handler(echoBot))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	dir := t.TempDir()
	group := filepath.Join(dir, "group.yaml")
	single := filepath.Join(dir, "single.yaml")
	for path, body := range map[string]string{
		group:  "name: g\nscenarios:\n  - name: a\n    turns: []\n  - name: b\n    turns: []\n",
		single: "name: single\nscenarios: [{name: single, turns: []}]\n",
	} {
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs([]string{"eval", "run", group, single, "--bot-url", wsURL})
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("eval run failed: %v\noutput:\n%s", err, out.String())
	}
	var played []string
	for line := range strings.Lines(out.String()) {
		if name, ok := strings.CutPrefix(strings.TrimSpace(line), "PASS "); ok {
			played = append(played, name)
		}
	}
	if want := []string{"g/a", "g/b", "single/single"}; !slices.Equal(played, want) {
		t.Fatalf("played %v, want %v\noutput:\n%s", played, want, out.String())
	}
}

// resultOf is a result whose turns have the given statuses.
func resultOf(statuses ...string) eval.Result {
	res := eval.Result{Scenario: "s"}
	for i, s := range statuses {
		res.Turns = append(res.Turns, eval.TurnResult{Turn: i + 1, Status: s})
	}
	return res
}

// TestTurnTally checks the rate beside a run's verdict: given only for a run
// that drove every turn and failed some.
func TestTurnTally(t *testing.T) {
	passed, failed, notRun := eval.TurnPassed, eval.TurnFailed, eval.TurnNotRun
	for _, tc := range []struct {
		statuses []string
		want     string
	}{
		// A fully driven run is counted, nothing passing included.
		{[]string{passed, failed, passed}, "2/3 turns"},
		{[]string{failed, failed}, "0/2 turns"},
		// The undriven turns were never attempted, so any fraction over them
		// would read as turns that failed.
		{[]string{passed, failed, notRun}, ""},
		{[]string{failed, notRun}, ""},
		// A run that passed already says so.
		{[]string{passed, passed}, ""},
		{nil, ""},
	} {
		if got := turnTally(resultOf(tc.statuses...)); got != tc.want {
			t.Errorf("turnTally(%v) = %q, want %q", tc.statuses, got, tc.want)
		}
	}
}

// runCLI runs `jargo eval run` on path against url with extra flags, returning
// the output and the error.
func runCLI(t *testing.T, path, url string, flags ...string) (string, error) {
	t.Helper()
	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs(append([]string{"eval", "run", path, "--bot-url", url}, flags...))
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	err := cmd.Execute()
	return out.String(), err
}

// TestEvalRunReportsABotThatRefusesTheConnection checks a run that never
// reaches the bot is a failure of that scenario, listed with its kind, rather
// than an error ending the command.
func TestEvalRunReportsABotThatRefusesTheConnection(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	path := writeScenario(t, "name: refused\nturns:\n  - user: hi\n    expect:\n      - event: llm_started\n")
	out, err := runCLI(t, path, "ws"+strings.TrimPrefix(srv.URL, "http"))
	if err == nil || !strings.Contains(err.Error(), "1 of 1") {
		t.Fatalf("err = %v, want the scenario counted as failed\noutput:\n%s", err, out)
	}
	if !strings.Contains(out, "FAIL refused") || !strings.Contains(out, "run <connect> connect_failed: ") ||
		!strings.Contains(out, "403") {
		t.Fatalf("output lacks the failure to connect:\n%s", out)
	}
}

// TestEvalRunTalliesTheTurnsOfARunThatDroveThemAll checks a failed run whose
// every turn was driven says how many passed, and lists each failure with
// where it was and its kind.
func TestEvalRunTalliesTheTurnsOfARunThatDroveThemAll(t *testing.T) {
	srv := httptest.NewServer(eval.Handler(echoBot))
	defer srv.Close()

	path := writeScenario(t, `
name: tally
stop_on_failure: false
turns:
  - user: "ping"
    expect:
      - event: llm_response
        text_contains: "pong"
        within_ms: 300
  - user: "ping"
    expect:
      - event: llm_response
        text_contains: "echo: ping"
`)
	out, err := runCLI(t, path, "ws"+strings.TrimPrefix(srv.URL, "http"))
	if err == nil {
		t.Fatalf("expected the run to fail\noutput:\n%s", out)
	}
	if !strings.Contains(out, "FAIL tally 1/2 turns") ||
		!strings.Contains(out, "turn 1 llm_response text_mismatch: ") {
		t.Fatalf("output lacks the tally or the failure:\n%s", out)
	}
}

// TestEvalRunBotReadyTimeoutFlag checks --bot-ready-timeout bounds the wait for
// a bot that accepts the connection but never says it is ready.
func TestEvalRunBotReadyTimeoutFlag(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		for {
			if _, _, err := conn.Read(r.Context()); err != nil {
				return
			}
		}
	}))
	defer srv.Close()

	path := writeScenario(t, "name: silent\nturns:\n  - user: hi\n    expect:\n      - event: llm_started\n")
	out, _ := runCLI(t, path, "ws"+strings.TrimPrefix(srv.URL, "http"), "--bot-ready-timeout", "0.3")
	if !strings.Contains(out, "run <bot-ready> handshake_timeout: bot-ready not received within 300ms") {
		t.Fatalf("output lacks the handshake timeout:\n%s", out)
	}
}
