package main

import (
	"context"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
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
	if buildJudge("", "", "") != nil {
		t.Fatal("no --judge-model should yield no judge")
	}
	if buildJudge("gpt-4o-mini", "", "") == nil {
		t.Fatal("a --judge-model should yield a judge")
	}
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
