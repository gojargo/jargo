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

func echoBot(in, out processor.Processor) *pipeline.Task {
	agg := aggregators.New(frames.NewLLMContext("test"))
	rtviProc := rtvi.NewProcessor()
	return pipeline.NewTask(pipeline.New(
		rtviProc, in, agg.User(), newEchoLLM(), out, agg.Assistant(),
	), pipeline.TaskParams{
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
