package main

import (
	"context"
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
	return pipeline.NewTask(pipeline.New(
		in, agg.User(), newEchoLLM(), rtvi.NewProcessor(), out, agg.Assistant(),
	), pipeline.TaskParams{})
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
