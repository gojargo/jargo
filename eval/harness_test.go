package eval_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
)

// fakeLLM answers each turn deterministically so the harness can be exercised
// end-to-end without a real model: it echoes the user's text, and on a weather
// question it emits a get_weather tool call instead.
type fakeLLM struct {
	*processor.Base
}

func newFakeLLM() *fakeLLM {
	f := &fakeLLM{}
	f.Base = processor.New("FakeLLM", f)
	return f
}

func (f *fakeLLM) ProcessFrame(ctx context.Context, frame frames.Frame, dir processor.Direction) error {
	if err := f.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	cf, ok := frame.(*frames.LLMContextFrame)
	if !ok {
		return f.PushFrame(ctx, frame, dir)
	}

	msgs := cf.Context.Messages()
	last := ""
	if len(msgs) > 0 {
		last = msgs[len(msgs)-1].Text
	}

	if err := f.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	if strings.Contains(strings.ToLower(last), "weather") {
		call := frames.NewFunctionCallInProgressFrame("call-1", "get_weather")
		if err := f.PushFrame(ctx, call, processor.Downstream); err != nil {
			return err
		}
	} else if err := f.PushFrame(ctx, frames.NewLLMTextFrame("you said: "+last), processor.Downstream); err != nil {
		return err
	}
	return f.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

func buildFakeBot(in, out processor.Processor) *pipeline.Task {
	agg := aggregators.New(frames.NewLLMContext("test system"))
	return pipeline.NewTask(pipeline.New(
		in, agg.User(), newFakeLLM(), rtvi.NewProcessor(), out, agg.Assistant(),
	), pipeline.TaskParams{})
}

func host(t *testing.T, body string) eval.Result {
	t.Helper()
	scenario, err := eval.Load(writeScenario(t, body))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	res, err := eval.Host(ctx, scenario, buildFakeBot, nil)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestHarnessTextResponse(t *testing.T) {
	res := host(t, `
name: echo
turns:
  - user: "hello world"
    expect:
      - event: llm_started
      - event: llm_response
        text_contains: "hello world"
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

func TestHarnessFunctionCall(t *testing.T) {
	res := host(t, `
name: weather
turns:
  - user: "what's the weather in Paris?"
    expect:
      - event: function_call
        function: get_weather
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

func TestHarnessReportsTextMismatch(t *testing.T) {
	res := host(t, `
name: mismatch
turns:
  - user: "hello"
    expect:
      - event: llm_response
        text_contains: "goodbye"
`)
	if res.Passed() {
		t.Fatal("expected a failure for the unmet text_contains")
	}
	if !strings.Contains(res.Failures[0].Reason, "does not contain") {
		t.Fatalf("unexpected failure reason: %s", res.Failures[0].Reason)
	}
}

func TestHarnessReportsMissingFunction(t *testing.T) {
	res := host(t, `
name: missing-call
turns:
  - user: "hello"
    expect:
      - event: function_call
        function: get_weather
        within_ms: 800
`)
	if res.Passed() {
		t.Fatal("expected a timeout failure for the tool call that never happens")
	}
	if !strings.Contains(res.Failures[0].Reason, "no matching function_call") {
		t.Fatalf("unexpected failure reason: %s", res.Failures[0].Reason)
	}
}
