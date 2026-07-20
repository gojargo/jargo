package eval_test

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// fakeGen is a canned llm.Generator: it emits reply and counts its calls.
type fakeGen struct {
	reply string
	calls atomic.Int32
}

func (g *fakeGen) Generate(_ context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.calls.Add(1)
	return emit(g.reply)
}

func TestLLMJudgeVerdicts(t *testing.T) {
	cases := map[string]struct {
		out      string
		wantPass bool
	}{
		"pass first line":   {"PASS — greets the user warmly", true},
		"fail first line":   {"FAIL: too terse", false},
		"yes":               {"YES, it answers the question", true},
		"no":                {"NO, it dodges", false},
		"pass mid-text":     {"The reply is good, so PASS.", true},
		"no verdict word":   {"The reply seems fine.", false},
		"contradictory ord": {"PASS. It does not FAIL the criterion.", true},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			j := eval.NewLLMJudge(&fakeGen{reply: tc.out})
			pass, reason, err := j.Evaluate(context.Background(), "greets warmly", "hi there")
			if err != nil {
				t.Fatal(err)
			}
			if pass != tc.wantPass {
				t.Fatalf("pass = %v, want %v (reason %q)", pass, tc.wantPass, reason)
			}
		})
	}
}

func TestLLMJudgeCaches(t *testing.T) {
	gen := &fakeGen{reply: "PASS ok"}
	j := eval.NewLLMJudge(gen)
	for range 3 {
		if _, _, err := j.Evaluate(context.Background(), "crit", "reply"); err != nil {
			t.Fatal(err)
		}
	}
	// A different (criterion, reply) is a cache miss.
	if _, _, err := j.Evaluate(context.Background(), "crit", "other reply"); err != nil {
		t.Fatal(err)
	}
	if got := gen.calls.Load(); got != 2 {
		t.Fatalf("generator called %d times, want 2 (one per distinct input)", got)
	}
}

func TestHarnessJudgePasses(t *testing.T) {
	// The bot echoes "you said: hello there"; the judge is told to PASS.
	scenario, err := eval.Load(writeScenario(t, `
name: judged
turns:
  - user: "hello there"
    expect:
      - event: llm_response
        judge: "acknowledges the user's greeting"
`))
	if err != nil {
		t.Fatal(err)
	}
	judge := eval.NewLLMJudge(&fakeGen{reply: "PASS — it echoes the greeting"})
	res, err := eval.Host(context.Background(), scenario, buildFakeBot, eval.Options{Judge: judge})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

func TestHarnessJudgeFailsSurfacesReason(t *testing.T) {
	scenario, _ := eval.Load(writeScenario(t, `
name: judged-fail
turns:
  - user: "hello there"
    expect:
      - event: llm_response
        judge: "asks a clarifying question"
`))
	judge := eval.NewLLMJudge(&fakeGen{reply: "FAIL: it just echoes, no question"})
	res, err := eval.Host(context.Background(), scenario, buildFakeBot, eval.Options{Judge: judge})
	if err != nil {
		t.Fatal(err)
	}
	if res.Passed() {
		t.Fatal("expected the judge to fail the assertion")
	}
	if !strings.Contains(res.Failures[0].Reason, "no question") {
		t.Fatalf("failure should surface the judge's reason, got: %s", res.Failures[0].Reason)
	}
}
