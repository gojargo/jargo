package eval_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// errJudgeUnreachable stands in for a judge endpoint that cannot be reached.
var errJudgeUnreachable = errors.New("connection refused")

// fakeGen is a canned llm.Generator: it emits reply and counts its calls.
type fakeGen struct {
	reply string
	err   error
	calls atomic.Int32
}

func (g *fakeGen) Generate(_ context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.calls.Add(1)
	if g.err != nil {
		return g.err
	}
	return emit(g.reply)
}

// TestLLMJudgeVerdicts covers the verdict the judge reads out of each shape of
// model response: the JSON it is asked for, JSON wrapped in prose or fences, and
// the keyword fallback for a model that ignores the instruction entirely.
func TestLLMJudgeVerdicts(t *testing.T) {
	cases := map[string]struct {
		out         string
		wantVerdict string
		wantReason  string
	}{
		"plain json": {
			out:         `{"verdict": "yes", "reason": "greets the user warmly"}`,
			wantVerdict: eval.VerdictYes, wantReason: "greets the user warmly",
		},
		"no with reason": {
			out:         `{"verdict": "no", "reason": "too terse"}`,
			wantVerdict: eval.VerdictNo, wantReason: "too terse",
		},
		"continue": {
			out:         `{"verdict": "continue", "reason": "only filler so far"}`,
			wantVerdict: eval.VerdictContinue, wantReason: "only filler so far",
		},
		"code fenced": {
			out:         "```json\n{\"verdict\": \"yes\", \"reason\": \"fine\"}\n```",
			wantVerdict: eval.VerdictYes, wantReason: "fine",
		},
		"json wrapped in prose": {
			out: `Sure! {"verdict": "yes", "reason": "answers it"} ` +
				"Let me know if you'd like to evaluate further turns!",
			wantVerdict: eval.VerdictYes, wantReason: "answers it",
		},
		"unknown verdict word is a no": {
			out:         `{"verdict": "maybe", "reason": "unsure"}`,
			wantVerdict: eval.VerdictNo, wantReason: "unsure",
		},
		"missing reason": {
			out:         `{"verdict": "yes"}`,
			wantVerdict: eval.VerdictYes, wantReason: "(no reason given)",
		},
		"unstructured yes": {
			out:         "Yes, the reply satisfies it.",
			wantVerdict: eval.VerdictYes, wantReason: "(unstructured yes)",
		},
		"unstructured no": {
			out:         "That does not satisfy the criterion.",
			wantVerdict: eval.VerdictNo, wantReason: "(unstructured no)",
		},
		"unstructured continue": {
			out:         "I would continue and wait for more.",
			wantVerdict: eval.VerdictContinue, wantReason: "(unstructured continue)",
		},
		"no verdict word at all is a no": {
			out:         "The reply seems fine.",
			wantVerdict: eval.VerdictNo, wantReason: "could not parse judge response",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			j := eval.NewLLMJudge(&fakeGen{reply: tc.out})
			j.AddAssistantMessage("hi there")
			v := j.Evaluate(t.Context(), "greets warmly")
			if v.Verdict != tc.wantVerdict {
				t.Fatalf("verdict = %q, want %q (reason %q)", v.Verdict, tc.wantVerdict, v.Reason)
			}
			if !strings.Contains(v.Reason, tc.wantReason) {
				t.Fatalf("reason %q does not contain %q", v.Reason, tc.wantReason)
			}
		})
	}
}

// A judge that cannot answer reports a no with the reason, rather than failing
// the run: an unavailable judge is a failed assertion.
func TestLLMJudgeCallFailureIsANo(t *testing.T) {
	j := eval.NewLLMJudge(&fakeGen{err: errJudgeUnreachable})
	j.AddAssistantMessage("hi there")

	v := j.Evaluate(t.Context(), "greets warmly")
	if v.Verdict != eval.VerdictNo || !strings.Contains(v.Reason, "judge call failed") {
		t.Fatalf("unexpected verdict: %+v", v)
	}
}

func TestLLMJudgeEmptyResponseIsANo(t *testing.T) {
	j := eval.NewLLMJudge(&fakeGen{reply: ""})
	j.AddAssistantMessage("hi there")

	v := j.Evaluate(t.Context(), "greets warmly")
	if v.Verdict != eval.VerdictNo || !strings.Contains(v.Reason, "empty response") {
		t.Fatalf("unexpected verdict: %+v", v)
	}
}

// Verdicts are cached by (criterion, conversation), so a repeated assertion over
// an unchanged conversation costs one round trip, and a grown conversation costs
// another.
func TestLLMJudgeCaches(t *testing.T) {
	gen := &fakeGen{reply: `{"verdict": "yes", "reason": "ok"}`}
	j := eval.NewLLMJudge(gen)
	j.AddAssistantMessage("reply")

	for range 3 {
		j.Evaluate(t.Context(), "crit")
	}
	j.AddAssistantMessage("more reply") // the conversation changed: a cache miss
	j.Evaluate(t.Context(), "crit")

	if got := gen.calls.Load(); got != 2 {
		t.Fatalf("generator called %d times, want 2 (one per distinct conversation)", got)
	}
}

// An empty or blank message is not recorded, so it cannot invalidate the cache
// or reach the judge as a turn.
func TestLLMJudgeIgnoresBlankMessages(t *testing.T) {
	gen := &fakeGen{reply: `{"verdict": "yes", "reason": "ok"}`}
	j := eval.NewLLMJudge(gen)
	j.AddAssistantMessage("reply")
	j.Evaluate(t.Context(), "crit")

	j.AddUserMessage("   ")
	j.AddAssistantMessage("")
	j.Evaluate(t.Context(), "crit")

	if got := gen.calls.Load(); got != 1 {
		t.Fatalf("generator called %d times, want 1", got)
	}
}

func TestHarnessJudgePasses(t *testing.T) {
	// The bot echoes "you said: hello there"; the judge is told to answer yes.
	scenario, err := eval.Load(writeScenario(t, `
name: judged
turns:
  - user: "hello there"
    expect:
      - event: llm_response
        eval: "acknowledges the user's greeting"
`))
	if err != nil {
		t.Fatal(err)
	}
	judge := eval.NewLLMJudge(&fakeGen{reply: `{"verdict": "yes", "reason": "it echoes the greeting"}`})
	res, err := eval.Host(t.Context(), scenario, buildFakeBot, eval.Options{Judge: judge})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

func TestHarnessJudgeFailsSurfacesReason(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: judged-fail
turns:
  - user: "hello there"
    expect:
      - event: llm_response
        eval: "asks a clarifying question"
        within_ms: 2000
`))
	if err != nil {
		t.Fatal(err)
	}
	judge := eval.NewLLMJudge(&fakeGen{reply: `{"verdict": "no", "reason": "it just echoes, no question"}`})
	res, err := eval.Host(t.Context(), scenario, buildFakeBot, eval.Options{Judge: judge})
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

// A criterion with no judge configured is a failure naming what is missing,
// rather than a silent pass.
func TestHarnessJudgeMissing(t *testing.T) {
	res := host(t, `
name: judged-no-judge
turns:
  - user: "hello there"
    expect:
      - event: llm_response
        eval: "acknowledges the greeting"
`)
	if res.Passed() {
		t.Fatal("expected a failure for the criterion with no judge")
	}
	if !strings.Contains(res.Failures[0].Reason, "no judge could be built") {
		t.Fatalf("unexpected failure reason: %s", res.Failures[0].Reason)
	}
}

// The judge sees the user turn as well as the reply, so it can resolve a terse
// answer that would not make sense on its own.
func TestHarnessJudgeSeesTheConversation(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: judged-context
turns:
  - user: "what is two plus two?"
    expect:
      - event: llm_response
        eval: "answers with four"
`))
	if err != nil {
		t.Fatal(err)
	}
	judge := &recordingJudge{verdict: eval.JudgeVerdict{Verdict: eval.VerdictYes, Reason: "ok"}}
	if _, err := eval.Host(t.Context(), scenario, buildFakeBot, eval.Options{Judge: judge}); err != nil {
		t.Fatal(err)
	}
	want := []string{"user: what is two plus two?", "assistant: you said: what is two plus two?"}
	if strings.Join(judge.messages, "|") != strings.Join(want, "|") {
		t.Fatalf("judge saw %v, want %v", judge.messages, want)
	}
}

// recordingJudge records the conversation it is fed and answers with a canned
// verdict.
type recordingJudge struct {
	verdict  eval.JudgeVerdict
	messages []string
}

func (j *recordingJudge) AddUserMessage(text string) {
	j.messages = append(j.messages, "user: "+text)
}

func (j *recordingJudge) AddAssistantMessage(text string) {
	j.messages = append(j.messages, "assistant: "+text)
}

func (j *recordingJudge) Evaluate(context.Context, string) eval.JudgeVerdict { return j.verdict }
