package eval_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/classifier"
	llmclassifier "github.com/gojargo/jargo/classifier/llm"
	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// replyOptions are the answers a reply is judged by.
var replyOptions = []string{"yes", "no", "continue"} //nolint:gochecknoglobals // fixed test data

// asked is one question the judge put to its classifier.
type asked struct {
	state     map[string]any
	questions []classifier.Named[classifier.Question]
}

// fakeClassifier answers with queued results, or with a function of the
// questions it is asked, and records every question, so a test can check what
// the judge asked and what it sent as the state. A queued error is returned
// instead of an answer.
type fakeClassifier struct {
	*classifier.Base

	mu      sync.Mutex
	answers []any
	answer  func([]classifier.Named[classifier.Question]) map[string]classifier.Result
	asked   []asked
}

func newFakeClassifier(answers ...any) *fakeClassifier {
	c := &fakeClassifier{answers: answers}
	c.Base = classifier.NewBase("FakeClassifier", "", c)
	return c
}

func (c *fakeClassifier) Model() string { return "" }

func (c *fakeClassifier) AskQuestions(
	_ context.Context, state any, questions []classifier.Named[classifier.Question],
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	encoded, err := json.Marshal(state)
	if err != nil {
		return nil, nil, err
	}
	var decoded map[string]any
	_ = json.Unmarshal(encoded, &decoded)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.asked = append(c.asked, asked{state: decoded, questions: questions})
	if c.answer != nil {
		return c.answer(questions), nil, nil
	}
	next := c.answers[0]
	c.answers = c.answers[1:]
	if err, ok := next.(error); ok {
		return nil, nil, err
	}
	results, _ := next.(map[string]classifier.Result)
	return results, nil, nil
}

func (c *fakeClassifier) questions() []asked {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.asked)
}

// choice is a reply's answer: choice at confidence, the rest shared out.
func choice(choice string, confidence float64, options ...string) map[string]classifier.Result {
	if len(options) == 0 {
		options = replyOptions
	}
	rest := (1 - confidence) / float64(len(options)-1)
	probabilities := map[string]float64{}
	for _, o := range options {
		probabilities[o] = rest
		if o == choice {
			probabilities[o] = confidence
		}
	}
	return map[string]classifier.Result{
		"verdict": classifier.ChoiceResult{Choice: choice, Probabilities: probabilities, Confidence: confidence},
	}
}

// yes is a call's answer: the probability of yes.
func yes(p float64) map[string]classifier.Result {
	return map[string]classifier.Result{"answer": classifier.YesNoResult{Probability: p}}
}

// answering is a classifier that always judges a reply choice at confidence.
func answering(verdict string, confidence float64) *fakeClassifier {
	c := newFakeClassifier()
	c.answer = func([]classifier.Named[classifier.Question]) map[string]classifier.Result {
		return choice(verdict, confidence)
	}
	return c
}

var errBoom = errors.New("boom") //nolint:gochecknoglobals // a classifier failure

func boom() error { return fmt.Errorf("%w: %w", classifier.ErrClassifier, errBoom) }

// call is one inference the fake LLM was asked for.
type call struct {
	ask         string
	instruction string
	maxTokens   int
}

// explainerLLM returns a queued answer to every inference, and records the call.
type explainerLLM struct {
	mu      sync.Mutex
	replies []string
	calls   []call
}

func (f *explainerLLM) RunInference(
	_ context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	msgs := convo.Messages()
	f.calls = append(f.calls, call{
		ask: msgs[len(msgs)-1].Text, instruction: opts.SystemInstruction, maxTokens: opts.MaxTokens,
	})
	if len(f.replies) == 0 {
		return "", errors.New("explainerLLM: no more queued responses") //nolint:err113 // a test double's own failure
	}
	reply := f.replies[0]
	f.replies = f.replies[1:]
	return reply, nil
}

func (f *explainerLLM) recorded() []call {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

// newJudge builds a judge over c, explained by explainer when it is not nil.
func newJudge(
	t *testing.T, c classifier.Classifier, explainer *explainerLLM, opts ...func(*eval.EvalJudgeConfig),
) *eval.EvalJudge {
	t.Helper()
	cfg := eval.EvalJudgeConfig{Classifier: c}
	if explainer != nil {
		cfg.Explainer = explainer
	}
	for _, o := range opts {
		o(&cfg)
	}
	j, err := eval.NewEvalJudge(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = j.Close(context.Background()) })
	return j
}

func noContinue(cfg *eval.EvalJudgeConfig) {
	off := false
	cfg.AllowContinue = &off
}

func optionNames(t *testing.T, q classifier.Question) []string {
	t.Helper()
	cq, ok := q.(classifier.ChoiceQuestion)
	if !ok {
		t.Fatalf("got a %T, want a choice question", q)
	}
	return cq.OptionNames()
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// Evaluate

func TestTheLatestReplyIsJudgedAfterTheSpokenConversation(t *testing.T) {
	c := newFakeClassifier(choice("yes", 0.95))
	j := newJudge(t, c, nil)
	j.AddUserMessage("What's the weather?")
	j.AddToolCall("lookup()")
	j.AddAssistantMessage("Let me check.")
	j.AddAssistantMessage("It's 72 and sunny.")

	v := j.Evaluate(t.Context(), "describes the weather")

	if !v.Passed() || v.Confidence == nil || *v.Confidence != 0.95 || !strings.Contains(v.Reason, "P(yes)=0.95") {
		t.Fatalf("verdict = %+v", v)
	}
	q := c.questions()[0]
	if q.state["latest_bot_reply"] != "Let me check. It's 72 and sunny." {
		t.Fatalf("latest reply = %v", q.state["latest_bot_reply"])
	}
	convo, _ := json.Marshal(q.state["conversation"])
	if string(convo) != `[{"speaker":"user","text":"What's the weather?"}]` {
		t.Fatalf("conversation = %s", convo)
	}
	if got := optionNames(t, q.questions[0].Question); !slices.Equal(got, replyOptions) {
		t.Fatalf("options = %v, want %v", got, replyOptions)
	}
	cq, _ := q.questions[0].Question.(classifier.ChoiceQuestion)
	if s, _ := cq.Instructions.(string); !strings.Contains(s, "describes the weather") {
		t.Fatalf("instructions = %v", cq.Instructions)
	}
}

func TestWithoutContinueAReplyIsJudgedYesOrNo(t *testing.T) {
	c := newFakeClassifier(choice("no", 0.95))
	j := newJudge(t, c, nil, noContinue)
	j.AddUserMessage("What is the capital of Germany?")
	j.AddAssistantMessage("No rush, take your time.")
	if v := j.Evaluate(t.Context(), "says the capital of Germany is Berlin"); v.Verdict != eval.VerdictNo {
		t.Fatalf("verdict = %+v", v)
	}
	if got := optionNames(t, c.questions()[0].questions[0].Question); !slices.Equal(got, []string{"yes", "no"}) {
		t.Fatalf("options = %v, want yes and no", got)
	}
}

func TestAVerdictIsCachedByCriterionAndConversation(t *testing.T) {
	c := newFakeClassifier(choice("yes", 0.95), choice("yes", 0.95))
	j := newJudge(t, c, nil)
	j.AddAssistantMessage("It rains.")
	j.Evaluate(t.Context(), "mentions weather")
	j.Evaluate(t.Context(), "mentions weather")
	if n := len(c.questions()); n != 1 {
		t.Fatalf("asked %d times, want once", n)
	}
	j.AddUserMessage("   ")
	j.AddAssistantMessage("")
	j.Evaluate(t.Context(), "mentions weather")
	if n := len(c.questions()); n != 1 {
		t.Fatalf("a blank message changed the conversation: asked %d times", n)
	}
	j.AddAssistantMessage("More rain.")
	j.Evaluate(t.Context(), "mentions weather")
	if n := len(c.questions()); n != 2 {
		t.Fatalf("a grown conversation was not asked again: asked %d times", n)
	}
}

func TestAFailedQuestionIsAskedAgainThenGivesANo(t *testing.T) {
	c := newFakeClassifier(boom(), choice("yes", 0.95))
	j := newJudge(t, c, nil)
	j.AddAssistantMessage("It rains.")
	if v := j.Evaluate(t.Context(), "mentions weather"); !v.Passed() || len(c.questions()) != 2 {
		t.Fatalf("verdict = %+v after %d questions", v, len(c.questions()))
	}

	c = newFakeClassifier(boom(), boom())
	j = newJudge(t, c, nil)
	j.AddAssistantMessage("anything")
	v := j.Evaluate(t.Context(), "anything")
	if v.Verdict != eval.VerdictNo || v.Reason != "judge call failed" || len(c.questions()) != 2 {
		t.Fatalf("verdict = %+v after %d questions", v, len(c.questions()))
	}
}

func TestAnAnswerOfTheWrongTypeFailsTheQuestion(t *testing.T) {
	c := newFakeClassifier(yesAs("verdict", 0.9), yesAs("verdict", 0.9))
	j := newJudge(t, c, nil)
	j.AddAssistantMessage("anything")
	if v := j.Evaluate(t.Context(), "anything"); v.Verdict != eval.VerdictNo || v.Reason != "judge call failed" {
		t.Fatalf("verdict = %+v", v)
	}
}

func yesAs(name string, p float64) map[string]classifier.Result {
	return map[string]classifier.Result{name: classifier.YesNoResult{Probability: p}}
}

// The explainer

func TestANoTakesTheExplainersReason(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "no", "reason": "it never mentions rain"}`}}
	j := newJudge(t, newFakeClassifier(choice("no", 0.95)), explainer)
	j.AddAssistantMessage("Hello.")
	v := j.Evaluate(t.Context(), "mentions weather")
	if v.Verdict != eval.VerdictNo || !strings.HasPrefix(v.Reason, "it never mentions rain") ||
		!strings.Contains(v.Reason, "P(no)=0.95") {
		t.Fatalf("verdict = %+v", v)
	}
}

func TestADisagreeingExplainerIsNotedAndTheClassifierStands(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "yes", "reason": "it says it rains"}`}}
	j := newJudge(t, newFakeClassifier(choice("no", 0.95)), explainer)
	j.AddAssistantMessage("It rains.")
	v := j.Evaluate(t.Context(), "mentions weather")
	if v.Verdict != eval.VerdictNo || !strings.Contains(v.Reason, "the explainer judged yes: it says it rains") {
		t.Fatalf("verdict = %+v", v)
	}
}

func TestAnAgreeingExplainerWithoutAReasonLeavesTheProbabilities(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "no"}`}}
	j := newJudge(t, newFakeClassifier(choice("no", 0.95)), explainer)
	j.AddAssistantMessage("Hello.")
	if v := j.Evaluate(t.Context(), "mentions weather"); !strings.HasPrefix(v.Reason, "P(yes)=") {
		t.Fatalf("verdict = %+v", v)
	}
}

func TestASureYesAndAContinueAreNotExplained(t *testing.T) {
	explainer := &explainerLLM{}
	j := newJudge(t, newFakeClassifier(choice("continue", 0.95), choice("yes", 0.95)), explainer)
	j.AddAssistantMessage("Let me check.")
	if v := j.Evaluate(t.Context(), "gives the weather"); v.Verdict != eval.VerdictContinue {
		t.Fatalf("verdict = %+v", v)
	}
	j.AddAssistantMessage("It's sunny.")
	if v := j.Evaluate(t.Context(), "gives the weather"); !v.Passed() {
		t.Fatalf("verdict = %+v", v)
	}
	if n := len(explainer.recorded()); n != 0 {
		t.Fatalf("the explainer was asked %d times, want never", n)
	}
}

func TestAnUnsureYesIsExplained(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "yes", "reason": "close enough"}`}}
	j := newJudge(t, newFakeClassifier(choice("yes", 0.6)), explainer)
	j.AddAssistantMessage("Sunny-ish.")
	v := j.Evaluate(t.Context(), "gives the weather")
	if !v.Passed() || !strings.HasPrefix(v.Reason, "close enough") || len(explainer.recorded()) != 1 {
		t.Fatalf("verdict = %+v, explainer asked %d times", v, len(explainer.recorded()))
	}
}

// EvaluateCall

func TestACallIsJudgedByNameAndArguments(t *testing.T) {
	c := newFakeClassifier(yes(0.2))
	j := newJudge(t, c, nil)
	j.AddUserMessage("Book six o'clock.")
	v := j.EvaluateCall(t.Context(), "book", map[string]any{"time": "7pm"}, "books six o'clock")
	if v.Verdict != eval.VerdictNo || v.Confidence == nil || !near(*v.Confidence, 0.8) {
		t.Fatalf("verdict = %+v", v)
	}
	got, _ := json.Marshal(c.questions()[0].state["call"])
	if string(got) != `{"arguments":{"time":"7pm"},"name":"book"}` {
		t.Fatalf("call = %s", got)
	}
}

func TestTheSameCriterionOnAnotherCallIsAnotherQuestion(t *testing.T) {
	c := newFakeClassifier(yes(0.9), yes(0.1))
	j := newJudge(t, c, nil)
	first := j.EvaluateCall(t.Context(), "submit", map[string]any{"n": 1}, "n is one")
	again := j.EvaluateCall(t.Context(), "submit", map[string]any{"n": 1}, "n is one")
	other := j.EvaluateCall(t.Context(), "submit", map[string]any{"n": 2}, "n is one")
	if !first.Passed() || !again.Passed() || other.Passed() {
		t.Fatalf("verdicts = %v, %v, %v; want yes, yes, no", first.Verdict, again.Verdict, other.Verdict)
	}
	if n := len(c.questions()); n != 2 {
		t.Fatalf("asked %d times, want twice", n)
	}
}

// Over an LLM classifier

func llmJudge(t *testing.T, llmReplies []string, explain bool) (*eval.EvalJudge, *explainerLLM) {
	t.Helper()
	service := &explainerLLM{replies: llmReplies}
	c, err := llmclassifier.New(llmclassifier.Config{LLM: service})
	if err != nil {
		t.Fatal(err)
	}
	var explainer *explainerLLM
	if explain {
		explainer = service
	}
	return newJudge(t, c, explainer), service
}

func TestAReplyIsClassifiedInOneCall(t *testing.T) {
	j, service := llmJudge(t, []string{`{"verdict": {"choice": "yes", "probabilities": {"yes": 0.95}}}`}, false)
	j.AddAssistantMessage("It's 72 and sunny.")
	if v := j.Evaluate(t.Context(), "describes the weather"); !v.Passed() {
		t.Fatalf("verdict = %+v", v)
	}
	calls := service.recorded()
	if len(calls) != 1 {
		t.Fatalf("made %d calls, want one", len(calls))
	}
	for _, want := range append([]string{"describes the weather", "It's 72 and sunny."},
		"- yes:", "- no:", "- continue:") {
		if !strings.Contains(calls[0].ask, want) {
			t.Errorf("ask lacks %q", want)
		}
	}
}

func TestANoIsExplainedByTheExplainersLLM(t *testing.T) {
	j, service := llmJudge(t, []string{
		`{"verdict": {"choice": "no", "probabilities": {"no": 0.9}}}`,
		`{"verdict": "no", "reason": "it never mentions the weather"}`,
	}, true)
	j.AddAssistantMessage("Hello there.")
	v := j.Evaluate(t.Context(), "describes the weather")
	if v.Verdict != eval.VerdictNo || !strings.HasPrefix(v.Reason, "it never mentions the weather") ||
		len(service.recorded()) != 2 {
		t.Fatalf("verdict = %+v after %d calls", v, len(service.recorded()))
	}
}

func TestWithoutAnExplainerTheProbabilitiesAreTheReason(t *testing.T) {
	j, service := llmJudge(t, []string{`{"verdict": {"choice": "no", "probabilities": {"no": 0.9}}}`}, false)
	j.AddAssistantMessage("Hello there.")
	v := j.Evaluate(t.Context(), "describes the weather")
	if v.Verdict != eval.VerdictNo || v.Reason != "P(yes)=0.00, P(no)=0.90, P(continue)=0.00" ||
		len(service.recorded()) != 1 {
		t.Fatalf("verdict = %+v after %d calls", v, len(service.recorded()))
	}
}

func TestALLMJudgeClassifiesAndExplainsWithOneService(t *testing.T) {
	service := &explainerLLM{replies: []string{
		`{"verdict": {"choice": "no", "probabilities": {"no": 0.9}}}`,
		`{"verdict": "no", "reason": "it never mentions the weather"}`,
	}}
	//nolint:staticcheck // the deprecated constructor is what is under test
	j := eval.NewLLMJudge(service)
	j.AddAssistantMessage("Hello there.")
	v := j.Evaluate(t.Context(), "describes the weather")
	if !strings.HasPrefix(v.Reason, "it never mentions the weather") || len(service.recorded()) != 2 {
		t.Fatalf("verdict = %+v after %d calls", v, len(service.recorded()))
	}
}

func TestAJudgeNeedsAClassifier(t *testing.T) {
	if _, err := eval.NewEvalJudge(eval.EvalJudgeConfig{}); err == nil {
		t.Fatal("a judge with no classifier was built")
	}
}

// What the explainer is asked

func TestAReplyIsExplainedOverTheConversation(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "no", "reason": "it never mentions rain"}`}}
	j := newJudge(t, newFakeClassifier(choice("no", 0.95)), explainer)
	j.AddUserMessage("What's the weather?")
	j.AddAssistantMessage("Hello there.")
	j.Evaluate(t.Context(), "describes the weather")
	c := explainer.recorded()[0]
	if !strings.Contains(c.ask, "Criterion: describes the weather") ||
		!strings.Contains(c.ask, "Answer yes, no, or continue.") || !strings.Contains(c.instruction, `"continue"`) {
		t.Fatalf("asked %+v", c)
	}
}

func TestWithoutContinueTheExplainerIsAskedYesOrNo(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "no", "reason": "never names Berlin"}`}}
	j := newJudge(t, newFakeClassifier(choice("no", 0.95, "yes", "no")), explainer, noContinue)
	j.AddAssistantMessage("Take your time.")
	j.Evaluate(t.Context(), "says the capital of Germany is Berlin")
	c := explainer.recorded()[0]
	if !strings.Contains(c.instruction, "The reply is the bot's final answer.") ||
		!strings.Contains(c.ask, "Answer yes or no.") {
		t.Fatalf("asked %+v", c)
	}
}

func TestACallIsExplainedByNameAndArguments(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "no", "reason": "wrong speaker"}`}}
	j := newJudge(t, newFakeClassifier(yes(0.1)), explainer)
	v := j.EvaluateCall(t.Context(), "submit", map[string]any{"speaker": "Ann"}, "submitted for Bob")
	if !strings.HasPrefix(v.Reason, "wrong speaker") {
		t.Fatalf("verdict = %+v", v)
	}
	c := explainer.recorded()[0]
	if !strings.Contains(c.ask, "called the function `submit` with arguments `{\"speaker\":\"Ann\"}`") ||
		!strings.Contains(c.instruction, "evaluating a function call") {
		t.Fatalf("asked %+v", c)
	}
}

func TestMaxTokensCapsTheExplainer(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "no", "reason": "no rain"}`}}
	j := newJudge(t, newFakeClassifier(choice("no", 0.95)), explainer,
		func(cfg *eval.EvalJudgeConfig) { cfg.MaxTokens = 64 })
	j.AddAssistantMessage("Hello.")
	j.Evaluate(t.Context(), "describes the weather")
	if got := explainer.recorded()[0].maxTokens; got != 64 {
		t.Fatalf("max tokens = %d, want 64", got)
	}
}

func TestTheExplainerDefaultsToAVerdictSizedBound(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"verdict": "no", "reason": "no rain"}`}}
	j := newJudge(t, newFakeClassifier(choice("no", 0.95)), explainer)
	j.AddAssistantMessage("Hello.")
	j.Evaluate(t.Context(), "describes the weather")
	if got := explainer.recorded()[0].maxTokens; got != 200 {
		t.Fatalf("max tokens = %d, want 200", got)
	}
}

func TestAFailedExplainerKeepsTheVerdict(t *testing.T) {
	j := newJudge(t, newFakeClassifier(choice("no", 0.95)), &explainerLLM{})
	j.AddAssistantMessage("Hello.")
	v := j.Evaluate(t.Context(), "describes the weather")
	if v.Verdict != eval.VerdictNo || !strings.Contains(v.Reason, "the explainer judged no: explainer call failed") &&
		!strings.Contains(v.Reason, "explainer call failed") {
		t.Fatalf("verdict = %+v", v)
	}
}

func TestPassedOnlyForYes(t *testing.T) {
	for verdict, want := range map[string]bool{"yes": true, "no": false, "continue": false} {
		if (eval.JudgeVerdict{Verdict: verdict}).Passed() != want {
			t.Errorf("Passed() for %q = %v, want %v", verdict, !want, want)
		}
	}
}

// The harness

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
	judge := newJudge(t, answering("yes", 0.95), &explainerLLM{replies: []string{}})
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
	judge := newJudge(t, answering("no", 0.95),
		&explainerLLM{replies: []string{`{"verdict": "no", "reason": "it just echoes, no question"}`}})
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

func (j *recordingJudge) EvaluateCall(
	context.Context, string, map[string]any, string,
) eval.JudgeVerdict {
	return j.verdict
}

// closingJudge records whether the run closed it.
type closingJudge struct {
	recordingJudge
	mu     sync.Mutex
	closed bool
}

func (j *closingJudge) Close(context.Context) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.closed = true
	return nil
}

func (j *closingJudge) wasClosed() bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.closed
}

func TestARunThatNeverReachesTheBotClosesItsJudge(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t,
		"name: x\nturns:\n  - user: hi\n    expect:\n      - event: llm_started\n"))
	if err != nil {
		t.Fatal(err)
	}
	judge := &closingJudge{}
	if _, err := eval.RunURL(t.Context(), scenario, "ws://127.0.0.1:1", judge); err == nil {
		t.Fatal("the run reached a bot that is not there")
	}
	if !judge.wasClosed() {
		t.Fatal("the judge was left open")
	}
}

func TestASkippedRunClosesItsJudge(t *testing.T) {
	m := suiteDir(t, "suite:\n  - bot_url: ws://127.0.0.1:1\n    scenarios: [missing.yaml]\n", 0)
	judge := &closingJudge{}
	results := eval.RunSuite(t.Context(), m, func() eval.Judge { return judge })
	if len(results) != 1 || results[0].Err == nil {
		t.Fatalf("got %+v, want the missing scenario reported", results)
	}
	if !judge.wasClosed() {
		t.Fatal("the judge was left open")
	}
}
