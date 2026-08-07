package eval_test

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/dtmf"
	"github.com/gojargo/jargo/processor/rtvi"
)

// fakeLLM answers each turn deterministically so the harness can be exercised
// end-to-end without a real model: it echoes the user's text, and on a weather
// question it emits a get_weather tool call, followed by a get_restaurants call
// when the turn asks about food too.
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

	lower := strings.ToLower(last)
	// A turn asking how much it remembers reports the size of its context, which
	// is how a scenario checks that its context: was seeded.
	if strings.Contains(lower, "how many") {
		return f.respond(ctx, fmt.Sprintf("I have %d messages", len(msgs)))
	}
	// A turn that asks the bot to check something answers in two responses, an
	// interim filler and then the answer, which is what an aggregating
	// expectation has to roll past.
	if strings.Contains(lower, "check") {
		for _, text := range []string{"Let me check on that.", "It is sunny in Paris."} {
			if err := f.respond(ctx, text); err != nil {
				return err
			}
		}
		return nil
	}

	if err := f.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	switch {
	case strings.Contains(lower, "weather"):
		// The restaurant call goes first, so a scenario listing the two in the
		// other order only passes if the matcher really is order-independent.
		if strings.Contains(lower, "food") {
			args := json.RawMessage(`{"city":"Paris","party":2}`)
			call := frames.NewFunctionCallInProgressFrame("call-2", "get_restaurants", args, true, "g1")
			if err := f.PushFrame(ctx, call, processor.Downstream); err != nil {
				return err
			}
		}
		args := json.RawMessage(`{"city":"Paris","units":"celsius"}`)
		call := frames.NewFunctionCallInProgressFrame("call-1", "get_weather", args, true, "g1")
		if err := f.PushFrame(ctx, call, processor.Downstream); err != nil {
			return err
		}
	default:
		if err := f.PushFrame(ctx, frames.NewLLMTextFrame("you said: "+last), processor.Downstream); err != nil {
			return err
		}
	}
	return f.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// respond pushes one complete LLM response carrying text.
func (f *fakeLLM) respond(ctx context.Context, text string) error {
	if err := f.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	if err := f.PushFrame(ctx, frames.NewLLMTextFrame(text), processor.Downstream); err != nil {
		return err
	}
	return f.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

func buildFakeBot(in, out processor.Processor) *pipeline.Task {
	agg := aggregators.New(frames.NewLLMContext("test system"))
	rtviProc := rtvi.NewProcessor()
	return pipeline.NewTask(pipeline.New(
		in, agg.User(), newFakeLLM(), rtviProc, out, agg.Assistant(),
	), pipeline.TaskParams{
		// The observer reports pipeline events; the processor carries them.
		Observers: []pipeline.Observer{rtvi.NewObserver(rtviProc)},
	})
}

// buildDTMFBot is the fake bot with a DTMF aggregator in front of it, so a
// keypress turn produces the transcription a bot reacts to.
func buildDTMFBot(in, out processor.Processor) *pipeline.Task {
	agg := aggregators.New(frames.NewLLMContext("test system"))
	rtviProc := rtvi.NewProcessor()
	keys := dtmf.NewAggregator(dtmf.AggregatorConfig{Prefix: "DTMF: "})
	return pipeline.NewTask(pipeline.New(
		in, keys, agg.User(), newFakeLLM(), rtviProc, out, agg.Assistant(),
	), pipeline.TaskParams{
		Observers: []pipeline.Observer{rtvi.NewObserver(rtviProc)},
	})
}

func host(t *testing.T, body string) eval.Result {
	t.Helper()
	return hostWith(t, buildFakeBot, body)
}

func hostWith(t *testing.T, build eval.Bot, body string) eval.Result {
	t.Helper()
	scenario, err := eval.Load(writeScenario(t, body))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	res, err := eval.Host(ctx, scenario, build, eval.Options{})
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
        name: get_weather
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// args is a subset check: every listed argument must be present with that
// value, and the ones the model passed on top are ignored.
func TestHarnessFunctionCallArgs(t *testing.T) {
	res := host(t, `
name: weather-args
turns:
  - user: "what's the weather in Paris?"
    expect:
      - event: function_call
        name: get_weather
        args: {city: Paris}
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// The right tool called with the wrong arguments is the failure worth catching.
func TestHarnessReportsWrongArgs(t *testing.T) {
	res := host(t, `
name: wrong-city
turns:
  - user: "what's the weather in Paris?"
    expect:
      - event: function_call
        name: get_weather
        args: {city: Berlin}
        within_ms: 2000
`)
	if res.Passed() {
		t.Fatal("expected a failure for the tool call with the wrong city")
	}
	if !strings.Contains(res.Failures[0].Reason, "missing expected") {
		t.Fatalf("unexpected failure reason: %s", res.Failures[0].Reason)
	}
}

// A turn's calls are matched by name in any order, so a call arriving before
// the one the scenario lists first is held and claimed later.
func TestHarnessFunctionCallsAnyOrder(t *testing.T) {
	res := host(t, `
name: weather-and-food
turns:
  - user: "what's the weather in Paris, and where's the food?"
    expect:
      - event: function_call
        calls:
          - name: get_weather
            args: {units: celsius}
          - name: get_restaurants
            args: {party: 2}
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// An absent expectation passes when the event stays away for the whole window,
// and the other events the turn does produce must not trip it.
func TestHarnessAbsentPasses(t *testing.T) {
	res := host(t, `
name: answers-without-a-tool-call
turns:
  - user: "hello"
    expect:
      - event: function_call
        absent: true
        within_ms: 700
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// An absent expectation fails the moment the event it forbids arrives, and says
// what turned up, so a duplicate-output regression shows what the bot said.
func TestHarnessAbsentFailsWhenEventArrives(t *testing.T) {
	res := host(t, `
name: forbids-a-reply
turns:
  - user: "hello"
    expect:
      - event: llm_response
        absent: true
        within_ms: 2000
`)
	if res.Passed() {
		t.Fatal("expected a failure for the reply that must not arrive")
	}
	if !strings.Contains(res.Failures[0].Reason, "you said: hello") {
		t.Fatalf("unexpected failure reason: %s", res.Failures[0].Reason)
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
        within_ms: 2000
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
        name: get_weather
        within_ms: 800
`)
	if res.Passed() {
		t.Fatal("expected a timeout failure for the tool call that never happens")
	}
	if !strings.Contains(res.Failures[0].Reason, `function call "get_weather" not seen`) {
		t.Fatalf("unexpected failure reason: %s", res.Failures[0].Reason)
	}
}

// The first response is filler and the answer arrives in a second one. An
// aggregating text_contains accumulates across both and matches, rather than
// mistaking the filler for the turn's answer.
func TestHarnessAggregatesPastFiller(t *testing.T) {
	res := host(t, `
name: filler
turns:
  - user: "can you check the weather? and the food?"
    expect:
      - event: llm_response
        text_contains: "sunny in Paris"
        within_ms: 3000
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// A judge that says continue on the filler and yes on the answer passes, and is
// asked once per segment.
func TestHarnessAggregatesPastJudgeContinue(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: filler-judge
turns:
  - user: "can you check the weather?"
    expect:
      - event: llm_response
        eval: "reports the weather"
        within_ms: 3000
`))
	if err != nil {
		t.Fatal(err)
	}
	judge := &scriptedJudge{verdicts: []string{eval.VerdictContinue, eval.VerdictYes}}
	res, err := eval.Host(t.Context(), scenario, buildFakeBot, eval.Options{Judge: judge})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
	if judge.calls != 2 {
		t.Fatalf("judge asked %d times, want one per segment", judge.calls)
	}
}

// A judge that says no is an affirmative rejection: the expectation fails at
// once rather than waiting out the budget for more text.
func TestHarnessJudgeNoFailsWithoutWaiting(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: judge-no
turns:
  - user: "can you check the weather?"
    expect:
      - event: llm_response
        eval: "reports the weather"
        within_ms: 30000
`))
	if err != nil {
		t.Fatal(err)
	}
	judge := &scriptedJudge{verdicts: []string{eval.VerdictNo}}
	start := time.Now()
	res, err := eval.Host(t.Context(), scenario, buildFakeBot, eval.Options{Judge: judge})
	if err != nil {
		t.Fatal(err)
	}
	if res.Passed() {
		t.Fatal("expected the judge's no to fail the assertion")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("a no should fail at once, took %s", elapsed)
	}
}

// A turn that fails ends the scenario: a later turn would run against a
// conversation in an unknown state and cost another budget for nothing.
func TestHarnessFailFastStopsTheScenario(t *testing.T) {
	res := host(t, `
name: fail-fast
turns:
  - user: "hello"
    expect:
      - event: llm_response
        text_contains: "goodbye"
        within_ms: 700
  - user: "hello again"
    expect:
      - event: function_call
        name: get_weather
        within_ms: 700
`)
	if len(res.Failures) != 1 {
		t.Fatalf("want only the first turn's failure, got:\n%s", res)
	}
	if res.Failures[0].Turn != 1 {
		t.Fatalf("want the failure on turn 1, got %+v", res.Failures[0])
	}
}

// A turn missing both a call and a reply reports both within one shared budget,
// because a call that never arrives is a failure rather than the end of the turn.
func TestHarnessTurnSharesOneDeadline(t *testing.T) {
	start := time.Now()
	res := host(t, `
name: shared-deadline
turns:
  - user: "silence please"
    expect:
      - event: function_call
        name: get_weather
        within_ms: 700
      - event: llm_started
        within_ms: 700
`)
	if len(res.Failures) != 2 {
		t.Fatalf("want both expectations to report, got:\n%s", res)
	}
	// A budget each would be 1.4s, a shared one 0.7s, so the bound has to sit
	// between them to tell the two apart.
	if elapsed := time.Since(start); elapsed > 1200*time.Millisecond {
		t.Fatalf("one shared 700ms budget, not one each: took %s", elapsed)
	}
}

// scriptedJudge answers with the next queued verdict, and counts the asks.
type scriptedJudge struct {
	verdicts []string
	calls    int
}

func (j *scriptedJudge) AddUserMessage(string)      {}
func (j *scriptedJudge) AddAssistantMessage(string) {}

func (j *scriptedJudge) Evaluate(context.Context, string) eval.JudgeVerdict {
	j.calls++
	v := eval.VerdictNo
	if len(j.verdicts) > 0 {
		v, j.verdicts = j.verdicts[0], j.verdicts[1:]
	}
	return eval.JudgeVerdict{Verdict: v, Reason: "(" + v + ")"}
}

// A turn scheduled from an event waits for it, plus the delay, before sending.
func TestHarnessSendAfterDelaysTheTurn(t *testing.T) {
	start := time.Now()
	res := host(t, `
name: send-after
turns:
  - user: "hello"
    expect:
      - event: llm_started
        within_ms: 2000
  - user: "hello again"
    send_after: {event: llm_started, delay_ms: 400}
    expect:
      - event: llm_response
        text_contains: "hello again"
        within_ms: 2000
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
	if elapsed := time.Since(start); elapsed < 400*time.Millisecond {
		t.Fatalf("the second turn should have waited out its delay, took %s", elapsed)
	}
}

// A pure delay needs no event to anchor on.
func TestHarnessSendAfterPureDelay(t *testing.T) {
	start := time.Now()
	res := host(t, `
name: paced
turns:
  - user: "hello"
    expect:
      - event: llm_started
        within_ms: 2000
  - user: "hello again"
    send_after: {delay_ms: 400}
    expect:
      - event: llm_started
        within_ms: 2000
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
	if elapsed := time.Since(start); elapsed < 400*time.Millisecond {
		t.Fatalf("the second turn should have waited out its delay, took %s", elapsed)
	}
}

// A schedule that never fires fails the turn itself: the turn's input never
// went out, so none of its expectations can be judged, and the failure names the
// anchor rather than an expectation.
//
// The wait is bounded at 30s, which is too long to sit through here, so the run
// is cut short instead; what this checks is the shape of the failure. Upstream
// leaves the 30s bound itself untested too.
func TestHarnessSendAfterNeverFires(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: never-fires
turns:
  - user: "hello"
    send_after: {event: user_started_speaking, delay_ms: 0}
    expect:
      - event: llm_response
        within_ms: 500
`))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer cancel()

	res, err := eval.Host(ctx, scenario, buildFakeBot, eval.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Passed() {
		t.Fatal("expected a failure for the schedule that never fires")
	}
	f := res.Failures[0]
	if f.Expectation != 0 || f.Event != eval.EventUserStartedSpeaking {
		t.Fatalf("want a turn-level failure naming the anchor, got %+v", f)
	}
}

// A turn sent while the bot is still answering cuts it off, which is a barge-in
// the client typed rather than spoke. The scheduled turn is what makes it land
// mid-answer.
func TestHarnessBargeInInterruptsTheBot(t *testing.T) {
	res := host(t, `
name: barge-in
turns:
  - user: "can you check the weather?"
    expect:
      - event: llm_started
        within_ms: 2000
  - user: "actually, never mind"
    send_after: {event: llm_started, delay_ms: 50}
    expect:
      - event: bot_interrupted
        within_ms: 2000
`)
	if !res.Passed() {
		t.Fatalf("expected the barge-in to interrupt the bot, got:\n%s", res)
	}
}

// A keypad turn reaches the bot as real keypresses, which its DTMF aggregator
// turns into a transcription it can react to.
func TestHarnessDTMFTurn(t *testing.T) {
	res := hostWith(t, buildDTMFBot, `
name: keypad
turns:
  - dtmf: "123#"
    expect:
      - event: user_transcription
        text_contains: "DTMF: 123"
        within_ms: 2000
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// A scenario's context seeds the conversation the bot starts from, so a turn
// can refer back to something the scenario, not the bot, established.
func TestHarnessContextSeeded(t *testing.T) {
	res := host(t, `
name: seeded
context:
  - role: user
    text: "remember the number four"
  - role: assistant
    text: "four, noted"
turns:
  - user: "how many messages do you have?"
    expect:
      # Two seeded plus this turn. Without the seeding it would be one.
      - event: llm_response
        text_contains: "I have 3 messages"
        within_ms: 2000
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// A turn with no input only waits and asserts, which is how a bot-first
// scenario checks an opening greeting.
func TestHarnessObserveOnlyTurn(t *testing.T) {
	res := host(t, `
name: observe
turns:
  - user: "hello"
    expect:
      - event: llm_started
        within_ms: 2000
  - expect:
      - event: function_call
        absent: true
        within_ms: 500
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// A run reports what it saw and what it decided, so a failure can be read
// against what the bot actually did.
func TestHarnessReportsDiagnostics(t *testing.T) {
	res := host(t, `
name: diagnosed
turns:
  - user: "hello world"
    expect:
      - event: llm_response
        text_contains: "hello world"
        within_ms: 2000
`)
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
	if res.Duration <= 0 {
		t.Fatalf("the run should be timed, got %s", res.Duration)
	}
	var kinds []string
	for _, ev := range res.Events {
		kinds = append(kinds, ev.Kind)
	}
	for _, want := range []string{eval.EventLLMStarted, eval.EventLLMResponse} {
		if !slices.Contains(kinds, want) {
			t.Fatalf("events seen %v should include %q", kinds, want)
		}
	}
	trace := strings.Join(res.DebugLog, "\n")
	for _, want := range []string{`send: "hello world"`, "event: llm_response"} {
		if !strings.Contains(trace, want) {
			t.Fatalf("the trace should mention %q:\n%s", want, trace)
		}
	}
}

// A caller can watch a run as it happens rather than waiting for the result.
func TestHarnessReportsProgress(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: watched
turns:
  - user: "hello world"
    expect:
      - event: llm_response
        text_contains: "hello world"
        within_ms: 2000
`))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	opts := eval.Options{OnProgress: func(p eval.Progress) {
		got = append(got, p.Status+":"+p.Event)
	}}
	if _, err := eval.Host(t.Context(), scenario, buildFakeBot, opts); err != nil {
		t.Fatal(err)
	}
	want := []string{"turn:hello world", "matched:llm_response"}
	if !slices.Equal(got, want) {
		t.Fatalf("progress %v, want %v", got, want)
	}
}
