package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/llm"
)

func TestPersonaInstructionLLMAndContext(t *testing.T) {
	service := newStubPersonaLLM()
	persona := NewPersona("A curious traveler.", "Learn the capital of Germany.", service)
	for _, want := range []string{"A curious traveler.", "Learn the capital of Germany.", EndCallFunction} {
		if !strings.Contains(persona.Instruction(), want) {
			t.Fatalf("instruction lacks %q:\n%s", want, persona.Instruction())
		}
	}
	if persona.LLM() != PersonaLLM(service) {
		t.Fatal("the persona's LLM is not the one it was given")
	}
	if got := persona.Context().Messages(); len(got) != 0 {
		t.Fatalf("messages = %v, want none", got)
	}
	var names []string
	for _, tool := range persona.Context().Tools() {
		names = append(names, tool.Name)
	}
	if !slices.Equal(names, []string{EndCallFunction}) {
		t.Fatalf("tools = %v", names)
	}
}

// fakeConversationJudge answers the judge's one run question from a script and
// records what it saw. verdicts feed the goal in order; turnVerdicts are one
// mapping per bot turn, a metric left out of a mapping passing that turn.
type fakeConversationJudge struct {
	mu           sync.Mutex
	verdicts     []string
	turnVerdicts []map[string]string
	transcript   []map[string]any
	criteria     []string
	runCriteria  []Criterion
	closed       bool
	// evaluated, when set, runs as the run is evaluated.
	evaluated func()
}

func newFakeConversationJudge(verdicts []string, turnVerdicts ...map[string]string) *fakeConversationJudge {
	return &fakeConversationJudge{verdicts: verdicts, turnVerdicts: turnVerdicts}
}

func (j *fakeConversationJudge) add(role, text string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.transcript = append(j.transcript, map[string]any{"role": role, "content": text})
}

func (j *fakeConversationJudge) AddUserMessage(text string)      { j.add("user", text) }
func (j *fakeConversationJudge) AddAssistantMessage(text string) { j.add("assistant", text) }
func (j *fakeConversationJudge) AddToolCall(text string)         { j.add("tool", text) }

// Evaluate and EvaluateCall make it a Judge too, as a session's services hold
// it.
func (*fakeConversationJudge) Evaluate(context.Context, string, *bool) JudgeVerdict {
	return JudgeVerdict{Verdict: VerdictYes}
}

func (*fakeConversationJudge) EvaluateCall(context.Context, string, map[string]any, string) JudgeVerdict {
	return JudgeVerdict{Verdict: VerdictYes}
}

func (j *fakeConversationJudge) Close(context.Context) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.closed = true
	return nil
}

func (j *fakeConversationJudge) EvaluateRun(_ context.Context, criteria []Criterion, success string) RunVerdicts {
	if j.evaluated != nil {
		j.evaluated()
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	j.criteria = append(j.criteria, success)
	j.runCriteria = slices.Clone(criteria)
	turns := 0
	for _, e := range j.transcript {
		if e["role"] == "assistant" {
			turns++
		}
	}
	verdict := VerdictYes
	if len(j.verdicts) > 0 {
		verdict, j.verdicts = j.verdicts[0], j.verdicts[1:]
	}
	verdicts := RunVerdicts{
		Goal:  JudgeVerdict{Verdict: verdict, Reason: "because " + success},
		Turns: map[string][]JudgeVerdict{},
	}
	for _, c := range criteria {
		for i := range turns {
			scripted := VerdictYes
			if i < len(j.turnVerdicts) {
				if v, ok := j.turnVerdicts[i][c.Name]; ok {
					scripted = v
				}
			}
			verdicts.Turns[c.Name] = append(verdicts.Turns[c.Name],
				JudgeVerdict{Verdict: scripted, Reason: "because " + c.Criterion})
		}
	}
	return verdicts
}

func (j *fakeConversationJudge) seen() []map[string]any {
	j.mu.Lock()
	defer j.mu.Unlock()
	return slices.Clone(j.transcript)
}

// fakePersonaClient is the connection to the bot, as the driver uses it.
type fakePersonaClient struct {
	stream *EventStream
	mu     sync.Mutex
	hungUp bool
	// report, when set, takes the end_call event in place of the stream.
	report func(event map[string]any)
}

func (c *fakePersonaClient) HangUp() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.hungUp = true
}

func (c *fakePersonaClient) isHungUp() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hungUp
}

func (c *fakePersonaClient) ReportEndCall(event map[string]any) {
	if c.report != nil {
		c.report(event)
		return
	}
	c.stream.Append(event)
}

// testSimulation is a simulation with the fields the driver reads, overridden
// by each of edit.
func testSimulation(edit ...func(*SimulationScenario)) *SimulationScenario {
	s := &SimulationScenario{
		Name:         "capital",
		Persona:      "A traveler.",
		Goal:         "Learn the capital of Germany.",
		Success:      "the bot said Berlin",
		MaxTurns:     10,
		MaxDurationS: 5,
		MaxSilenceS:  DefaultMaxSilenceS,
	}
	for _, e := range edit {
		e(s)
	}
	return s
}

func withMetrics(metrics ...SimulationMetric) func(*SimulationScenario) {
	return func(s *SimulationScenario) { s.Metrics = metrics }
}

// simulationRun is a driver over fakes, and what it reports.
type simulationRun struct {
	driver  *SimulationDriver
	stream  *EventStream
	persona *stubPersonaLLM
	client  *fakePersonaClient
	judge   *fakeConversationJudge

	mu       sync.Mutex
	progress []SimulationProgress
}

func newSimulationRun(simulation *SimulationScenario, judge *fakeConversationJudge) *simulationRun {
	trace := NewTrace()
	stream := NewEventStream(simulation.BotAudio, trace)
	r := &simulationRun{
		stream: stream, persona: newStubPersonaLLM(), client: &fakePersonaClient{stream: stream}, judge: judge,
	}
	var runJudge RunJudge
	if judge != nil {
		runJudge = judge
	}
	r.driver = newSimulationDriver(simulation, NewPersona(simulation.Persona, simulation.Goal, r.persona),
		r.client, stream, runJudge, trace, func(p SimulationProgress) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.progress = append(r.progress, p)
		})
	return r
}

func (r *simulationRun) records() []SimulationProgress {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.progress)
}

// play runs the driver while conversation plays the other side, and returns
// the driver's failures.
func (r *simulationRun) play(t *testing.T, conversation func()) []AssertionFailure {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		if conversation != nil {
			conversation()
		}
	}()
	failures, err := r.driver.run(t.Context())
	<-done
	if err != nil {
		t.Fatal(err)
	}
	return failures
}

func (r *simulationRun) result(failures []AssertionFailure) *SimulationResult {
	return r.driver.result(failures, 10, r.stream.EventsSeen(), nil, "")
}

func (r *simulationRun) say(event map[string]any) { r.stream.Append(event) }

// endCall invokes the registered end_call the way the LLM service would, once
// the driver has registered it, and returns what it reported as the call's
// result.
func (r *simulationRun) endCall(
	t *testing.T, success bool, reason string,
) (string, *frames.FunctionCallResultProperties) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	handler := r.persona.handler(EndCallFunction)
	for handler == nil {
		if time.Now().After(deadline) {
			t.Error("end_call was never registered")
			return "", nil
		}
		time.Sleep(time.Millisecond)
		handler = r.persona.handler(EndCallFunction)
	}
	args, _ := json.Marshal(map[string]any{"success": success, "reason": reason})
	var (
		result string
		props  *frames.FunctionCallResultProperties
	)
	err := handler(context.Background(), llm.FunctionCallParams{
		FunctionName: EndCallFunction, ToolCallID: "c1", Arguments: args, Context: frames.NewLLMContext(""),
		Result: func(_ context.Context, res string, p *frames.FunctionCallResultProperties) error {
			result, props = res, p
			return nil
		},
	})
	if err != nil {
		t.Error(err)
	}
	return result, props
}

func scores(metrics []SimulationMetricScore) []any {
	var out []any
	for _, m := range metrics {
		if m.Score == nil {
			out = append(out, nil)
		} else {
			out = append(out, *m.Score)
		}
	}
	return out
}

func metricNamed(t *testing.T, res *SimulationResult, name string) SimulationMetricScore {
	t.Helper()
	for _, m := range res.Metrics {
		if m.Name == name {
			return m
		}
	}
	t.Fatalf("no metric %s in %+v", name, res.Metrics)
	return SimulationMetricScore{}
}

func TestSimulationDriverEndCallEndsTheRunAndTheJudgeSeesTheSwappedConversation(t *testing.T) {
	// The second bot turn is short but the judge finds it curt.
	judge := newFakeConversationJudge([]string{"yes"}, map[string]string{}, map[string]string{"brevity": "no"})
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{Name: "politeness", Criterion: "stayed polite", MinScore: new(1.0)},
		SimulationMetric{Name: "brevity", Criterion: "kept it short"},
	)), judge)

	failures := r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Hi! How can I help?"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "What is the capital of Germany?"})
		r.say(map[string]any{"type": "llm_response", "text": "Berlin."})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "Thanks!"})
		result, props := r.endCall(t, true, "I got my answer")
		var decoded map[string]any
		if err := json.Unmarshal([]byte(result), &decoded); err != nil ||
			!reflect.DeepEqual(decoded, map[string]any{"status": "call ended"}) {
			t.Errorf("result = %q", result)
		}
		if props == nil || props.RunLLM == nil || *props.RunLLM {
			t.Errorf("props = %+v, want no follow-up response", props)
		}
		// The bot's reply to the persona's last line closes the conversation.
		r.say(map[string]any{"type": "llm_response", "text": "You're welcome!"})
	})

	if len(failures) != 0 {
		t.Fatalf("failures = %v", failures)
	}
	if !r.client.isHungUp() {
		t.Fatal("the persona did not hang up")
	}
	// One judge call saw the whole conversation, the criteria, and the goal.
	wantTranscript := []map[string]any{
		{"role": "assistant", "content": "Hi! How can I help?"},
		{"role": "user", "content": "What is the capital of Germany?"},
		{"role": "assistant", "content": "Berlin."},
		{"role": "user", "content": "Thanks!"},
		{"role": "assistant", "content": "You're welcome!"},
	}
	if got := judge.seen(); !reflect.DeepEqual(got, wantTranscript) {
		t.Fatalf("transcript = %v", got)
	}
	wantCriteria := []Criterion{{"politeness", "stayed polite"}, {"brevity", "kept it short"}}
	if !reflect.DeepEqual(judge.runCriteria, wantCriteria) {
		t.Fatalf("criteria = %v", judge.runCriteria)
	}
	if !slices.Equal(judge.criteria, []string{"the bot said Berlin"}) {
		t.Fatalf("goal criteria = %v", judge.criteria)
	}

	res := r.result(nil)
	if !res.Succeeded || !res.Passed() {
		t.Fatalf("result = %+v", res)
	}
	if res.Reason != "because the bot said Berlin" || res.Success != "the bot said Berlin" {
		t.Fatalf("reason %q success %q", res.Reason, res.Success)
	}
	if res.EndedBy != SimulationEndedByEndCall || res.Turns != 2 {
		t.Fatalf("ended by %s after %d turns", res.EndedBy, res.Turns)
	}
	if want := map[string]any{"success": true, "reason": "I got my answer"}; !reflect.DeepEqual(res.EndCall, want) {
		t.Fatalf("end call = %v", res.EndCall)
	}
	if got := scores(res.Metrics); !reflect.DeepEqual(got, []any{1.0, 2.0 / 3}) {
		t.Fatalf("scores = %v", got)
	}
	// brevity scored below 1 but gates nothing, so the run still passes.
	if !res.Metrics[0].Passed || !res.Metrics[1].Passed {
		t.Fatalf("metrics = %+v", res.Metrics)
	}
	if res.Metrics[0].Reason != "all 3 turn(s)" || res.Metrics[0].Criterion != "stayed polite" {
		t.Fatalf("politeness = %+v", res.Metrics[0])
	}
	if res.Metrics[1].Reason != "turn 2: because kept it short" {
		t.Fatalf("brevity = %+v", res.Metrics[1])
	}
	var failedTurns []int
	for _, v := range res.Metrics[1].Verdicts {
		if !v.Passed {
			failedTurns = append(failedTurns, v.Turn)
		}
	}
	if !slices.Equal(failedTurns, []int{2}) {
		t.Fatalf("failed turns = %v", failedTurns)
	}
	if res.Failure() != "" {
		t.Fatalf("failure = %q", res.Failure())
	}
	if !reflect.DeepEqual(res.Messages, wantTranscript) {
		t.Fatalf("messages = %v", res.Messages)
	}
	if !slices.ContainsFunc(r.stream.EventsSeen(), func(e map[string]any) bool { return e["type"] == EventEndCall }) {
		t.Fatal("no end_call event")
	}
}

func TestSimulationDriverAMetricBelowItsMinScoreFailsTheRun(t *testing.T) {
	judge := newFakeConversationJudge([]string{"yes"}, map[string]string{"politeness": "no"})
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{Name: "politeness", Criterion: "stayed polite", MinScore: new(1.0)},
	)), judge)
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "What do you want."})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "The capital of Germany?"})
		r.endCall(t, true, "rude but answered")
		r.say(map[string]any{"type": "llm_response", "text": "Berlin. Anything else."})
	})
	res := r.result(nil)
	if !res.Succeeded || res.Passed() {
		t.Fatalf("result = %+v", res)
	}
	if want := "politeness 0.50 below 1.00: turn 1: because stayed polite"; res.Failure() != want {
		t.Fatalf("failure = %q", res.Failure())
	}
	var lines []SimulationProgress
	for _, p := range r.records() {
		if p.Status != "bot" {
			lines = append(lines, p)
		}
	}
	want := []SimulationProgress{{"user", "The capital of Germany?", 1}, {"ended", "end_call", 1}}
	if !reflect.DeepEqual(lines, want) {
		t.Fatalf("progress = %v", lines)
	}
}

func TestSimulationDriverMeasuresComeFromTheRunAndAValueOutOfRangeFailsIt(t *testing.T) {
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{Name: "latency", Measure: MeasureLatency, MaxValue: new(1.0)},
		SimulationMetric{Name: "words", Measure: MeasureWords, MaxValue: new(3.0)},
		SimulationMetric{Name: "turns", Measure: MeasureTurns, MaxValue: new(5.0)},
		SimulationMetric{Name: "duration", Measure: MeasureDuration, MinValue: new(0.0)},
	)), newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		// Times are seconds on the stream's clock; a reply's first token is
		// started_at.
		r.say(map[string]any{"type": "llm_response", "text": "Hi there", "at": 0.5, "started_at": 0.2})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "Capital of Germany?", "at": 1.0})
		r.say(map[string]any{"type": "bot_interrupted", "at": 1.1})
		r.say(map[string]any{"type": "llm_response", "text": "It is Berlin, of course.", "at": 3.0, "started_at": 2.4})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "Thanks", "at": 3.5})
		r.say(map[string]any{"type": "llm_response", "text": "Bye!", "at": 3.9, "started_at": 3.8})
		r.endCall(t, true, "done")
	})
	res := r.result(nil)
	// The slowest reply took 1.4 s, over the 1 s bound; the greeting had no
	// send before it.
	latency := metricNamed(t, res, "latency")
	if *latency.Value != 1.4 || latency.Passed {
		t.Fatalf("latency = %+v", latency)
	}
	if want := "slowest reply 1.40 s from the persona's send to the first token, at most 1"; latency.Reason != want {
		t.Fatalf("latency reason = %q", latency.Reason)
	}
	if words := metricNamed(t, res, "words"); *words.Value != 5 || words.Passed {
		t.Fatalf("words = %+v", words)
	}
	if turns := metricNamed(t, res, "turns"); *turns.Value != 2 || !turns.Passed {
		t.Fatalf("turns = %+v", turns)
	}
	if !metricNamed(t, res, "duration").Passed {
		t.Fatal("duration failed")
	}
	if !res.Succeeded || res.Passed() {
		t.Fatalf("result = %+v", res)
	}
	want := "latency: slowest reply 1.40 s from the persona's send to the first token, at most 1"
	if res.Failure() != want {
		t.Fatalf("failure = %q", res.Failure())
	}
}

func TestSimulationDriverARequiredMeasurementWithoutAValueFailsASuccessfulGoal(t *testing.T) {
	for _, measure := range []string{MeasureLatency, MeasureWords} {
		for _, bounds := range [][2]*float64{{new(0.0), nil}, {nil, new(3.0)}, {new(0.0), new(3.0)}} {
			r := newSimulationRun(testSimulation(withMetrics(
				SimulationMetric{Name: measure, Measure: measure, MinValue: bounds[0], MaxValue: bounds[1]},
			)), newFakeConversationJudge([]string{"yes"}))
			failures := r.play(t, func() { r.endCall(t, true, "done") })
			res := r.result(failures)
			metric := res.Metrics[0]
			if !res.Succeeded || res.Error != "" {
				t.Fatalf("%s: result = %+v", measure, res)
			}
			if metric.Value != nil || metric.Score != nil || metric.Passed || metric.FailureKind != FailureError {
				t.Fatalf("%s: metric = %+v", measure, metric)
			}
			if res.Passed() || !strings.Contains(res.Failure(), "no reply") {
				t.Fatalf("%s: failure = %q", measure, res.Failure())
			}
		}
	}
}

func TestSimulationDriverAMeasuredZeroCanPassItsBounds(t *testing.T) {
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{Name: "turns", Measure: MeasureTurns, MaxValue: new(0.0)},
	)), newFakeConversationJudge([]string{"yes"}))
	failures := r.play(t, func() { r.endCall(t, true, "done") })
	res := r.result(failures)
	metric := res.Metrics[0]
	if *metric.Value != 0 || *metric.Score != 1 || !metric.Passed || metric.FailureKind != "" {
		t.Fatalf("metric = %+v", metric)
	}
	if !res.Passed() {
		t.Fatalf("result = %+v", res)
	}
}

func TestSimulationDriverWordsMeasureCountsEachCJKCharacter(t *testing.T) {
	chinese := "这是一个比较长的中文回复用来测量字数"
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{Name: "words", Measure: MeasureWords, MaxValue: new(10.0)},
	)), newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": chinese})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "好"})
		r.say(map[string]any{"type": "llm_response", "text": "Berlin."})
		r.endCall(t, true, "done")
	})
	words := r.result(nil).Metrics[0]
	if *words.Value != float64(utf8.RuneCountInString(chinese)) || words.Passed ||
		!strings.Contains(words.Reason, "longest reply") {
		t.Fatalf("words = %+v", words)
	}
}

func TestSimulationDriverFunctionCallsAreCheckedAgainstTheListTheBotShouldMake(t *testing.T) {
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{
			Name: "booking", Measure: MeasureFunctionCalls,
			Calls: []FunctionCall{{Name: "book_table", Args: map[string]any{"party_size": 2}}},
		},
		SimulationMetric{Name: "hands_off", Measure: MeasureFunctionCalls, Calls: []FunctionCall{}},
		SimulationMetric{
			Name: "lookup", Measure: MeasureFunctionCalls,
			Calls: []FunctionCall{{Name: "check_availability"}, {Name: "book_table"}},
		},
	)), newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Hello!"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "A table for two at six."})
		r.say(map[string]any{"type": "function_call", "name": "check_availability", "args": map[string]any{"time": "6"}})
		// A canceled call did not happen.
		r.say(map[string]any{"type": "function_call", "name": "send_confirmation", "args": map[string]any{}})
		r.say(map[string]any{
			"type": "function_call_stopped", "name": "send_confirmation",
			"args": map[string]any{"tool_call_id": "1", keyCanceled: true},
		})
		r.say(map[string]any{
			"type": "function_call", "name": "book_table", "args": map[string]any{"party_size": 2, "time": "6"},
		})
		r.say(map[string]any{"type": "llm_response", "text": "Booked."})
		r.endCall(t, true, "booked")
	})
	res := r.result(nil)
	made := "check_availability(time='6'), book_table(party_size=2, time='6')"
	// The list is the whole set: the lookup is not on the booking list, so it
	// fails.
	booking := metricNamed(t, res, "booking")
	if *booking.Value != 2 || booking.Passed || booking.Reason != made+", expected book_table(party_size=2)" {
		t.Fatalf("booking = %+v", booking)
	}
	if handsOff := metricNamed(t, res, "hands_off"); handsOff.Passed || handsOff.Reason != made+", expected none" {
		t.Fatalf("hands_off = %+v", handsOff)
	}
	// Names alone match, in any order, and the canceled call is not counted.
	lookup := metricNamed(t, res, "lookup")
	if !lookup.Passed || lookup.Reason != made+", expected check_availability, book_table" {
		t.Fatalf("lookup = %+v", lookup)
	}
	if want := "booking: " + made + ", expected book_table(party_size=2)"; res.Failure() != want {
		t.Fatalf("failure = %q", res.Failure())
	}
}

func TestSimulationDriverABotThatShouldCallNothingPassesWhenItCallsNothing(t *testing.T) {
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{Name: "hands_off", Measure: MeasureFunctionCalls, Calls: []FunctionCall{}},
	)), newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "I can't do that for you."})
		r.endCall(t, true, "turned down")
	})
	res := r.result(nil)
	handsOff := res.Metrics[0]
	if *handsOff.Value != 0 || !handsOff.Passed || handsOff.Reason != "no calls, expected none" {
		t.Fatalf("hands_off = %+v", handsOff)
	}
	if !res.Passed() {
		t.Fatalf("result = %+v", res)
	}
}

func TestSimulationDriverABotTurnIsWhatItSaidBetweenPersonaTurnsWithTheCallsByThen(t *testing.T) {
	judge := newFakeConversationJudge([]string{"yes"})
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{Name: "honesty", Criterion: "claims only what a call backs"},
	)), judge)
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Hello!"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "A table at six, please."})
		// A function call splits the reply in two; both halves are one turn,
		// and the call is that turn's evidence, not the greeting's.
		r.say(map[string]any{"type": "llm_response", "text": "Let me check."})
		r.say(map[string]any{"type": "function_call", "name": "check_availability", "args": map[string]any{"time": "6"}})
		r.say(map[string]any{"type": "llm_response", "text": "Six is free, booked."})
		r.say(map[string]any{"type": EventPersonaTurn, "text": ""})
		r.endCall(t, true, "booked")
	})
	// The call sits in the transcript where it happened, before the turn it
	// split.
	wantTranscript := []map[string]any{
		{"role": "assistant", "content": "Hello!"},
		{"role": "user", "content": "A table at six, please."},
		{"role": "tool", "content": `check_availability({"time": "6"})`},
		{"role": "assistant", "content": "Let me check. Six is free, booked."},
	}
	if got := judge.seen(); !reflect.DeepEqual(got, wantTranscript) {
		t.Fatalf("transcript = %v", got)
	}
	res := r.result(nil)
	wantMessages := []map[string]any{
		{"role": "assistant", "content": "Hello!"},
		{"role": "user", "content": "A table at six, please."},
		{"role": "assistant", "content": "Let me check. Six is free, booked."},
	}
	if !reflect.DeepEqual(res.Messages, wantMessages) {
		t.Fatalf("messages = %v", res.Messages)
	}
	// The conversation file has the call in place too.
	wantConversation := []map[string]any{
		{"role": "assistant", "content": "Hello!"},
		{"role": "user", "content": "A table at six, please."},
		{"role": "tool_call", "name": "check_availability", "arguments": map[string]any{"time": "6"}, keyCanceled: false},
		{"role": "assistant", "content": "Let me check. Six is free, booked."},
	}
	if !reflect.DeepEqual(res.Conversation, wantConversation) {
		t.Fatalf("conversation = %v", res.Conversation)
	}
}

func TestSimulationDriverTheConversationIsReportedAsItHappens(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Hi! How can I help?"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "What is the capital?"})
		r.say(map[string]any{"type": "llm_response", "text": "Berlin."})
		// A response without text (a function call's own) is not a line.
		r.say(map[string]any{"type": "llm_response", "text": ""})
		r.endCall(t, true, "done")
	})
	want := []SimulationProgress{
		{"bot", "Hi! How can I help?", 0},
		{"user", "What is the capital?", 1},
		{"bot", "Berlin.", 1},
		{"ended", "end_call", 1},
	}
	if got := r.records(); !reflect.DeepEqual(got, want) {
		t.Fatalf("progress = %v", got)
	}
}

func TestSimulationDriverBotTurnCapEndsTheRun(t *testing.T) {
	r := newSimulationRun(testSimulation(func(s *SimulationScenario) { s.MaxTurns = 2 }),
		newFakeConversationJudge([]string{"no"}))
	r.play(t, func() {
		for _, text := range []string{"one", "two", "three"} {
			r.say(map[string]any{"type": "llm_response", "text": text})
			r.say(map[string]any{"type": EventPersonaTurn})
		}
	})
	res := r.result(nil)
	if res.EndedBy != SimulationEndedByMaxTurns || res.Turns != 2 || res.Succeeded {
		t.Fatalf("result = %+v", res)
	}
}

func TestSimulationDriverTheBotHangingUpEndsTheRun(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Bye!"})
		r.say(map[string]any{"type": EventBotEnded})
	})
	if res := r.result(nil); res.EndedBy != SimulationEndedByBot {
		t.Fatalf("ended by %s", res.EndedBy)
	}
	if !r.client.isHungUp() {
		t.Fatal("the persona did not hang up")
	}
}

func TestSimulationDriverTheBotsLastWordsBeforeHangingUpAreJudged(t *testing.T) {
	r := newSimulationRun(testSimulation(func(s *SimulationScenario) { s.BotAudio = true }),
		newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.stream.BotTurnStarted()
		// The bot hangs up as it stops speaking, before the harness has closed
		// its last words into a turn.
		r.say(map[string]any{"type": EventBotEnded})
		time.Sleep(100 * time.Millisecond)
		r.stream.BotTurnStopped("Have a wonderful day!")
	})
	res := r.result(nil)
	if res.EndedBy != SimulationEndedByBot {
		t.Fatalf("ended by %s", res.EndedBy)
	}
	if last := res.Messages[len(res.Messages)-1]; !reflect.DeepEqual(last,
		map[string]any{"role": "assistant", "content": "Have a wonderful day!"}) {
		t.Fatalf("last line = %v", last)
	}
	// The live conversation shows them too, before the run's end.
	records := r.records()
	if len(records) < 2 {
		t.Fatalf("records = %+v", records)
	}
	said, ended := records[len(records)-2], records[len(records)-1]
	if said.Status != "bot" || said.Text != "Have a wonderful day!" || ended.Status != "ended" ||
		ended.Text != SimulationEndedByBot {
		t.Fatalf("last records = %+v, %+v", said, ended)
	}
}

func TestSimulationDriverTheBotsToolCallsAreTheJudgesEvidence(t *testing.T) {
	judge := newFakeConversationJudge([]string{"yes"})
	r := newSimulationRun(testSimulation(), judge)
	r.play(t, func() {
		r.say(map[string]any{
			"type": "function_call", "name": "check_availability", "args": map[string]any{"time": "6:00 PM"},
		})
		r.say(map[string]any{
			"type": "function_call_stopped", "name": "check_availability", "args": map[string]any{keyCanceled: false},
		})
		r.say(map[string]any{"type": "function_call", "name": "end_conversation", "args": map[string]any{}})
		r.say(map[string]any{
			"type": "function_call_stopped", "name": "end_conversation", "args": map[string]any{keyCanceled: true},
		})
		r.endCall(t, true, "booked")
	})
	calls := []string{
		`check_availability({"time": "6:00 PM"})`, "end_conversation()", "end_conversation" + callCanceledSuffix,
	}
	var told []string
	for _, e := range judge.seen() {
		if e["role"] == "tool" {
			content, _ := e["content"].(string)
			told = append(told, content)
		}
	}
	if !slices.Equal(told, calls) {
		t.Fatalf("tool lines = %v", told)
	}
	var transcribed []string
	for _, e := range r.driver.Transcript() { //nolint:staticcheck // the deprecated view still works
		if e["role"] == "tool" {
			content, _ := e["content"].(string)
			transcribed = append(transcribed, content)
		}
	}
	if !slices.Equal(transcribed, calls) {
		t.Fatalf("transcript tool lines = %v", transcribed)
	}
	if !slices.Equal(r.driver.ToolCalls(), calls) {
		t.Fatalf("tool calls = %v", r.driver.ToolCalls())
	}
	var got []string
	for _, m := range r.result(nil).Conversation {
		name, _ := m["name"].(string)
		canceled, _ := m[keyCanceled].(bool)
		got = append(got, fmt.Sprintf("%s %v", name, canceled))
	}
	if want := []string{"check_availability false", "end_conversation true"}; !slices.Equal(got, want) {
		t.Fatalf("conversation = %v", got)
	}
}

func TestSimulationDriverWallClockCapEndsTheRun(t *testing.T) {
	r := newSimulationRun(testSimulation(func(s *SimulationScenario) { s.MaxDurationS = 0.05 }),
		newFakeConversationJudge([]string{"no"}))
	r.play(t, nil)
	if res := r.result(nil); res.EndedBy != SimulationEndedByMaxDuration {
		t.Fatalf("ended by %s", res.EndedBy)
	}
}

func TestSimulationDriverOnlyPersonaTurnsCount(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "the bot's turn"})
		r.say(map[string]any{"type": "response", "text": "its transcription"})
		r.say(map[string]any{"type": EventPersonaTurn})
		r.endCall(t, true, "done")
	})
	if res := r.result(nil); res.Turns != 1 {
		t.Fatalf("turns = %d", res.Turns)
	}
}

func TestSimulationDriverARunLevelFailureIsAnErrorNotAGoalFailure(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge(nil))
	failure := AssertionFailure{
		TurnIndex: -1, ExpectationIndex: -1, EventName: RunEventConnect, Reason: "refused", Kind: FailureConnectFailed,
	}
	res := r.result([]AssertionFailure{failure})
	if res.Error != "refused" || res.Succeeded || res.Passed() || res.EndedBy != SimulationEndedByError ||
		res.Reason != "refused" || res.FailureKind != FailureConnectFailed {
		t.Fatalf("result = %+v", res)
	}
}

// Latency is timed from the caller's side by its modality, to the bot's side by
// its own.
func TestReplyLatencyEachEndIsTimedByItsOwnSidesModality(t *testing.T) {
	// In the first three cases the reply follows an LLM response that only
	// calls a function, which has no text and is not the reply.
	for _, tc := range []struct {
		userAudio, botAudio bool
		events              []map[string]any
		value               float64
		reason              string
	}{
		{false, false, []map[string]any{
			{"type": EventPersonaTurn, "text": "A table for two.", "at": 1.0},
			{"type": "llm_response", "text": "", "at": 1.5},
			{"type": "llm_response", "text": "Done.", "at": 3.0, "started_at": 2.4},
		}, 1.4, "slowest reply 1.40 s from the persona's send to the first token"},
		{true, false, []map[string]any{
			// The persona's line is sent before its audio has been heard.
			{"type": EventPersonaTurn, "text": "A table for two.", "at": 1.0},
			{"type": "user_started_speaking", "at": 1.2},
			{"type": "user_stopped_speaking", "at": 3.0},
			{"type": "llm_response", "text": "", "at": 3.4},
			{"type": "llm_response", "text": "Done.", "at": 4.5, "started_at": 3.9},
		}, 0.9, "slowest reply 0.90 s from the persona stopping to the first token"},
		{false, true, []map[string]any{
			// Text input never makes the bot report the persona stopping.
			{"type": EventPersonaTurn, "text": "A table for two.", "at": 1.0},
			{"type": "llm_response", "text": "", "at": 1.5},
			{"type": "bot_started_speaking", "at": 2.2},
			{"type": "tts_response", "text": "Done.", "at": 2.3},
		}, 1.2, "slowest reply 1.20 s from the persona's send to the bot starting to speak"},
		{true, true, []map[string]any{
			{"type": EventPersonaTurn, "text": "A table for two.", "at": 1.0},
			{"type": "user_stopped_speaking", "at": 3.0},
			// The bot's report of its first spoken word lags its audio.
			{"type": "bot_started_speaking", "at": 3.8},
			{"type": "tts_response", "text": "Done.", "at": 3.9},
		}, 0.8, "slowest reply 0.80 s from the persona stopping to the bot starting to speak"},
	} {
		r := newSimulationRun(testSimulation(func(s *SimulationScenario) {
			s.UserAudio, s.BotAudio = tc.userAudio, tc.botAudio
		}), newFakeConversationJudge([]string{"yes"}))
		r.say(map[string]any{"type": "llm_response", "text": "Hi.", "at": 0.5})
		for _, e := range tc.events {
			r.say(e)
		}
		value, reason := r.driver.measurement(MeasureLatency)
		if value == nil || *value != tc.value || reason != tc.reason {
			t.Fatalf("user audio %v, bot audio %v: %v %q", tc.userAudio, tc.botAudio, value, reason)
		}
	}
}

func TestReplyLatencyASpokenTurnTheBotEndsTwiceIsTimedFromTheLaterEnd(t *testing.T) {
	// A pause mid-line makes the bot end the persona's turn once too early.
	r := newSimulationRun(testSimulation(func(s *SimulationScenario) { s.UserAudio = true }),
		newFakeConversationJudge([]string{"yes"}))
	for _, e := range []map[string]any{
		{"type": EventPersonaTurn, "text": "A table for two, at seven.", "at": 1.0},
		{"type": "user_stopped_speaking", "at": 2.0},
		{"type": "user_started_speaking", "at": 2.2},
		{"type": "user_stopped_speaking", "at": 3.0},
		{"type": "llm_response", "text": "Done.", "at": 4.0, "started_at": 3.5},
	} {
		r.say(e)
	}
	value, reason := r.driver.measurement(MeasureLatency)
	if value == nil || *value != 0.5 || reason != "slowest reply 0.50 s from the persona stopping to the first token" {
		t.Fatalf("%v %q", value, reason)
	}
}

// runEndingWithAnUnansweredTurn is a reply in 0.05 s, then a turn the bot
// answers only with a function call.
func runEndingWithAnUnansweredTurn(t *testing.T, botAudio, hangUp bool) (*SimulationResult, SimulationMetricScore) {
	t.Helper()
	r := newSimulationRun(testSimulation(func(s *SimulationScenario) {
		s.BotAudio, s.MaxSilenceS = botAudio, 0.4
	}, withMetrics(SimulationMetric{Name: "latency", Measure: MeasureLatency, MaxValue: new(0.3)})),
		newFakeConversationJudge([]string{"yes"}))
	reply := map[string]any{"type": "llm_response", "text": "Sure."}
	if botAudio {
		reply = map[string]any{"type": "bot_started_speaking"}
	}
	r.play(t, func() {
		r.say(map[string]any{"type": EventPersonaTurn, "text": "A table for two?"})
		time.Sleep(50 * time.Millisecond)
		r.say(reply)
		r.say(map[string]any{"type": EventPersonaTurn, "text": "At seven."})
		r.say(map[string]any{"type": "llm_response", "text": ""})
		if hangUp {
			r.endCall(t, true, "done")
		}
	})
	res := r.result(nil)
	return res, metricNamed(t, res, "latency")
}

func TestReplyLatencyATurnUnansweredWhenTheRunFallsSilentCountsItsWait(t *testing.T) {
	for _, botAudio := range []bool{false, true} {
		res, latency := runEndingWithAnUnansweredTurn(t, botAudio, false)
		if res.EndedBy != SimulationEndedBySilence {
			t.Fatalf("bot audio %v: ended by %s", botAudio, res.EndedBy)
		}
		// The caller waited out the silence window, well over the bound.
		if *latency.Value < 0.4 || latency.Passed ||
			!strings.Contains(latency.Reason, "the last turn unanswered when the run ended by silence") {
			t.Fatalf("bot audio %v: latency = %+v", botAudio, latency)
		}
		if res.Passed() {
			t.Fatalf("bot audio %v: passed", botAudio)
		}
	}
}

func TestReplyLatencyATurnLeftWhenThePersonaHangsUpIsNotCounted(t *testing.T) {
	res, latency := runEndingWithAnUnansweredTurn(t, false, true)
	if res.EndedBy != SimulationEndedByEndCall || *latency.Value >= 0.3 || !latency.Passed {
		t.Fatalf("ended by %s, latency = %+v", res.EndedBy, latency)
	}
}

// A persona that hangs up on its own last line leaves the bot its reply to it.

// hangUpWhileSpeaking runs a conversation whose last persona line ends with
// end_call, made while the line is spoken.
func hangUpWhileSpeaking(t *testing.T, simulation *SimulationScenario) (*SimulationResult, *fakeConversationJudge) {
	t.Helper()
	// In audio mode the persona calls end_call before the user TTS has spoken
	// the line the call came with.
	judge := newFakeConversationJudge([]string{"yes"})
	r := newSimulationRun(simulation, judge)
	relay := newPersonaTurnRelay(func(string) rtvi.Message { return rtvi.Message{} }, NewTrace(), r.stream)
	if err := relay.Setup(t.Context(), processor.Setup{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = relay.Cleanup(context.Background()) })
	relay.Link(newCollector())
	r.client.report = relay.reportEndCall
	push := func(fs ...frames.Frame) {
		for _, f := range fs {
			if err := relay.ProcessFrame(t.Context(), f, processor.Downstream); err != nil {
				t.Error(err)
			}
		}
	}
	r.play(t, func() {
		r.say(map[string]any{"type": "response", "text": "Anything else?"})
		push(frames.NewLLMFullResponseStartFrame())
		r.endCall(t, true, "done")
		if !r.client.isHungUp() {
			t.Error("the persona did not hang up")
		}
		// Speaking the line takes a while.
		time.Sleep(100 * time.Millisecond)
		push(frames.NewTTSTextFrame("No, goodbye!", frames.TextTypeSentence), frames.NewLLMFullResponseEndFrame())
		time.Sleep(100 * time.Millisecond)
		r.say(map[string]any{"type": "response", "text": "Goodbye!"})
	})
	return r.result(nil), judge
}

func TestClosingTurnTheBotsReplyToTheLastLineClosesTheRun(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Is that all correct?"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "Yes, all correct."})
		r.endCall(t, true, "done")
		time.Sleep(100 * time.Millisecond)
		// The persona is silent by now, and the bot's reply still counts.
		if !r.client.isHungUp() {
			t.Error("the persona did not hang up")
		}
		r.say(map[string]any{"type": "llm_response", "text": "Great, you're all set."})
	})
	res := r.result(nil)
	if res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("ended by %s", res.EndedBy)
	}
	if last := res.Messages[len(res.Messages)-1]; !reflect.DeepEqual(last,
		map[string]any{"role": "assistant", "content": "Great, you're all set."}) {
		t.Fatalf("last line = %v", last)
	}
}

func TestClosingTurnAHangUpWhileTheLastLineIsSpokenWaitsForTheBotsReply(t *testing.T) {
	res, judge := hangUpWhileSpeaking(t, testSimulation(func(s *SimulationScenario) { s.BotAudio = true }))
	if res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("ended by %s", res.EndedBy)
	}
	// The judge read the persona's last line and the bot's reply to it.
	seen := judge.seen()
	want := []map[string]any{{"role": "user", "content": "No, goodbye!"}, {"role": "assistant", "content": "Goodbye!"}}
	if len(seen) < 2 || !reflect.DeepEqual(seen[len(seen)-2:], want) {
		t.Fatalf("transcript = %v", seen)
	}
}

func TestClosingTurnAHangUpOnTheLastAllowedTurnWaitsForTheBotsReply(t *testing.T) {
	res, _ := hangUpWhileSpeaking(t, testSimulation(func(s *SimulationScenario) { s.BotAudio, s.MaxTurns = true, 1 }))
	if res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("ended by %s", res.EndedBy)
	}
	if last := res.Messages[len(res.Messages)-1]; !reflect.DeepEqual(last,
		map[string]any{"role": "assistant", "content": "Goodbye!"}) {
		t.Fatalf("last line = %v", last)
	}
}

func TestClosingTurnTheWaitForTheBotsReplyCountsFromWhenItHeardTheLineEnd(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"yes"}))
	r.driver.closingReplyWait = 500 * time.Millisecond
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Is that all correct?"})
		r.say(map[string]any{"type": EventUserStartedSpeaking})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "Yes, thanks, goodbye."})
		r.endCall(t, true, "done")
		// The line plays to the bot for longer than the wait itself.
		time.Sleep(700 * time.Millisecond)
		r.say(map[string]any{"type": EventUserStoppedSpeaking})
		time.Sleep(300 * time.Millisecond)
		r.say(map[string]any{"type": "llm_response", "text": "Goodbye!"})
	})
	res := r.result(nil)
	if res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("ended by %s", res.EndedBy)
	}
	if last := res.Messages[len(res.Messages)-1]; !reflect.DeepEqual(last,
		map[string]any{"role": "assistant", "content": "Goodbye!"}) {
		t.Fatalf("last line = %v", last)
	}
}

func TestClosingTurnALongClosingReplyHoldsTheWaitUntilTheBotStops(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"yes"}))
	r.driver.closingReplyWait = 500 * time.Millisecond
	botReport := func(msgType string) {
		event := r.stream.RTVIMessageToEvent(rawMessage(t, msgType, nil))
		if event == nil {
			t.Errorf("%s maps to no event", msgType)
			return
		}
		r.say(event)
	}
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Is that all correct?"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "Yes, goodbye."})
		r.endCall(t, true, "done")
		botReport(rtvi.TypeBotStartedSpeaking)
		// The bot's reply goes on for longer than the wait itself.
		time.Sleep(700 * time.Millisecond)
		botReport(rtvi.TypeBotStoppedSpeaking)
		time.Sleep(200 * time.Millisecond)
		r.say(map[string]any{"type": "llm_response", "text": "Goodbye, safe travels!"})
	})
	res := r.result(nil)
	if res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("ended by %s", res.EndedBy)
	}
	if last := res.Messages[len(res.Messages)-1]; !reflect.DeepEqual(last,
		map[string]any{"role": "assistant", "content": "Goodbye, safe travels!"}) {
		t.Fatalf("last line = %v", last)
	}
}

func TestClosingTurnAHangUpWithNoReplyComingStillEndsAsEndCall(t *testing.T) {
	r := newSimulationRun(testSimulation(func(s *SimulationScenario) { s.MaxDurationS = 0.3 }),
		newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Is that all correct?"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "Yes, bye."})
		r.endCall(t, true, "done")
	})
	res := r.result(nil)
	if res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("ended by %s", res.EndedBy)
	}
	if last := res.Messages[len(res.Messages)-1]; !reflect.DeepEqual(last,
		map[string]any{"role": "user", "content": "Yes, bye."}) {
		t.Fatalf("last line = %v", last)
	}
}

func TestClosingTurnAHangUpOnTheBotsLineEndsAtOnce(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"yes"}))
	started := time.Now()
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Hello?"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "Wrong number."})
		r.say(map[string]any{"type": "llm_response", "text": "No problem, goodbye."})
		r.endCall(t, false, "wrong number")
	})
	if elapsed := time.Since(started); elapsed >= time.Second {
		t.Fatalf("took %s", elapsed)
	}
	if res := r.result(nil); res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("ended by %s", res.EndedBy)
	}
}

// A lull, a harness failure, or a judge with no verdict ends the run without
// waiting out the caps.

func TestSimulationEarlyEndingsALullEndsTheRunAsSilence(t *testing.T) {
	r := newSimulationRun(testSimulation(func(s *SimulationScenario) { s.MaxSilenceS, s.MaxDurationS = 0.05, 5 }),
		newFakeConversationJudge([]string{"no"}))
	r.play(t, nil)
	if res := r.result(nil); res.EndedBy != SimulationEndedBySilence || res.Passed() {
		t.Fatalf("result = %+v", res)
	}
}

func TestSimulationEarlyEndingsActivityKeepsALullFromEndingTheRun(t *testing.T) {
	r := newSimulationRun(testSimulation(func(s *SimulationScenario) { s.MaxSilenceS, s.MaxDurationS = 0.2, 5 }),
		newFakeConversationJudge([]string{"yes"}))
	r.play(t, func() {
		// Activity spaced inside the lull cap, spanning more than the cap: an
		// event, then frames that only buffer toward one.
		time.Sleep(100 * time.Millisecond)
		r.say(map[string]any{"type": "llm_started"})
		for range 3 {
			time.Sleep(100 * time.Millisecond)
			r.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "token "}))
		}
		r.endCall(t, true, "done")
	})
	if res := r.result(nil); res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("ended by %s", res.EndedBy)
	}
}

func TestSimulationEarlyEndingsAHarnessPipelineErrorIsTheRunsError(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"yes"}))
	failures := r.play(t, func() {
		r.say(map[string]any{"type": EventHarnessError, "text": "OLLamaLLMService: 400"})
	})
	if len(failures) != 1 || failures[0].Kind != FailureError || failures[0].Reason != "OLLamaLLMService: 400" {
		t.Fatalf("failures = %v", failures)
	}
	if !r.client.isHungUp() {
		t.Fatal("the persona did not hang up")
	}
	res := r.result(failures)
	if res.EndedBy != SimulationEndedByError || res.Error != "OLLamaLLMService: 400" || res.Passed() {
		t.Fatalf("result = %+v", res)
	}
}

func TestSimulationEarlyEndingsAJudgeWithNoGoalVerdictIsTheRunsError(t *testing.T) {
	r := newSimulationRun(testSimulation(), newFakeConversationJudge([]string{"none"}))
	failures := r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "Berlin."})
		r.endCall(t, true, "done")
	})
	if len(failures) != 1 || failures[0].Kind != FailureJudgeNoVerdict {
		t.Fatalf("failures = %v", failures)
	}
	res := r.result(failures)
	if res.EndedBy != SimulationEndedByEndCall || res.Succeeded || !strings.Contains(res.Error, "because") {
		t.Fatalf("result = %+v", res)
	}
}

// A failed metric's failure kind says whether the judge rejected a turn or left
// it unanswered.
func scoreMetricKinds(t *testing.T, turnVerdicts ...map[string]string) SimulationMetricScore {
	t.Helper()
	r := newSimulationRun(testSimulation(withMetrics(
		SimulationMetric{Name: "brevity", Criterion: "short", MinScore: new(1.0)},
	)), newFakeConversationJudge([]string{"yes"}, turnVerdicts...))
	r.play(t, func() {
		r.say(map[string]any{"type": "llm_response", "text": "one"})
		r.say(map[string]any{"type": EventPersonaTurn, "text": "and?"})
		r.say(map[string]any{"type": "llm_response", "text": "two"})
		r.endCall(t, true, "done")
	})
	return r.result(nil).Metrics[0]
}

func verdictsOf(m SimulationMetricScore) []string {
	var out []string
	for _, v := range m.Verdicts {
		out = append(out, v.Verdict)
	}
	return out
}

func TestSimulationMetricKindsARejectedTurnIsJudgeNo(t *testing.T) {
	m := scoreMetricKinds(t, map[string]string{}, map[string]string{"brevity": "no"})
	if m.Passed || m.FailureKind != FailureJudgeNo || !slices.Equal(verdictsOf(m), []string{"yes", "no"}) {
		t.Fatalf("metric = %+v", m)
	}
}

func TestSimulationMetricKindsAnUnansweredTurnAloneIsJudgeNoVerdict(t *testing.T) {
	m := scoreMetricKinds(t, map[string]string{}, map[string]string{"brevity": "none"})
	if m.Passed || m.FailureKind != FailureJudgeNoVerdict || !slices.Equal(verdictsOf(m), []string{"yes", "none"}) {
		t.Fatalf("metric = %+v", m)
	}
}

func TestSimulationMetricKindsAPassingMetricHasNoKind(t *testing.T) {
	m := scoreMetricKinds(t, map[string]string{}, map[string]string{})
	if !m.Passed || m.FailureKind != "" {
		t.Fatalf("metric = %+v", m)
	}
}

// The tool call lines are written the way the judge reads them.
func TestCallLinesWriteTheArgumentsAsTheJudgeReadsThem(t *testing.T) {
	got := CallLines(map[string]any{
		"name": "book", keyCanceled: true,
		"arguments": map[string]any{"when": "été", "party": 2.0, "tags": []any{"a", nil, true}, "ok": true},
	})
	want := []string{
		`book({"ok": true, "party": 2, "tags": ["a", null, true], "when": "\u00e9t\u00e9"})`, "book" + callCanceledSuffix,
	}
	if !slices.Equal(got, want) {
		t.Fatalf("lines = %q, want %q", got, want)
	}
}
