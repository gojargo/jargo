package eval

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// judgeFake records the conversation it is told, and passes every turn but the
// ones in failing.
type judgeFake struct {
	goal       string
	failing    []int
	transcript []map[string]any
	criteria   []Criterion
	closed     bool
}

func (j *judgeFake) AddUserMessage(text string) {
	j.transcript = append(j.transcript, map[string]any{"role": "user", "content": text})
}

func (j *judgeFake) AddAssistantMessage(text string) {
	j.transcript = append(j.transcript, map[string]any{"role": "assistant", "content": text})
}

func (j *judgeFake) AddToolCall(text string) {
	j.transcript = append(j.transcript, map[string]any{"role": "tool", "content": text})
}

func (j *judgeFake) EvaluateRun(_ context.Context, criteria []Criterion, success string) RunVerdicts {
	j.criteria = slices.Clone(criteria)
	turns := 0
	for _, e := range j.transcript {
		if e["role"] == "assistant" {
			turns++
		}
	}
	goal := j.goal
	if goal == "" {
		goal = VerdictYes
	}
	verdicts := RunVerdicts{
		Goal:  JudgeVerdict{Verdict: goal, Reason: "because " + success},
		Turns: map[string][]JudgeVerdict{},
	}
	for _, c := range criteria {
		for turn := 1; turn <= turns; turn++ {
			verdict := VerdictYes
			if slices.Contains(j.failing, turn) {
				verdict = VerdictNo
			}
			verdicts.Turns[c.Name] = append(verdicts.Turns[c.Name],
				JudgeVerdict{Verdict: verdict, Reason: fmt.Sprintf("turn %d %s", turn, c.Name)})
		}
	}
	return verdicts
}

func (j *judgeFake) Close(context.Context) error {
	j.closed = true
	return nil
}

func judgedSimulation(metrics ...SimulationMetric) *SimulationScenario {
	return &SimulationScenario{
		Name: "book/available", Persona: "A caller.", Goal: "Book a table.",
		Success: "the bot booked a table", Metrics: metrics, Judge: DefaultJudgeEvalConfig(),
	}
}

// judgedMessages is a conversation file's messages.
func judgedMessages() []map[string]any {
	return []map[string]any{
		{"role": "user", "content": "A table at six."},
		{"role": "tool_call", "name": "check", "arguments": map[string]any{"time": "6"}, keyCanceled: false},
		{"role": "tool_call", "name": "book", "arguments": map[string]any{}, keyCanceled: true},
		{"role": "assistant", "content": "Six is free."},
		{"role": "user", "content": "Thanks."},
		{"role": "assistant", "content": "Bye."},
	}
}

func TestSimulationJudgeTheJudgeReadsTheConversationWithEachCallInPlace(t *testing.T) {
	judge := &judgeFake{}
	if _, err := NewSimulationJudge(judgedSimulation(), judge, nil, nil).Judge(t.Context(), judgedMessages()); err != nil {
		t.Fatal(err)
	}
	want := []map[string]any{
		{"role": "user", "content": "A table at six."},
		{"role": "tool", "content": `check({"time": "6"})`},
		{"role": "tool", "content": "book()"},
		{"role": "tool", "content": "book" + callCanceledSuffix},
		{"role": "assistant", "content": "Six is free."},
		{"role": "user", "content": "Thanks."},
		{"role": "assistant", "content": "Bye."},
	}
	if !reflect.DeepEqual(judge.transcript, want) {
		t.Fatalf("transcript = %v", judge.transcript)
	}
	// A judge passed in is its owner's to close.
	if judge.closed {
		t.Fatal("the judge passed in was closed")
	}
}

func TestSimulationJudgeWithoutAJudgeEachConversationGetsOneBuiltAndClosed(t *testing.T) {
	var (
		configs []JudgeEvalConfig
		built   []*judgeFake
	)
	simulation := judgedSimulation(SimulationMetric{Name: "politeness", Criterion: "stays polite"})
	explainBelow := 2.0
	simulationJudge := NewSimulationJudge(simulation, nil, &explainBelow, nil)
	simulationJudge.newJudge = func(cfg *JudgeEvalConfig) (RunJudge, error) {
		configs = append(configs, *cfg)
		built = append(built, &judgeFake{})
		return built[len(built)-1], nil
	}
	if _, err := simulationJudge.Judge(t.Context(), judgedMessages()); err != nil {
		t.Fatal(err)
	}
	judgement, err := simulationJudge.Judge(t.Context(), judgedMessages())
	if err != nil {
		t.Fatal(err)
	}
	want := simulation.Judge
	want.ExplainBelow = &explainBelow
	if len(configs) != 2 || !reflect.DeepEqual(configs[0], want) || !reflect.DeepEqual(configs[1], want) {
		t.Fatalf("configs = %+v", configs)
	}
	if !built[0].closed || !built[1].closed {
		t.Fatal("a judge it built was left open")
	}
	if len(built[1].transcript) != 7 {
		t.Fatalf("transcript = %v", built[1].transcript)
	}
	if *judgement.Metrics[0].Score != 1 {
		t.Fatalf("score = %v", *judgement.Metrics[0].Score)
	}
	// The simulation's own block is left as it was.
	if simulation.Judge.ExplainBelow != nil {
		t.Fatal("the simulation's judge block was changed")
	}
}

func TestSimulationJudgeTheGoalAndTheCriterionMetricsAreJudgedAndTheMeasuresLeftOut(t *testing.T) {
	judge := &judgeFake{failing: []int{2}}
	trace := NewTrace()
	judgement, err := NewSimulationJudge(judgedSimulation(
		SimulationMetric{Name: "politeness", Criterion: "stays polite", MinScore: new(1.0)},
		SimulationMetric{Name: "turns", Measure: MeasureTurns, MaxValue: new(10.0)},
		SimulationMetric{Name: "brevity", Criterion: "keeps it short"},
	), judge, nil, trace).Judge(t.Context(), judgedMessages())
	if err != nil {
		t.Fatal(err)
	}
	want := []Criterion{{"politeness", "stays polite"}, {"brevity", "keeps it short"}}
	if !reflect.DeepEqual(judge.criteria, want) {
		t.Fatalf("criteria = %v", judge.criteria)
	}
	if judgement.Goal.Verdict != VerdictYes || judgement.Goal.Reason != "because the bot booked a table" {
		t.Fatalf("goal = %+v", judgement.Goal)
	}
	if len(judgement.Metrics) != 2 {
		t.Fatalf("metrics = %+v", judgement.Metrics)
	}
	politeness, brevity := judgement.Metrics[0], judgement.Metrics[1]
	if politeness.Name != "politeness" || *politeness.Score != 0.5 || politeness.Passed ||
		politeness.FailureKind != FailureJudgeNo || politeness.Reason != "turn 2: turn 2 politeness" ||
		politeness.Criterion != "stays polite" {
		t.Fatalf("politeness = %+v", politeness)
	}
	// Without a min_score, a metric reports and fails nothing.
	if *brevity.Score != 0.5 || !brevity.Passed {
		t.Fatalf("brevity = %+v", brevity)
	}
	if !strings.Contains(strings.Join(trace.Lines(), "\n"),
		"judge: politeness = 0.50 (below 1.00): turn 2: turn 2 politeness") {
		t.Fatalf("trace = %v", trace.Lines())
	}
}

func TestRubricIsJudgedLikeASimulation(t *testing.T) {
	judge := &judgeFake{failing: []int{2}}
	rubric, err := NewRubric("booking", "the bot booked a table",
		[]SimulationMetric{{Name: "politeness", Criterion: "stays polite", MinScore: new(1.0)}})
	if err != nil {
		t.Fatal(err)
	}
	judgement, err := NewSimulationJudge(rubric, judge, nil, nil).Judge(t.Context(), judgedMessages())
	if err != nil {
		t.Fatal(err)
	}
	if judgement.Goal.Reason != "because the bot booked a table" {
		t.Fatalf("goal = %+v", judgement.Goal)
	}
	if len(judgement.Metrics) != 1 || *judgement.Metrics[0].Score != 0.5 || judgement.Metrics[0].Passed {
		t.Fatalf("metrics = %+v", judgement.Metrics)
	}
}

func TestRubricBuildsItsJudgeFromItsConfig(t *testing.T) {
	var configs []JudgeEvalConfig
	rubric := &Rubric{
		Name: "booking", Success: "the bot booked a table",
		Judge: JudgeEvalConfig{LLMServiceConfig: LLMServiceConfig{Service: "ollama", Model: "judge"}},
	}
	simulationJudge := NewSimulationJudge(rubric, nil, nil, nil)
	simulationJudge.newJudge = func(cfg *JudgeEvalConfig) (RunJudge, error) {
		configs = append(configs, *cfg)
		return &judgeFake{}, nil
	}
	if _, err := simulationJudge.Judge(t.Context(), judgedMessages()); err != nil {
		t.Fatal(err)
	}
	if len(configs) != 1 || !reflect.DeepEqual(configs[0], rubric.Judge) {
		t.Fatalf("configs = %+v", configs)
	}
}
