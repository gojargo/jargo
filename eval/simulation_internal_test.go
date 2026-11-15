package eval

import (
	"reflect"
	"strings"
	"testing"
)

// minimalSimulation is a simulation's own mapping, as parseSimulation parses it.
const minimalSimulation = `
name: capital_curious
persona: "A curious traveler."
goal: "Learn the capital of Germany."
simulator: {service: openai, model: gpt-4o-mini}
success: "the bot said the capital of Germany is Berlin"
`

// minimalSimulationFile is the same simulation as a scenario file holds it.
const minimalSimulationFile = `
name: capital_curious
simulator: {service: openai, model: gpt-4o-mini}
scenarios:
  - name: capital_curious
    persona: "A curious traveler."
    goal: "Learn the capital of Germany."
    success: "the bot said the capital of Germany is Berlin"
`

// parseSimulationFile parses a file holding one simulation's own keys at its
// top level.
func parseSimulationFile(path string) (*SimulationScenario, error) {
	data, err := loadMapping(path)
	if err != nil {
		return nil, err
	}
	return parseSimulation(data, path)
}

// simulation parses body as a simulation, failing the test on an error.
func simulation(t *testing.T, body string) *SimulationScenario {
	t.Helper()
	s, err := parseSimulationFile(writeYAML(t, body))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// simulationErr parses body and returns the error it is rejected with.
func simulationErr(t *testing.T, body string) error {
	t.Helper()
	_, err := parseSimulationFile(writeYAML(t, body))
	if err == nil {
		t.Fatalf("expected an error for:\n%s", body)
	}
	return err
}

// withoutLines is body without the lines starting with prefix.
func withoutLines(body, prefix string) string {
	var kept []string
	for line := range strings.SplitSeq(body, "\n") {
		if !strings.HasPrefix(line, prefix) {
			kept = append(kept, line)
		}
	}
	return strings.Join(kept, "\n")
}

func TestSimulationMinimalHasDefaults(t *testing.T) {
	s := simulation(t, minimalSimulation)
	if s.Name != "capital_curious" || s.Persona != "A curious traveler." ||
		s.Goal != "Learn the capital of Germany." || s.Success != "the bot said the capital of Germany is Berlin" {
		t.Fatalf("simulation = %+v", s)
	}
	if !reflect.DeepEqual(s.Simulator.Raw, map[string]any{"service": "openai", "model": "gpt-4o-mini"}) ||
		s.Simulator.Service != "openai" || s.Simulator.Model != "gpt-4o-mini" {
		t.Fatalf("simulator = %+v", s.Simulator)
	}
	if len(s.Metrics) != 0 || s.BotAudio || s.UserAudio {
		t.Fatalf("simulation = %+v", s)
	}
	if s.MaxTurns != 20 || s.MaxDurationS != 300.0 || s.Runs != 1 || s.Judge.Service != "ollama" {
		t.Fatalf("defaults = %+v", s)
	}
}

func TestSimulationMetricsAndCaps(t *testing.T) {
	s := simulation(t, minimalSimulation+`
metrics:
  - name: politeness
    criterion: "stayed courteous"
  - {name: accuracy, criterion: "no invented facts", min_score: 0.8}
max_turns: 5
max_duration_s: 42
runs: 3
`)
	if len(s.Metrics) != 2 || s.Metrics[0].Name != "politeness" || s.Metrics[1].Name != "accuracy" {
		t.Fatalf("metrics = %+v", s.Metrics)
	}
	if s.Metrics[0].MinScore != nil || s.Metrics[1].MinScore == nil || *s.Metrics[1].MinScore != 0.8 {
		t.Fatalf("min scores = %+v", s.Metrics)
	}
	if s.MaxTurns != 5 || s.MaxDurationS != 42.0 || s.Runs != 3 {
		t.Fatalf("caps = %+v", s)
	}
}

func TestSimulationAMeasuredMetricTakesAMeasureAndARange(t *testing.T) {
	s := simulation(t, minimalSimulation+`
metrics:
  - measure: latency
    max_value: 2
  - name: quick
    measure: turns
    min_value: 1
    max_value: 6
`)
	latency, quick := s.Metrics[0], s.Metrics[1]
	if latency.Name != "latency" || latency.Measure != MeasureLatency || *latency.MaxValue != 2.0 ||
		latency.Criterion != "" {
		t.Fatalf("latency = %+v", latency)
	}
	if quick.Measure != MeasureTurns || *quick.MinValue != 1.0 || *quick.MaxValue != 6.0 {
		t.Fatalf("quick = %+v", quick)
	}
	for _, tc := range []struct{ bad, message string }{
		{"  - measure: mood\n    max_value: 1\n", "must be one of"},
		{"  - measure: turns\n", "needs a 'min_value:' or a 'max_value:'"},
		{"  - measure: turns\n    max_value: 3\n    min_score: 1\n", "takes a range"},
		{"  - name: both\n    criterion: x\n    measure: turns\n    max_value: 3\n", "one of the two"},
	} {
		wantErrContains(t, simulationErr(t, minimalSimulation+"metrics:\n"+tc.bad), tc.message)
	}
}

func TestSimulationMinScoreIsAShareAndNamesAreUnique(t *testing.T) {
	wantErrContains(t, simulationErr(t, minimalSimulation+"metrics:\n  - {name: a, criterion: x, min_score: 2}\n"), "0..1")
	wantErrContains(t, simulationErr(t, minimalSimulation+
		"metrics:\n  - {name: a, criterion: x}\n  - {name: a, criterion: y}\n"), "twice")
}

func TestSimulationAudioModalities(t *testing.T) {
	s := simulation(t, minimalSimulation+`
user:
  modality: audio
  speech: {service: kokoro, voice: af_heart}
judge:
  modality: audio
  transcription: {service: moonshine}
`)
	if !s.UserAudio || !reflect.DeepEqual(s.UserSpeech.Raw, map[string]any{"service": "kokoro", "voice": "af_heart"}) {
		t.Fatalf("speech = %+v", s.UserSpeech)
	}
	if !s.BotAudio || !reflect.DeepEqual(s.Transcriber.Raw, map[string]any{"service": "moonshine"}) {
		t.Fatalf("transcriber = %+v", s.Transcriber)
	}
}

func TestSimulationUserAudioRequiresSpeech(t *testing.T) {
	wantErrContains(t, simulationErr(t, minimalSimulation+"user: {modality: audio}\n"), "user.speech")
}

func TestSimulationRequiredFields(t *testing.T) {
	for _, missing := range []string{"persona", "goal", "success"} {
		wantErrContains(t, simulationErr(t, withoutLines(minimalSimulation, missing)), missing)
	}
}

func TestSimulationAFunctionCallsMeasureTakesTheCallsTheBotShouldMake(t *testing.T) {
	s := simulation(t, minimalSimulation+`
metrics:
  - measure: function_calls
    calls:
      - book_table
      - name: send_confirmation
        args: { channel: sms }
  - name: hands_off
    measure: function_calls
    calls: []
`)
	booked, handsOff := s.Metrics[0], s.Metrics[1]
	want := []FunctionCall{{Name: "book_table"}, {Name: "send_confirmation", Args: map[string]any{"channel": "sms"}}}
	if booked.Name != "function_calls" || !reflect.DeepEqual(booked.Calls, want) {
		t.Fatalf("booked = %+v", booked)
	}
	if handsOff.Measure != MeasureFunctionCalls || handsOff.Calls == nil || len(handsOff.Calls) != 0 {
		t.Fatalf("hands_off = %+v", handsOff)
	}
	for _, tc := range []struct{ bad, message string }{
		{"  - measure: function_calls\n", "needs a 'calls:' list"},
		{"  - measure: function_calls\n    calls: [book_table]\n    max_value: 1\n", "not a range"},
		{"  - measure: function_calls\n    calls: [{args: {a: 1}}]\n", "entry #0 must be a name"},
		{"  - measure: turns\n    max_value: 3\n    calls: []\n", "'calls:' belongs to"},
	} {
		wantErrContains(t, simulationErr(t, minimalSimulation+"metrics:\n"+tc.bad), tc.message)
	}
}

func TestSimulationMetricNeedsACriterion(t *testing.T) {
	wantErrContains(t, simulationErr(t, minimalSimulation+"metrics: [{name: politeness}]\n"), "criterion")
}

func TestSimulationCapsMustBePositive(t *testing.T) {
	_ = simulationErr(t, minimalSimulation+"max_turns: 0\n")
	_ = simulationErr(t, minimalSimulation+"max_duration_s: -1\n")
}

func TestDescribeSimulation(t *testing.T) {
	text := DescribeSimulation(simulation(t, minimalSimulation), false)
	for _, want := range []string{
		"user  -> modality: text | persona: openai/gpt-4o-mini | max_turns: 20 | max_duration_s: 300",
		"judge -> modality: text | eval: ollama/",
		"goal  -> Learn the capital of Germany.",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("summary lacks %q:\n%s", want, text)
		}
	}
	if n := len(strings.Split(text, "\n")); n != 3 {
		t.Fatalf("summary has %d lines, want 3", n)
	}
}

func TestLoadScenariosAPersonaMakesASimulation(t *testing.T) {
	file, err := LoadFile(writeYAML(t, minimalSimulationFile))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := file.Scenarios[0].(*SimulationScenario); !ok || len(file.Scenarios) != 1 {
		t.Fatalf("scenarios = %+v", file.Scenarios)
	}
}

func TestLoadScenariosTurnsMakeAScriptedScenario(t *testing.T) {
	file, err := LoadFile(writeYAML(t, "name: greet\nscenarios: [{name: greet, turns: []}]\n"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := file.Scenarios[0].(*ScriptScenario); !ok || len(file.Scenarios) != 1 {
		t.Fatalf("scenarios = %+v", file.Scenarios)
	}
}

func TestLoadScenariosAScenarioIsOneKindOrTheOther(t *testing.T) {
	_, err := LoadFile(writeYAML(t, "name: nothing\nscenarios: [{name: nothing}]\n"))
	if err == nil {
		t.Fatal("expected an error for a scenario of no kind")
	}
	wantErrContains(t, err, "'turns:'", "'persona:'")
	_, err = LoadFile(writeYAML(t, minimalSimulationFile+"    turns: []\n"))
	if err == nil {
		t.Fatal("expected an error for a scenario of both kinds")
	}
	wantErrContains(t, err, "not both")
}

func TestSimulatorIsOptionalAndDescribesAsTheDefaultJudgeModel(t *testing.T) {
	s := simulation(t, withoutLines(minimalSimulation, "simulator"))
	if s.Simulator.Raw != nil || s.MaxSilenceS != 30.0 {
		t.Fatalf("simulation = %+v", s)
	}
	described := DescribeSimulation(s, false)
	for _, want := range []string{"persona: ollama/gemma4:12b", "max_silence_s: 30"} {
		if !strings.Contains(described, want) {
			t.Fatalf("summary lacks %q:\n%s", want, described)
		}
	}
}

func TestSimulatorAPartialSimulatorKeepsItsOwnValues(t *testing.T) {
	s := simulation(t, minimalSimulation+"max_silence_s: 7\n")
	if s.MaxSilenceS != 7.0 || !strings.Contains(DescribeSimulation(s, false), "persona: openai/gpt-4o-mini") {
		t.Fatalf("simulation = %+v", s)
	}
}

// A service without a model shows the model the builder defaults to; a factory
// is named as such, since the summary cannot know what it builds.
func TestSimulatorTheSummaryNamesWhatTheBlockBuilds(t *testing.T) {
	without := withoutLines(minimalSimulation, "simulator")
	openai := simulation(t, without+"\nsimulator: {service: openai}\n")
	if !strings.Contains(DescribeSimulation(openai, false), "persona: openai/gpt-4o") {
		t.Fatalf("summary:\n%s", DescribeSimulation(openai, false))
	}
	factory := simulation(t, without+"\nsimulator: {factory: my_evals.persona}\n")
	if !strings.Contains(DescribeSimulation(factory, false), "persona: factory:my_evals.persona") {
		t.Fatalf("summary:\n%s", DescribeSimulation(factory, false))
	}
	judged := simulation(t, minimalSimulation+"judge: {eval: {factory: my_evals.judge}}\n")
	if !strings.Contains(DescribeSimulation(judged, false), "eval: factory:my_evals.judge") {
		t.Fatalf("summary:\n%s", DescribeSimulation(judged, false))
	}
}

func TestRubricFromMapReadsARubricFilesKeys(t *testing.T) {
	r, err := RubricFromMap(map[string]any{
		"name":    "booking",
		"judge":   map[string]any{"eval": map[string]any{"service": "ollama", "model": "judge"}},
		"success": " the bot booked a table ",
		"metrics": []any{map[string]any{"name": "politeness", "criterion": "stays polite", "min_score": 1}},
	}, "")
	if err != nil {
		t.Fatal(err)
	}
	if r.Name != "booking" || r.Success != "the bot booked a table" {
		t.Fatalf("rubric = %+v", r)
	}
	one := 1.0
	want := []SimulationMetric{{Name: "politeness", Criterion: "stays polite", MinScore: &one}}
	if !reflect.DeepEqual(r.Metrics, want) {
		t.Fatalf("metrics = %+v, want %+v", r.Metrics, want)
	}
	if !reflect.DeepEqual(r.Judge.Raw, map[string]any{"service": "ollama", "model": "judge"}) {
		t.Fatalf("judge = %+v", r.Judge)
	}
}

func TestRubricAnInvalidRubricIsRejected(t *testing.T) {
	cases := map[string]map[string]any{
		"no name":       {"success": "booked"},
		"no success":    {"name": "r"},
		"empty success": {"name": "r", "success": ""},
		"a measured metric": {
			"name": "r", "success": "booked",
			"metrics": []any{map[string]any{"measure": "words", "max_value": 5}},
		},
		"a bad metric": {
			"name": "r", "success": "booked",
			"metrics": []any{map[string]any{"name": "x", "criterion": "c", "min_score": 2}},
		},
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := RubricFromMap(data, "")
			if err == nil || !strings.HasPrefix(err.Error(), "rubric: ") {
				t.Fatalf("err = %v, want one starting with rubric: ", err)
			}
		})
	}
}

// A rubric built directly refuses a measured metric too.
func TestNewRubricRefusesAMeasuredMetric(t *testing.T) {
	five := 5.0
	_, err := NewRubric("r", "booked", []SimulationMetric{{Name: "words", Measure: MeasureWords, MaxValue: &five}})
	if err == nil {
		t.Fatal("expected an error for a measured metric")
	}
	wantErrContains(t, err, "metric 'words' is measured")
}
