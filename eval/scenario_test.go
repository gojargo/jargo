package eval_test

import (
	"bytes"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gojargo/jargo/eval"
)

func writeScenario(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "scenario.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// loadFile loads body as a scenario file, failing the test on an error.
func loadFile(t *testing.T, body string) *eval.ScenarioFile {
	t.Helper()
	file, err := eval.LoadFile(writeScenario(t, body))
	if err != nil {
		t.Fatal(err)
	}
	return file
}

// loadFileErr loads body and returns the error it is rejected with.
func loadFileErr(t *testing.T, body string) error {
	t.Helper()
	_, err := eval.LoadFile(writeScenario(t, body))
	if err == nil {
		t.Fatalf("expected an error for:\n%s", body)
	}
	if _, ok := errors.AsType[*eval.ScenarioError](err); !ok {
		t.Fatalf("error %v is not a *ScenarioError", err)
	}
	return err
}

// mustContain asserts err's message holds want.
func mustContain(t *testing.T, err error, want string) {
	t.Helper()
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("error %q does not contain %q", err, want)
	}
}

// names are the names of a file's scenarios, in order.
func names(file *eval.ScenarioFile) []string {
	out := make([]string, 0, len(file.Scenarios))
	for _, s := range file.Scenarios {
		out = append(out, s.ScenarioName())
	}
	return out
}

// get is the scenario called name, of type T.
func get[T eval.LoadedScenario](t *testing.T, file *eval.ScenarioFile, name string) T {
	t.Helper()
	s, err := file.Scenario(name)
	if err != nil {
		t.Fatal(err)
	}
	typed, ok := s.(T)
	if !ok {
		t.Fatalf("%s is a %T", name, s)
	}
	return typed
}

const group = `
name: turn_completion
judge: {modality: text}
context:
  - role: system
    content: "You are a travel assistant."
stop_on_failure: false

scenarios:
  - name: short_answer
    turns:
      - user: "Japan."
        expect: [{event: response}]
  - name: with_history
    stop_on_failure: true
    context:
      - role: system
        content: "You are a travel assistant."
      - role: assistant
        content: "Where would you go?"
    turns:
      - user: "Japan."
        expect: [{event: response}]
`

func TestAFileOfOneScenarioNamesItUnderTheFile(t *testing.T) {
	file := loadFile(t, "name: greet\nscenarios:\n  - name: greet\n    turns: [{user: hi}]\n")
	if got := names(file); len(got) != 1 || got[0] != "greet/greet" {
		t.Fatalf("names = %v", got)
	}
}

func TestAFlatFileIsDeprecatedAndKeepsItsName(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	defer slog.SetDefault(prev)

	file := loadFile(t, "name: greet\nturns: [{user: hi}]\n")
	if got := names(file); len(got) != 1 || got[0] != "greet" {
		t.Fatalf("names = %v", got)
	}
	if !strings.Contains(buf.String(), "`scenarios:` list") {
		t.Fatalf("want a deprecation warning, logged:\n%s", buf.String())
	}
}

func TestEntriesAreNamedUnderTheGroup(t *testing.T) {
	file := loadFile(t, group)
	got := names(file)
	if len(got) != 2 || got[0] != "turn_completion/short_answer" || got[1] != "turn_completion/with_history" {
		t.Fatalf("names = %v", got)
	}
	for _, s := range file.Scenarios {
		if _, ok := s.(*eval.ScriptScenario); !ok {
			t.Fatalf("%s is a %T, want a scripted scenario", s.ScenarioName(), s)
		}
	}
}

func TestHeaderKeysAreDefaultsAndAnEntryReplacesAWholeValue(t *testing.T) {
	file := loadFile(t, group)
	short := get[*eval.ScriptScenario](t, file, "turn_completion/short_answer")
	history := get[*eval.ScriptScenario](t, file, "turn_completion/with_history")
	// Inherited from the header.
	if len(short.Context) != 1 || short.Context[0]["content"] != "You are a travel assistant." {
		t.Fatalf("context = %+v", short.Context)
	}
	if *short.StopOnFailure || short.BotAudio {
		t.Fatalf("short = %+v", short)
	}
	// Replaced as a whole, not merged.
	if len(history.Context) != 2 || history.Context[1]["role"] != "assistant" {
		t.Fatalf("context = %+v", history.Context)
	}
	if !*history.StopOnFailure || history.BotAudio {
		t.Fatalf("history = %+v", history)
	}
}

func TestEntriesCanBeSimulations(t *testing.T) {
	file := loadFile(t, `
name: diner
persona: "An impatient diner."
success: "the bot confirms"
scenarios:
  - name: book
    goal: "Book a table."
  - name: cancel
    goal: "Cancel a booking."
`)
	book := get[*eval.SimulationScenario](t, file, "diner/book")
	cancel := get[*eval.SimulationScenario](t, file, "diner/cancel")
	if book.Goal != "Book a table." || cancel.Goal != "Cancel a booking." || book.Persona != "An impatient diner." {
		t.Fatalf("book %+v, cancel %+v", book, cancel)
	}
}

func TestAnEntryWithSuccessAndNoPersonaOrTurnsIsARubric(t *testing.T) {
	file := loadFile(t, `
name: diner
judge: {eval: {service: ollama, model: judge}}
metrics:
  - name: politeness
    criterion: "stays polite"
scenarios:
  - name: simulated
    persona: "An impatient diner."
    goal: "Book a table."
    success: "the bot booked a table"
  - name: production
    success: "the bot booked the table the caller asked for"
`)
	get[*eval.SimulationScenario](t, file, "diner/simulated")
	rubric := get[*eval.Rubric](t, file, "diner/production")
	if rubric.Success != "the bot booked the table the caller asked for" {
		t.Fatalf("success = %q", rubric.Success)
	}
	// The file's defaults are merged in, as for any entry.
	if len(rubric.Metrics) != 1 || rubric.Metrics[0].Name != "politeness" {
		t.Fatalf("metrics = %+v", rubric.Metrics)
	}
	if rubric.Judge.Service != "ollama" || rubric.Judge.Model != "judge" || len(rubric.Judge.Raw) != 2 {
		t.Fatalf("judge = %+v", rubric.Judge)
	}
}

func TestARubricLeavesOutTheMeasuredMetricsItTakesFromTheFile(t *testing.T) {
	file := loadFile(t, `
name: diner
metrics:
  - name: politeness
    criterion: "stays polite"
  - measure: words
    max_value: 40
scenarios:
  - name: simulated
    persona: "An impatient diner."
    goal: "Book a table."
    success: "the bot booked a table"
  - name: production
    success: "the bot booked the table the caller asked for"
`)
	simulated := get[*eval.SimulationScenario](t, file, "diner/simulated")
	if len(simulated.Metrics) != 2 || simulated.Metrics[0].Name != "politeness" || simulated.Metrics[1].Name != "words" {
		t.Fatalf("simulated metrics = %+v", simulated.Metrics)
	}
	production := get[*eval.Rubric](t, file, "diner/production")
	if len(production.Metrics) != 1 || production.Metrics[0].Name != "politeness" {
		t.Fatalf("production metrics = %+v", production.Metrics)
	}
}

func TestARubricRefusesAMeasuredMetricItListsItself(t *testing.T) {
	err := loadFileErr(t, `
name: diner
scenarios:
  - name: production
    success: "the bot booked a table"
    metrics:
      - measure: words
        max_value: 40
`)
	mustContain(t, err, "metric 'words' is measured")
}

func TestAFileKnowsItsNameAndPathAndFindsAScenarioByName(t *testing.T) {
	path := writeScenario(t, group)
	file, err := eval.LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if file.Name != "turn_completion" || file.Path != path || len(file.Scenarios) != 2 {
		t.Fatalf("file = %+v", file)
	}
	if s := get[*eval.ScriptScenario](t, file, "turn_completion/with_history"); s.Name != "turn_completion/with_history" {
		t.Fatalf("name = %q", s.Name)
	}
	_, err = file.Scenario("turn_completion/nope")
	if err == nil {
		t.Fatal("expected an error for a missing scenario")
	}
	mustContain(t, err, "no scenario called 'turn_completion/nope'")
}

func TestLoadScenarioFileWantsALoneScenario(t *testing.T) {
	one := "name: greet\nscenarios:\n  - name: greet\n    turns: [{user: hi}]\n"
	s, err := eval.LoadScenarioFile(writeScenario(t, one))
	if err != nil {
		t.Fatal(err)
	}
	if s.ScenarioName() != "greet/greet" {
		t.Fatalf("name = %q", s.ScenarioName())
	}
	_, err = eval.LoadScenarioFile(writeScenario(t, group))
	if err == nil {
		t.Fatal("expected an error for a file of two scenarios")
	}
	mustContain(t, err, "holds 2 scenarios")
}

func TestGroupNeedsAName(t *testing.T) {
	mustContain(t, loadFileErr(t, "scenarios: [{name: a, turns: []}]\n"), "missing or invalid 'name:'")
}

func TestGroupNeedsEntries(t *testing.T) {
	mustContain(t, loadFileErr(t, "name: g\nscenarios: []\n"), "non-empty list")
}

func TestEntryNeedsAName(t *testing.T) {
	mustContain(t, loadFileErr(t, "name: g\nscenarios: [{turns: []}]\n"), "scenario #0 needs a 'name:'")
}

func TestEntryCannotNestAGroup(t *testing.T) {
	mustContain(t, loadFileErr(t, "name: g\nscenarios: [{name: a, scenarios: []}]\n"), "cannot hold a 'scenarios:'")
}

func TestDuplicateEntryNamesRejected(t *testing.T) {
	err := loadFileErr(t, "name: g\nscenarios: [{name: a, turns: []}, {name: a, turns: []}]\n")
	mustContain(t, err, "duplicate scenario names: g/a")
}

func TestEntryErrorsNameTheFile(t *testing.T) {
	path := writeScenario(t, "name: g\nscenarios: [{name: a, turns: [{user: hi, expect: nope}]}]\n")
	_, err := eval.LoadFile(path)
	if err == nil {
		t.Fatal("expected an error")
	}
	mustContain(t, err, path)
}

func TestAScenarioIsOneKindOrTheOther(t *testing.T) {
	err := loadFileErr(t, "name: nothing\nscenarios: [{name: nothing}]\n")
	mustContain(t, err, "'turns:'")
	mustContain(t, err, "'persona:'")
	err = loadFileErr(t, "name: both\nscenarios: [{name: both, persona: p, goal: g, success: s, turns: []}]\n")
	mustContain(t, err, "not both")
}

// checkFromFile is a one-scenario file whose one turn has one expectation.
func checkFromFile(expectation string) string {
	return "name: g\nscenarios:\n  - name: a\n    turns:\n      - user: hi\n" +
		"        expect:\n          - " + expectation + "\n"
}

func TestCheckFrom(t *testing.T) {
	t.Run("defaults to previous", func(t *testing.T) {
		s := get[*eval.ScriptScenario](t, loadFile(t, checkFromFile("{event: response}")), "g/a")
		if got := s.Turns[0].Expect[0].CheckFrom; got != eval.CheckFromPrevious {
			t.Fatalf("check_from = %q", got)
		}
	})
	t.Run("turn_start is accepted", func(t *testing.T) {
		s := get[*eval.ScriptScenario](t, loadFile(t, checkFromFile("{event: response, check_from: turn_start}")), "g/a")
		if got := s.Turns[0].Expect[0].CheckFrom; got != eval.CheckFromTurnStart {
			t.Fatalf("check_from = %q", got)
		}
	})
	t.Run("other values are rejected", func(t *testing.T) {
		err := loadFileErr(t, checkFromFile("{event: response, check_from: later}"))
		mustContain(t, err, "'check_from:' must be one of previous, turn_start")
	})
	t.Run("not combined with absent", func(t *testing.T) {
		err := loadFileErr(t, checkFromFile("{event: response, absent: true, check_from: turn_start}"))
		mustContain(t, err, "cannot be combined with 'check_from'")
	})
}

// The deprecated single-scenario loader still reads a scripted scenario.
func TestLoadReadsALoneScriptedScenario(t *testing.T) {
	s, err := eval.Load(writeScenario(t, "name: greet\nscenarios:\n  - name: greet\n    turns: [{user: hi}]\n"))
	if err != nil {
		t.Fatal(err)
	}
	if s.Name != "greet/greet" || s.Turns[0].User != "hi" {
		t.Fatalf("scenario = %+v", s)
	}
	_, err = eval.Load(writeScenario(t, "name: s\nscenarios: [{name: s, persona: p, goal: g, success: s}]\n"))
	if err == nil {
		t.Fatal("expected an error for a simulation")
	}
}

// !include pulls any value from a separate file, so scenarios can share a block
// they all need. Paths resolve against the scenario file.
func TestLoadInclude(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "turns.yaml", "- user: \"hello\"\n  expect: [{event: llm_response}]\n")
	writeFile(t, dir, "scenario.yaml", "name: shared\nscenarios: [{name: shared, turns: !include turns.yaml}]\n")

	file, err := eval.LoadFile(filepath.Join(dir, "scenario.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	s := get[*eval.ScriptScenario](t, file, "shared/shared")
	if len(s.Turns) != 1 || s.Turns[0].User != "hello" {
		t.Fatalf("the included turns should have been pulled in, got %+v", s.Turns)
	}
}

// A file that includes itself is reported rather than exhausting the stack.
func TestLoadIncludeCycleRejected(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "loop.yaml", "name: loop\nturns: !include loop.yaml\n")

	if _, err := eval.LoadFile(filepath.Join(dir, "loop.yaml")); err == nil {
		t.Fatal("expected an error for the include cycle")
	} else if !strings.Contains(err.Error(), "nested too deep") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// The event names a scenario may use are open: the harness matches whatever the
// bot reports, so an unrecognized name loads and fails at match time, naming the
// event, rather than being rejected up front.
func TestLoadAcceptsAnyEventName(t *testing.T) {
	file := loadFile(t, "name: x\nscenarios: [{name: x, turns: [{user: hi, expect: [{event: something_new}]}]}]\n")
	if got := get[*eval.ScriptScenario](t, file, "x/x").Turns[0].Expect[0].Event; got != "something_new" {
		t.Fatalf("event = %q", got)
	}
}

// Calls only mean something on a function call event; elsewhere they are
// dropped rather than rejected.
func TestLoadIgnoresCallFieldsOffFunctionCall(t *testing.T) {
	file := loadFile(t, "name: x\n"+
		"scenarios: [{name: x, turns: [{user: hi, expect: [{event: llm_response, name: foo}]}]}]\n")
	if got := get[*eval.ScriptScenario](t, file, "x/x").Turns[0].Expect[0].Calls; got != nil {
		t.Fatalf("calls should be dropped off a function_call, got %+v", got)
	}
}

// An empty turns: list only asserts that the bot completes the handshake; a
// missing one is the mistake.
func TestLoadEmptyTurnsAllowed(t *testing.T) {
	loadFile(t, "name: handshake_only\nscenarios: [{name: h, turns: []}]\n")
}

// The fragments a directory's scenarios include have no name of their own, so
// they are not scenarios. A file that does not parse still counts as one, so
// loading it reports the error rather than a directory run skipping it.
func TestIsScenarioFileLeavesIncludeFragmentsOut(t *testing.T) {
	dir := t.TempDir()
	files := map[string]struct {
		body string
		want bool
	}{
		"alpha.yaml":  {"name: alpha\nscenarios: [{name: alpha, turns: []}]\n", true},
		"judge.yaml":  {"modality: text\n", false},
		"user.yaml":   {"service: openai\n", false},
		"broken.yaml": {"name: [broken\n", true},
	}
	for file, f := range files {
		writeFile(t, dir, file, f.body)
	}
	for file, f := range files {
		if got := eval.IsScenarioFile(filepath.Join(dir, file)); got != f.want {
			t.Errorf("IsScenarioFile(%s) = %v, want %v", file, got, f.want)
		}
	}
}

// A scenario's config renders as the two-line summary printed before a run.
func TestDescribeConfig(t *testing.T) {
	file := loadFile(t, `
name: a
user: {modality: audio, speech: {service: kokoro, voice: af_heart}}
judge:
  modality: audio
  transcription: {service: moonshine, model: small-streaming}
scenarios: [{name: a, turns: [{user: hi, expect: [{event: response}]}]}]
`)
	s := get[*eval.ScriptScenario](t, file, "a/a")
	want := "user  -> modality: audio | speech: kokoro/af_heart\n" +
		"judge -> modality: audio | transcription: moonshine/small-streaming | eval: ollama/gemma4:12b"
	if got := eval.DescribeConfig(&s.Configured, false); got != want {
		t.Fatalf("DescribeConfig =\n%s\nwant\n%s", got, want)
	}
	colored := eval.DescribeConfig(&s.Configured, true)
	if !strings.Contains(colored, "\033[1muser \033[0m") || !strings.Contains(colored, "\033[35meval:\033[0m") {
		t.Fatalf("colored summary = %q", colored)
	}
}
