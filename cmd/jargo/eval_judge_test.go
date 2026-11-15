package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
)

// verdictClassifier answers every question with the same verdict, surely: the
// judge the judge command's tests judge with.
type verdictClassifier struct {
	*classifier.Base
	yes bool
}

func (*verdictClassifier) Model() string { return "" }

func (c *verdictClassifier) AskQuestions(
	_ context.Context, _ any, questions []classifier.Named[classifier.Question], _ []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	verdict, probability := eval.VerdictNo, 0.01
	if c.yes {
		verdict, probability = eval.VerdictYes, 0.99
	}
	results := map[string]classifier.Result{}
	for _, q := range questions {
		switch q.Question.(type) {
		case classifier.YesNoQuestion:
			results[q.Name] = classifier.YesNoResult{Probability: probability}
		case classifier.ChoiceQuestion:
			results[q.Name] = classifier.ChoiceResult{
				Choice: verdict, Probabilities: map[string]float64{verdict: 0.99}, Confidence: 0.99,
			}
		}
	}
	return results, nil, nil
}

func init() {
	for name, yes := range map[string]bool{"clitest.yes_classifier": true, "clitest.no_classifier": false} {
		eval.RegisterFactory(name, func(map[string]any) (any, error) {
			c := &verdictClassifier{yes: yes}
			c.Base = classifier.NewBase("VerdictClassifier", "", c)
			return c, nil
		})
	}
}

const rubricFile = `
name: booking
judge:
  eval: {factory: %s, explainer: false}
scenarios:
  - name: production
    success: "the bot booked a table"
    metrics:
      - name: politeness
        criterion: "stays polite"
        min_score: 1
`

//nolint:misspell // the conversation file's spelling
const conversationFile = `[
  {"role": "user", "content": "A table at six."},
  {"role": "tool_call", "name": "book", "arguments": {"time": "6"}, "cancelled": false},
  {"role": "assistant", "content": "Booked for six."},
  {"role": "user", "content": "Thanks."},
  {"role": "assistant", "content": "Bye."}
]`

// judgeCLI runs `jargo eval judge` with args, its output in dir.
func judgeCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs(append([]string{"eval", "judge"}, args...))
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	err := cmd.Execute()
	return out.String(), err
}

func writeIn(t *testing.T, dir, name, body string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func readJSONLines(t *testing.T, path string) []map[string]any {
	t.Helper()
	f, err := os.Open(path) //nolint:gosec // a test file
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var records []map[string]any
	for s := bufio.NewScanner(f); s.Scan(); {
		var r map[string]any
		if err := json.Unmarshal(s.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		records = append(records, r)
	}
	return records
}

// A conversation is judged against a rubric's success and criteria, and its
// verdicts written as one JSONL line.
func TestEvalJudgeJudgesAConversationAgainstARubric(t *testing.T) {
	dir := t.TempDir()
	scenario := writeIn(t, dir, "booking.yaml", strings.Replace(rubricFile, "%s", "clitest.yes_classifier", 1))
	conversation := writeIn(t, dir, "call.conversation.json", conversationFile)
	output := filepath.Join(dir, "judged.jsonl")

	out, err := judgeCLI(t, scenario, conversation, "-o", output, "-v")
	if err != nil {
		t.Fatalf("judge failed: %v\n%s", err, out)
	}
	for _, want := range []string{"✓ " + conversation, "judge:", "politeness: 1.00", "1/1 passed", "results: " + output} {
		if !strings.Contains(out, want) {
			t.Fatalf("output lacks %q:\n%s", want, out)
		}
	}
	records := readJSONLines(t, output)
	if len(records) != 1 {
		t.Fatalf("records = %v", records)
	}
	r := records[0]
	if r["conversation"] != conversation || r["scenario"] != "booking/production" || r["passed"] != true ||
		r["success"] != "the bot booked a table" || r["error"] != nil {
		t.Fatalf("record = %v", r)
	}
	metrics, _ := r["metrics"].([]any)
	if len(metrics) != 1 {
		t.Fatalf("metrics = %v", r["metrics"])
	}
	politeness, _ := metrics[0].(map[string]any)
	verdicts, _ := politeness["verdicts"].([]any)
	if politeness["criterion"] != "stays polite" || len(verdicts) != 2 {
		t.Fatalf("politeness = %v", politeness)
	}
	if second, _ := verdicts[1].(map[string]any); second["turn"] != 2.0 || second["text"] != "Bye." {
		t.Fatalf("turn 2 = %v", second)
	}
}

// A conversation that does not pass, or cannot be read, fails the command.
func TestEvalJudgeAConversationThatFailsFailsTheCommand(t *testing.T) {
	dir := t.TempDir()
	scenario := writeIn(t, dir, "booking.yaml", strings.Replace(rubricFile, "%s", "clitest.no_classifier", 1))
	conversation := writeIn(t, dir, "call.conversation.json", conversationFile)
	missing := filepath.Join(dir, "missing.conversation.json")
	output := filepath.Join(dir, "judged.jsonl")

	out, err := judgeCLI(t, scenario, conversation, missing, "-o", output, "-c", "1")
	if err == nil || !strings.Contains(err.Error(), "2 of 2") {
		t.Fatalf("err = %v\n%s", err, out)
	}
	if !strings.Contains(out, "0/2 passed") || !strings.Contains(out, "goal not met") {
		t.Fatalf("output:\n%s", out)
	}
	records := readJSONLines(t, output)
	if len(records) != 2 {
		t.Fatalf("records = %v", records)
	}
	for _, r := range records {
		if reason, _ := r["error"].(string); r["conversation"] == missing && !strings.HasPrefix(reason, "error: ") {
			t.Fatalf("record = %v", r)
		}
	}
}

// A file holding several simulations and rubrics needs -s to pick one, by its
// name with or without the file's.
func TestEvalJudgePicksTheSimulationOrRubricToJudgeWith(t *testing.T) {
	dir := t.TempDir()
	scenario := writeIn(t, dir, "booking.yaml", strings.Replace(rubricFile, "%s", "clitest.yes_classifier", 1)+`
  - name: other
    success: "the bot was polite"
  - name: scripted
    turns: []
`)
	conversation := writeIn(t, dir, "call.conversation.json", conversationFile)
	output := filepath.Join(dir, "judged.jsonl")

	if _, err := judgeCLI(t, scenario, conversation, "-o", output); err == nil ||
		!strings.Contains(err.Error(), "pick one of the file's simulations and rubrics: booking/production, booking/other") {
		t.Fatalf("err = %v", err)
	}
	if _, err := judgeCLI(t, scenario, conversation, "-o", output, "-s", "nope"); err == nil ||
		!strings.Contains(err.Error(), "no simulation or rubric 'nope'") {
		t.Fatalf("err = %v", err)
	}
	for _, name := range []string{"production", "booking/production"} {
		if out, err := judgeCLI(t, scenario, conversation, "-o", output, "-s", name); err != nil {
			t.Fatalf("-s %s: %v\n%s", name, err, out)
		}
		if r := readJSONLines(t, output); r[0]["scenario"] != "booking/production" {
			t.Fatalf("-s %s judged %v", name, r[0]["scenario"])
		}
	}
}

// `jargo eval run` runs a simulation too, keeping its conversation beside its
// logs; one that cannot reach the bot fails with the kind of failure.
func TestEvalRunRunsASimulation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()
	path := writeScenario(t, `
name: capital
persona: "A curious traveler."
goal: "Learn the capital of Germany."
success: "the bot named Berlin"
`)
	logs := t.TempDir()
	var out strings.Builder
	cmd := rootCmd()
	cmd.SetArgs([]string{"eval", "run", path, "--bot-url", "ws" + strings.TrimPrefix(srv.URL, "http"), "--logs-dir", logs})
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "1 of 1") {
		t.Fatalf("err = %v\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "FAIL capital (ended by error)") ||
		!strings.Contains(out.String(), "connect_failed: ") {
		t.Fatalf("output:\n%s", out.String())
	}
	data, err := os.ReadFile(filepath.Join(logs, "capital.conversation.json")) //nolint:gosec // a test file
	if err != nil || strings.TrimSpace(string(data)) != "[]" {
		t.Fatalf("conversation file = %q, %v", data, err)
	}
}
