package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"sync"

	"github.com/gojargo/jargo/eval"
	"github.com/spf13/cobra"
)

// `jargo eval judge`: judge saved conversations against a simulation or a
// rubric.

// Judge command errors.
var (
	// errPickOne is a scenario file holding several simulations and rubrics
	// with none picked.
	errPickOne = errors.New("pick one of the file's simulations and rubrics")
	// errNoSuchJudged is a -s naming nothing the file holds.
	errNoSuchJudged = errors.New("no simulation or rubric")
	// errConversationsFailed is a judge run with a conversation that did not
	// pass.
	errConversationsFailed = errors.New("conversations did not pass")
)

// judgeFlags are what `jargo eval judge` is given.
type judgeFlags struct {
	scenario    string
	output      string
	concurrency int
	verbose     bool
}

// evalJudgeCmd is `jargo eval judge`: judge conversation files against a
// simulation's or a rubric's success and criterion metrics.
func evalJudgeCmd() *cobra.Command {
	var f judgeFlags
	cmd := &cobra.Command{
		Use:   "judge <scenario.yaml> <conversation.json>...",
		Short: "Judge saved conversations against a simulation's or a rubric's success and criterion metrics",
		Long: "Judge saved conversations against a simulation's or a rubric's success and criterion metrics.\n\n" +
			"Each conversation gets the verdicts a simulation's own run would, with a\n" +
			"reason for every one, passes included. A rubric judges a recorded call the\n" +
			"same way, with no simulation behind it. Measured metrics are left out, since\n" +
			"they come from a live run. A conversation file is a run's\n" +
			"<stem>.conversation.json, or the same format saved from a real call. Exits\n" +
			"non-zero when a conversation did not pass.",
		Args: cobra.MinimumNArgs(2),
	}
	cmd.Flags().StringVarP(&f.scenario, "scenario", "s", "",
		"the simulation or rubric to judge with when the file holds several, with or without "+
			"the file's name (book_table/available or available)")
	cmd.Flags().StringVarP(&f.output, "output", "o", "judged.jsonl", "JSONL file to write, one line per file")
	cmd.Flags().IntVarP(&f.concurrency, "concurrency", "c", 4, "how many conversations to judge at once")
	cmd.Flags().BoolVarP(&f.verbose, "verbose", "v", false, "print the judge's reason and each metric's score")
	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		restore := eval.QuietLogs()
		defer restore()
		simulation, err := judgedScenario(args[0], f.scenario)
		if err != nil {
			return err
		}
		out := cmd.OutOrStdout()
		d := display{out: out, color: isTerminal(out)}
		results, err := judgeAll(cmd.Context(), d, simulation, args[1:], f)
		if err != nil {
			return err
		}
		passed := 0
		for _, r := range results {
			if r.Passed() {
				passed++
			}
		}
		tally := fmt.Sprintf("%d/%d passed", passed, len(results))
		if passed == len(results) {
			tally = d.green(tally)
		} else {
			tally = d.red(tally)
		}
		d.printf("\n  %s\n  results: %s\n\n", tally, f.output)
		if passed != len(results) {
			return fmt.Errorf("%w: %d of %d", errConversationsFailed, len(results)-passed, len(results))
		}
		return nil
	}
	return cmd
}

// judgedScenario is the simulation or rubric in path to judge with: the one
// called name, or the file's only one.
func judgedScenario(path, name string) (eval.JudgedScenario, error) {
	file, err := eval.LoadFile(path)
	if err != nil {
		return nil, fmt.Errorf("invalid value for SCENARIO_PATH: %w", err)
	}
	var judging []eval.JudgedScenario
	var names []string
	for _, loaded := range file.Scenarios {
		if judged, ok := loaded.(eval.JudgedScenario); ok {
			judging = append(judging, judged)
			names = append(names, judged.ScenarioName())
		}
	}
	listed := strings.Join(names, ", ")
	if listed == "" {
		listed = "none"
	}
	if name == "" && len(judging) != 1 {
		return nil, fmt.Errorf("invalid value for -s: %w: %s", errPickOne, listed)
	}
	for _, s := range judging {
		if name == "" || s.ScenarioName() == name || s.ScenarioName() == file.Name+"/"+name {
			return s, nil
		}
	}
	return nil, fmt.Errorf("invalid value for -s: %w '%s'; the file has: %s", errNoSuchJudged, name, listed)
}

// judgedSuccess is the success a simulation or rubric judges the goal on.
func judgedSuccess(s eval.JudgedScenario) string {
	switch x := s.(type) {
	case *eval.SimulationScenario:
		return x.Success
	case *eval.Rubric:
		return x.Success
	}
	return ""
}

// judgeConversation judges one conversation file, with a judge of its own that
// explains every verdict. A file that cannot be read or judged is the result's
// error.
func judgeConversation(ctx context.Context, simulation eval.JudgedScenario, path string) *eval.SimulationResult {
	failed := func(err error) *eval.SimulationResult {
		return &eval.SimulationResult{
			SimulationName: simulation.ScenarioName(),
			Error:          "error: " + err.Error(),
			Success:        judgedSuccess(simulation),
		}
	}
	data, err := os.ReadFile(path) //nolint:gosec // a conversation file the operator named
	if err != nil {
		return failed(err)
	}
	var messages []map[string]any
	if err = json.Unmarshal(data, &messages); err != nil {
		return failed(err)
	}
	everything := math.Inf(1)
	judgement, err := eval.NewSimulationJudge(simulation, nil, &everything, nil).Judge(ctx, messages)
	if err != nil {
		return failed(err)
	}
	goal := judgement.Goal
	result := &eval.SimulationResult{
		SimulationName: simulation.ScenarioName(),
		Succeeded:      goal.Verdict == eval.VerdictYes,
		Reason:         goal.Reason,
		Confidence:     goal.Confidence,
		Probabilities:  goal.Probabilities,
		Metrics:        judgement.Metrics,
		Messages:       []map[string]any{},
		Conversation:   messages,
		Success:        judgedSuccess(simulation),
	}
	if goal.Verdict == eval.VerdictNone {
		result.Error = "judge gave no verdict on the goal: " + goal.Reason
	}
	for _, m := range messages {
		role := m["role"]
		if role == eval.ConversationRoleUser || role == eval.ConversationRoleAssistant {
			result.Messages = append(result.Messages, map[string]any{"role": role, "content": m["content"]})
		}
		if role == eval.ConversationRoleUser {
			result.Turns++
		}
	}
	return result
}

// judgedVerdict is one turn's verdict in a judged conversation's line.
type judgedVerdict struct {
	Turn          int                `json:"turn"`
	Passed        bool               `json:"passed"`
	Verdict       string             `json:"verdict"`
	Confidence    *float64           `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Reason        string             `json:"reason"`
	Text          string             `json:"text"`
}

// judgedMetric is one metric's outcome in a judged conversation's line.
type judgedMetric struct {
	Name        string          `json:"name"`
	Criterion   *string         `json:"criterion"`
	Score       *float64        `json:"score"`
	Passed      bool            `json:"passed"`
	MinScore    *float64        `json:"min_score"`
	Reason      string          `json:"reason"`
	FailureKind *string         `json:"failure_kind"`
	Verdicts    []judgedVerdict `json:"verdicts"`
}

// judgementRecord is the JSONL line of a judged conversation.
type judgementRecord struct {
	Conversation  string             `json:"conversation"`
	Scenario      string             `json:"scenario"`
	Success       string             `json:"success"`
	Passed        bool               `json:"passed"`
	Succeeded     bool               `json:"succeeded"`
	Error         *string            `json:"error"`
	Confidence    *float64           `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Reason        string             `json:"reason"`
	Metrics       []judgedMetric     `json:"metrics"`
}

// optionalString is s, or nil when it is empty.
func optionalString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// nonNilProbabilities is p, or an empty mapping when it is nil.
func nonNilProbabilities(p map[string]float64) map[string]float64 {
	if p == nil {
		return map[string]float64{}
	}
	return p
}

// botTurns are what the bot said in each of its turns, in order.
func botTurns(result *eval.SimulationResult) []string {
	var turns []string
	for _, m := range result.Messages {
		if m["role"] == eval.ConversationRoleAssistant {
			content, _ := m["content"].(string)
			turns = append(turns, content)
		}
	}
	return turns
}

// judgementRecordOf is the line of a judged conversation: the goal's verdict
// and every metric's, each with the wording it was judged on, and each turn's
// with its reason and what the bot said.
func judgementRecordOf(path string, result *eval.SimulationResult) judgementRecord {
	said := botTurns(result)
	record := judgementRecord{
		Conversation:  path,
		Scenario:      result.SimulationName,
		Success:       result.Success,
		Passed:        result.Passed(),
		Succeeded:     result.Succeeded,
		Error:         optionalString(result.Error),
		Confidence:    result.Confidence,
		Probabilities: nonNilProbabilities(result.Probabilities),
		Reason:        result.Reason,
		Metrics:       []judgedMetric{},
	}
	for _, m := range result.Metrics {
		metric := judgedMetric{
			Name: m.Name, Criterion: optionalString(m.Criterion), Score: m.Score, Passed: m.Passed,
			MinScore: m.MinScore, Reason: m.Reason, FailureKind: optionalString(m.FailureKind),
			Verdicts: []judgedVerdict{},
		}
		for _, v := range m.Verdicts {
			text := ""
			if v.Turn >= 1 && v.Turn <= len(said) {
				text = said[v.Turn-1]
			}
			metric.Verdicts = append(metric.Verdicts, judgedVerdict{
				Turn: v.Turn, Passed: v.Passed, Verdict: v.Verdict, Confidence: v.Confidence,
				Probabilities: nonNilProbabilities(v.Probabilities), Reason: v.Reason, Text: text,
			})
		}
		record.Metrics = append(record.Metrics, metric)
	}
	return record
}

// printJudgement prints a judged conversation's verdict line and, verbose, the
// judge's reasons under it.
func (d display) printJudgement(path string, result *eval.SimulationResult, verbose bool) {
	if result.Passed() {
		d.printf("  %s %s\n", d.green("✓"), path)
	} else {
		d.printf("  %s %s  %s\n", d.red("✗"), path, d.dim(result.Failure()))
	}
	if verbose {
		d.simulationDetail(result)
	}
}

// simulationDetail prints a simulation's verdict detail (verbose mode): the
// judge's reason, each metric's score and reason, and the persona's own claim
// from its end_call.
func (d display) simulationDetail(result *eval.SimulationResult) {
	if result.Error != "" {
		return
	}
	// A blank line sets each section apart, and one more before the verdict
	// line.
	d.printf("\n    %s %s\n", d.paint("judge:", "1"), result.Reason)
	if len(result.Metrics) > 0 {
		d.printf("\n    %s\n", d.paint("metrics:", "1"))
		said := botTurns(result)
		for _, m := range result.Metrics {
			d.metricDetail(m, said)
		}
	}
	if result.EndCall != nil {
		claim := d.red("gave up")
		if success, _ := result.EndCall["success"].(bool); success {
			claim = d.green("succeeded")
		}
		reason, _ := result.EndCall["reason"].(string)
		d.printf("\n    %s %s: %s\n", d.paint("persona:", "1"), claim, reason)
	}
	d.printf("\n")
}

// metricDetail prints one metric's score and, for each turn it failed, what the
// bot said and why it fell short. said is what the bot said in each turn.
func (d display) metricDetail(m eval.SimulationMetricScore, said []string) {
	score := "unscored"
	if m.Score != nil {
		score = fmt.Sprintf("%.2f", *m.Score)
	}
	if m.Passed {
		score = d.green(score)
	} else {
		score = d.red(score)
	}
	bound := ""
	if m.MinScore != nil {
		bound = fmt.Sprintf(" (min %.2f)", *m.MinScore)
	}
	var failed []eval.SimulationTurnVerdict
	for _, v := range m.Verdicts {
		if !v.Passed {
			failed = append(failed, v)
		}
	}
	summary := m.Reason
	if len(m.Verdicts) > 0 {
		summary = fmt.Sprintf("%d/%d turns", len(m.Verdicts)-len(failed), len(m.Verdicts))
	}
	d.printf("      %s %s%s%s\n", d.paint(m.Name+":", "36"), score, d.dim(bound), d.dim(" | "+summary))
	for _, v := range failed {
		text := ""
		if v.Turn >= 1 && v.Turn <= len(said) {
			text = said[v.Turn-1]
		}
		d.printf("        %s %s %s %s\n", d.red("✗"), d.dim(fmt.Sprintf("turn %d:", v.Turn)),
			strings.Join(strings.Fields(text), " "), d.dim(": "+v.Reason))
	}
}

// judgeAll judges every conversation, f.concurrency at a time, writing each line
// as it is judged.
func judgeAll(
	ctx context.Context, d display, simulation eval.JudgedScenario, paths []string, f judgeFlags,
) ([]*eval.SimulationResult, error) {
	file, err := os.Create(f.output)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	var (
		mu      sync.Mutex
		results []*eval.SimulationResult
		wg      sync.WaitGroup
	)
	slots := make(chan struct{}, max(1, f.concurrency))
	for _, path := range paths {
		wg.Go(func() {
			slots <- struct{}{}
			result := judgeConversation(ctx, simulation, path)
			<-slots
			mu.Lock()
			defer mu.Unlock()
			writeJSONLine(file, judgementRecordOf(path, result))
			d.printJudgement(path, result, f.verbose)
			results = append(results, result)
		})
	}
	wg.Wait()
	return results, nil
}

// writeJSONLine writes a judged conversation's line, without escaping HTML.
func writeJSONLine(w io.Writer, record judgementRecord) {
	var line bytes.Buffer
	enc := json.NewEncoder(&line)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(record); err == nil {
		_, _ = w.Write(line.Bytes())
	}
}
