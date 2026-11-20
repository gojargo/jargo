package eval

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
)

// The suite's results.jsonl: one record per finished run.

// recordArtifacts are the paths to a run's artifacts that exist.
type recordArtifacts struct {
	Log          string `json:"log"`
	EvalLog      string `json:"eval_log,omitempty"`
	DebugLog     string `json:"debug_log,omitempty"`
	Conversation string `json:"conversation,omitempty"`
	Recording    string `json:"recording,omitempty"`
}

// failureRecord is one failure of a scenario record.
type failureRecord struct {
	TurnIndex        int    `json:"turn_index"`
	ExpectationIndex int    `json:"expectation_index"`
	EventName        string `json:"event_name"`
	Kind             string `json:"kind"`
	Reason           string `json:"reason"`
}

// expectationRecord is what one expectation of a turn resolved to.
type expectationRecord struct {
	ExpectationIndex int    `json:"expectation_index"`
	EventName        string `json:"event_name"`
	Passed           bool   `json:"passed"`
	Matched          string `json:"matched"`
}

// turnRecord is one turn of a scenario record.
type turnRecord struct {
	TurnIndex    int                 `json:"turn_index"`
	Status       string              `json:"status"`
	DurationMS   int                 `json:"duration_ms"`
	Expectations []expectationRecord `json:"expectations"`
}

// scenarioRecord is the results.jsonl record of a scenario run.
type scenarioRecord struct {
	Bot        string             `json:"bot"`
	Name       string             `json:"name"`
	Scenario   string             `json:"scenario"`
	Kind       Kind               `json:"kind"`
	Attempt    int                `json:"attempt"`
	Passed     bool               `json:"passed"`
	Skipped    *string            `json:"skipped"`
	Error      *string            `json:"error"`
	DurationMS *int               `json:"duration_ms"`
	Failures   []failureRecord    `json:"failures"`
	Turns      []turnRecord       `json:"turns"`
	BotMetrics []BotMetricsReport `json:"bot_metrics"`
	Artifacts  recordArtifacts    `json:"artifacts"`
	EventsSeen *[]map[string]any  `json:"events_seen,omitempty"`
}

// verdictRecord is the judge's verdict on one bot turn of a metric.
type verdictRecord struct {
	Turn          int                `json:"turn"`
	Passed        bool               `json:"passed"`
	Verdict       string             `json:"verdict"`
	Confidence    *float64           `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Reason        string             `json:"reason"`
}

// metricRecord is one metric's score of a simulation record.
type metricRecord struct {
	Name        string          `json:"name"`
	Criterion   *string         `json:"criterion"`
	Score       *float64        `json:"score"`
	Passed      bool            `json:"passed"`
	MinScore    *float64        `json:"min_score"`
	Value       *float64        `json:"value"`
	Reason      string          `json:"reason"`
	FailureKind *string         `json:"failure_kind"`
	Verdicts    []verdictRecord `json:"verdicts"`
}

// simulationRecord is the results.jsonl record of a simulation run.
type simulationRecord struct {
	Bot           string             `json:"bot"`
	Name          string             `json:"name"`
	Scenario      string             `json:"scenario"`
	Kind          Kind               `json:"kind"`
	Attempt       int                `json:"attempt"`
	Success       string             `json:"success"`
	Passed        bool               `json:"passed"`
	Succeeded     bool               `json:"succeeded"`
	Error         *string            `json:"error"`
	FailureKind   *string            `json:"failure_kind"`
	Confidence    *float64           `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	EndedBy       string             `json:"ended_by"`
	Turns         int                `json:"turns"`
	Metrics       []metricRecord     `json:"metrics"`
	Reason        string             `json:"reason"`
	EndCall       map[string]any     `json:"end_call"`
	DurationMS    *int               `json:"duration_ms"`
	Messages      []map[string]any   `json:"messages"`
	BotMetrics    []BotMetricsReport `json:"bot_metrics"`
	Artifacts     recordArtifacts    `json:"artifacts"`
	EventsSeen    *[]map[string]any  `json:"events_seen,omitempty"`
}

// appendResult appends one JSON line describing a finished run.
//
// The record carries enough to compute pass rates and group failures by kind,
// plus paths to the artifacts, and bot_metrics, what the bot's processors
// measured during the run. events_seen is included only for runs that did
// not pass, since that is what a failure is diagnosed from. A write that fails
// is dropped: the record is a by-product of the run.
func appendResult(resultsPath string, run *SuiteRun, stem, logsDir, recordDir string) {
	artifacts := recordArtifacts{Log: filepath.Join(logsDir, stem+".log")}
	for suffix, field := range map[string]*string{
		".eval.log":          &artifacts.EvalLog,
		".debug.log":         &artifacts.DebugLog,
		".conversation.json": &artifacts.Conversation,
	} {
		if path := filepath.Join(logsDir, stem+suffix); exists(path) {
			*field = path
		}
	}
	if recordDir != "" {
		if path := filepath.Join(recordDir, stem+".wav"); exists(path) {
			artifacts.Recording = path
		}
	}
	var record any
	if run.RunKind() == KindSimulation {
		record = simulationRecordOf(run, artifacts)
	} else {
		record = scenarioRecordOf(run, artifacts)
	}
	var line bytes.Buffer
	enc := json.NewEncoder(&line)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(record); err != nil {
		return
	}
	f, err := os.OpenFile(resultsPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600) //nolint:gosec // operator-chosen path
	if err != nil {
		return
	}
	_, _ = f.Write(line.Bytes())
	_ = f.Close()
}

// optional is s, or nil when it is empty.
func optional(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// scenarioRecordOf is the results.jsonl record of a scenario run.
func scenarioRecordOf(run *SuiteRun, artifacts recordArtifacts) scenarioRecord {
	result := run.ScriptResult
	record := scenarioRecord{
		Bot:        run.Bot,
		Name:       run.Label(),
		Scenario:   run.Scenario,
		Kind:       run.RunKind(),
		Attempt:    run.attempt(),
		Error:      optional(run.Error),
		DurationMS: run.DurationMS,
		Failures:   []failureRecord{},
		Turns:      []turnRecord{},
		BotMetrics: []BotMetricsReport{},
		Artifacts:  artifacts,
	}
	if result == nil {
		return record
	}
	record.Passed = result.Passed()
	record.Skipped = optional(result.Skipped)
	for _, f := range result.Failures {
		record.Failures = append(record.Failures, failureRecord{
			TurnIndex: f.TurnIndex, ExpectationIndex: f.ExpectationIndex,
			EventName: f.EventName, Kind: f.Kind, Reason: f.Reason,
		})
	}
	for _, t := range result.Turns {
		turn := turnRecord{
			TurnIndex: t.TurnIndex, Status: t.Status, DurationMS: t.DurationMS, Expectations: []expectationRecord{},
		}
		for _, e := range t.Expectations {
			turn.Expectations = append(turn.Expectations, expectationRecord(e))
		}
		record.Turns = append(record.Turns, turn)
	}
	record.BotMetrics = nonNil(result.BotMetrics)
	if !record.Passed {
		seen := nonNil(result.EventsSeen)
		record.EventsSeen = &seen
	}
	return record
}

// simulationRecordOf is the results.jsonl record of a simulation run: how it
// ended, what the judge said and on what wording, each metric's score, the
// conversation, and for a failed run the bot's events.
func simulationRecordOf(run *SuiteRun, artifacts recordArtifacts) simulationRecord {
	result := run.SimulationResult
	record := simulationRecord{
		Bot:           run.Bot,
		Name:          run.Label(),
		Scenario:      run.Scenario,
		Kind:          run.RunKind(),
		Attempt:       run.attempt(),
		Error:         optional(run.Error),
		Probabilities: map[string]float64{},
		EndedBy:       SimulationEndedByError,
		Metrics:       []metricRecord{},
		DurationMS:    run.DurationMS,
		Messages:      []map[string]any{},
		BotMetrics:    []BotMetricsReport{},
		Artifacts:     artifacts,
	}
	if result == nil {
		return record
	}
	if record.Error == nil {
		record.Error = optional(result.Error)
	}
	record.Success = result.Success
	record.Passed = result.Passed()
	record.Succeeded = result.Succeeded
	record.FailureKind = optional(result.FailureKind)
	record.Confidence = result.Confidence
	if result.Probabilities != nil {
		record.Probabilities = result.Probabilities
	}
	record.EndedBy = result.EndedBy
	record.Turns = result.Turns
	for _, m := range result.Metrics {
		record.Metrics = append(record.Metrics, metricRecordOf(m))
	}
	record.Reason = result.Reason
	record.EndCall = result.EndCall
	record.Messages = nonNil(result.Messages)
	record.BotMetrics = nonNil(result.BotMetrics)
	if !record.Passed {
		seen := nonNil(result.EventsSeen)
		record.EventsSeen = &seen
	}
	return record
}

// metricRecordOf is one metric's score as a simulation record holds it.
func metricRecordOf(m SimulationMetricScore) metricRecord {
	record := metricRecord{
		Name:        m.Name,
		Criterion:   optional(m.Criterion),
		Score:       m.Score,
		Passed:      m.Passed,
		MinScore:    m.MinScore,
		Value:       m.Value,
		Reason:      m.Reason,
		FailureKind: optional(m.FailureKind),
		Verdicts:    []verdictRecord{},
	}
	for _, v := range m.Verdicts {
		probabilities := v.Probabilities
		if probabilities == nil {
			probabilities = map[string]float64{}
		}
		record.Verdicts = append(record.Verdicts, verdictRecord{
			Turn: v.Turn, Passed: v.Passed, Verdict: v.Verdict, Confidence: v.Confidence,
			Probabilities: probabilities, Reason: v.Reason,
		})
	}
	return record
}
