package eval

import (
	"maps"
	"slices"
)

// The fixed tables of names a scenario and a result use, for a tool that reads
// or writes them. Each call returns a fresh copy, so a caller can change what it
// gets without changing what the harness checks against.

// JudgeableEvents returns the events whose payloads carry bot-generated text a
// judge can evaluate.
func JudgeableEvents() []string { return slices.Clone(judgeableEvents) }

// FunctionCallEvents returns the events carrying a function call.
func FunctionCallEvents() []string { return slices.Clone(functionCallEvents) }

// MarkerKinds returns what a marker: may name on an llm_marker expectation.
func MarkerKinds() []string { return slices.Clone(markerKinds) }

// CheckFrom returns the values check_from: may take.
func CheckFrom() []string { return slices.Clone(checkFromSources) }

// FailureKinds returns the categories of AssertionFailure.Kind.
func FailureKinds() []string { return slices.Clone(failureKinds) }

// TurnStatuses returns the values ScriptTurnResult.Status takes.
func TurnStatuses() []string { return slices.Clone(turnStatuses) }

// SimulationEndings returns the values SimulationResult.EndedBy takes.
func SimulationEndings() []string { return slices.Clone(simulationEndings) }

// SimulationMeasures returns what a measured metric can measure.
func SimulationMeasures() []string { return slices.Clone(simulationMeasures) }

// ScenarioSuffixes returns the file name suffixes a scenario file is looked for
// under.
func ScenarioSuffixes() []string { return slices.Clone(scenarioSuffixes) }

// PipelineLogCategories returns the log categories, in the order a run's debug
// log lists them.
func PipelineLogCategories() []string { return slices.Clone(pipelineLogCategories) }

// PipelineLogLabels returns the heading of each category's section in a run's
// debug log.
func PipelineLogLabels() map[string]string { return maps.Clone(pipelineLogLabels) }
