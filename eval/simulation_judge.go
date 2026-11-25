package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
)

// Judges a conversation against a simulation's, or a rubric's, success and
// criterion metrics.
//
// The simulation driver judges a run with it when the run ends, and `jargo eval
// judge` judges saved conversation files with it, so a conversation gets the
// same verdicts either way. With a Rubric, it judges a conversation no
// simulation produced, such as a production call. Measured metrics are not its
// part: they come from a live run, not from what was said.

// JudgedScenario is what a conversation is judged against: a
// *SimulationScenario or a *Rubric.
type JudgedScenario interface {
	LoadedScenario
	// judged is the success the goal is judged on, the metrics, and the
	// judge.eval: block a judge is built from.
	judged() (success string, metrics []SimulationMetric, judge JudgeEvalConfig)
}

func (s *SimulationScenario) judged() (string, []SimulationMetric, JudgeEvalConfig) {
	return s.Success, s.Metrics, s.Judge
}

func (r *Rubric) judged() (string, []SimulationMetric, JudgeEvalConfig) {
	return r.Success, r.Metrics, r.Judge
}

// RunJudge is the judge a whole conversation is put to: told the conversation,
// it decides the goal and every bot turn on every criterion. ClassifierJudge is one.
// A run judge that also has a Close(context.Context) error method is closed by
// whoever built it.
type RunJudge interface {
	// AddUserMessage records a user turn in the conversation.
	AddUserMessage(text string)
	// AddAssistantMessage appends a segment of the bot's current reply.
	AddAssistantMessage(text string)
	// AddToolCall records a tool call the bot made, on one line.
	AddToolCall(text string)
	// EvaluateRun judges the whole conversation: the goal, and every bot turn
	// on every criterion.
	EvaluateRun(ctx context.Context, criteria []Criterion, success string) RunVerdicts
}

// SimulationJudgement is what the judge decided about one conversation.
type SimulationJudgement struct {
	// Goal is the verdict on the success: yes, no, or none when the judge gave
	// none.
	Goal JudgeVerdict
	// Metrics are the criterion metrics' outcomes, in the simulation's order.
	Metrics []SimulationMetricScore
}

// SimulationJudge judges conversations against a simulation or a rubric.
//
// Each conversation is judged by a judge built from the simulation's judge:
// block, closed once the conversation is judged. A judge passed in is used
// instead and left open for its owner to close; it keeps the conversation it is
// told, so it judges one conversation.
//
//	file, _ := eval.LoadFile("book_table.yaml")
//	scenario, _ := file.Scenario("book_table/available")
//	var messages []map[string]any
//	data, _ := os.ReadFile("run.conversation.json")
//	_ = json.Unmarshal(data, &messages)
//	judgement, err := eval.NewSimulationJudge(scenario.(eval.JudgedScenario), nil, nil, nil).Judge(ctx, messages)
//
// A production call, recorded by ConversationObserver, is judged against a
// rubric the same way, with the observer's conversation as the messages.
type SimulationJudge struct {
	simulation   JudgedScenario
	judge        RunJudge
	explainBelow *float64
	trace        *Trace
	// newJudge builds a judge from a judge.eval: block; a test stands in for
	// it.
	newJudge func(cfg *JudgeEvalConfig) (RunJudge, error)
}

// NewSimulationJudge builds the judge of simulation, a *SimulationScenario or a
// *Rubric.
//
// judge is the judge to ask, which should not have seen any conversation yet;
// nil builds one for each conversation. explainBelow is, for the judges it
// builds, the confidence under which a yes is explained too (above 1, every
// verdict is); nil keeps the simulation's judge: block. trace is the run's
// trace, to log each verdict in, or nil.
func NewSimulationJudge(
	simulation JudgedScenario, judge RunJudge, explainBelow *float64, trace *Trace,
) *SimulationJudge {
	return &SimulationJudge{
		simulation:   simulation,
		judge:        judge,
		explainBelow: explainBelow,
		trace:        trace,
		newJudge: func(cfg *JudgeEvalConfig) (RunJudge, error) {
			return NewClassifierJudgeFromConfig(cfg)
		},
	}
}

// callCanceledSuffix ends the line that tells the judge a call was canceled.
const callCanceledSuffix = " was cancelled" //nolint:misspell // the wording the judge has always read

// CallLines is a tool_call message as the judge reads it: the call, and a line
// more when it was canceled.
func CallLines(message map[string]any) []string {
	name, _ := message[keyName].(string)
	call := name + "()"
	if args, _ := message[keyArguments].(map[string]any); len(args) > 0 {
		call = name + "(" + pyJSON(args) + ")"
	}
	lines := []string{call}
	if canceled, _ := message[keyCanceled].(bool); canceled {
		lines = append(lines, name+callCanceledSuffix)
	}
	return lines
}

// Judge judges a conversation, its messages as a conversation file holds them
// (see Conversation), and gives the verdict on the goal and the criterion
// metrics' outcomes.
//
// It fails only when the judge it builds cannot be built.
func (j *SimulationJudge) Judge(ctx context.Context, messages []map[string]any) (SimulationJudgement, error) {
	if j.judge != nil {
		return j.ask(ctx, j.judge, messages), nil
	}
	judge, err := j.buildJudge()
	if err != nil {
		return SimulationJudgement{}, err
	}
	defer closeJudge(context.WithoutCancel(ctx), judge)
	return j.ask(ctx, judge, messages), nil
}

// buildJudge is a judge from the simulation's judge: block.
func (j *SimulationJudge) buildJudge() (RunJudge, error) {
	_, _, cfg := j.simulation.judged()
	if j.explainBelow != nil {
		explainBelow := *j.explainBelow
		cfg.ExplainBelow = &explainBelow
	}
	return j.newJudge(&cfg)
}

// ask tells the judge the conversation and asks it about the goal and the
// criteria.
func (j *SimulationJudge) ask(ctx context.Context, judge RunJudge, messages []map[string]any) SimulationJudgement {
	j.tellJudge(judge, messages)
	success, metrics, _ := j.simulation.judged()
	var criteria []Criterion
	for _, m := range metrics {
		if m.Criterion != "" {
			criteria = append(criteria, Criterion{Name: m.Name, Criterion: m.Criterion})
		}
	}
	judged := judge.EvaluateRun(withEvalPipeline(ctx, PipelineJudge), criteria, success)
	var scores []SimulationMetricScore
	for _, m := range metrics {
		if m.Criterion != "" {
			scores = append(scores, j.score(m, judged.Turns[m.Name]))
		}
	}
	return SimulationJudgement{Goal: judged.Goal, Metrics: scores}
}

// log logs a line in the run's trace, when there is one.
func (j *SimulationJudge) log(line string) {
	if j.trace != nil {
		j.trace.Log(line)
	}
}

// tellJudge gives the judge the conversation, each tool call where it was made.
func (j *SimulationJudge) tellJudge(judge RunJudge, messages []map[string]any) {
	var calls []string
	for _, m := range messages {
		content, _ := m[keyContent].(string)
		switch m[keyRole] {
		case ConversationRoleUser:
			judge.AddUserMessage(content)
		case ConversationRoleAssistant:
			judge.AddAssistantMessage(content)
		case ConversationRoleToolCall:
			for _, line := range CallLines(m) {
				judge.AddToolCall(line)
				calls = append(calls, line)
			}
		}
	}
	if len(calls) > 0 {
		j.log("judge: the bot's tool calls: " + strings.Join(calls, "; "))
	}
}

// score is a metric's score over its turn verdicts: the share of turns that
// passed.
//
// A metric that fell short only on turns the judge left unanswered is judge
// trouble, and its failure kind says so.
func (j *SimulationJudge) score(metric SimulationMetric, judged []JudgeVerdict) SimulationMetricScore {
	verdicts := make([]SimulationTurnVerdict, 0, len(judged))
	for i, v := range judged {
		verdicts = append(verdicts, SimulationTurnVerdict{
			Turn:          i + 1,
			Passed:        v.Verdict == VerdictYes,
			Reason:        v.Reason,
			Verdict:       v.Verdict,
			Confidence:    v.Confidence,
			Probabilities: v.Probabilities,
		})
	}
	for _, v := range verdicts {
		j.log(fmt.Sprintf("judge: turn %d %s %s: %s", v.Turn, metric.Name, v.Verdict, v.Reason))
	}
	score, reason, passed, failureKind := scoreVerdicts(metric, verdicts)
	scored := unscored
	if score != nil {
		scored = fmt.Sprintf("%.2f", *score)
	}
	below := ""
	if !passed {
		below = fmt.Sprintf(" (below %.2f)", *metric.MinScore)
	}
	j.log(fmt.Sprintf("judge: %s = %s%s: %s", metric.Name, scored, below, reason))
	return SimulationMetricScore{
		Name:        metric.Name,
		Score:       score,
		Passed:      passed,
		Reason:      reason,
		MinScore:    metric.MinScore,
		Verdicts:    verdicts,
		FailureKind: failureKind,
		Criterion:   metric.Criterion,
	}
}

// scoreVerdicts is a metric's score over its turn verdicts, nil with none, what
// it rests on, whether it let the run pass, and how it failed when it did not.
func scoreVerdicts(
	metric SimulationMetric, verdicts []SimulationTurnVerdict,
) (score *float64, reason string, passed bool, failureKind string) {
	if len(verdicts) == 0 {
		return nil, "no bot turn to judge", true, ""
	}
	var failed []SimulationTurnVerdict
	for _, v := range verdicts {
		if !v.Passed {
			failed = append(failed, v)
		}
	}
	share := float64(len(verdicts)-len(failed)) / float64(len(verdicts))
	reason = fmt.Sprintf("all %d turn(s)", len(verdicts))
	if len(failed) > 0 {
		parts := make([]string, 0, len(failed))
		for _, v := range failed {
			parts = append(parts, fmt.Sprintf("turn %d: %s", v.Turn, v.Reason))
		}
		reason = strings.Join(parts, "; ")
	}
	passed = metric.MinScore == nil || share >= *metric.MinScore
	if !passed {
		failureKind = FailureJudgeNoVerdict
		if slices.ContainsFunc(failed, func(v SimulationTurnVerdict) bool { return v.Verdict == VerdictNo }) {
			failureKind = FailureJudgeNo
		}
	}
	return &share, reason, passed, failureKind
}

// pyJSON is v as JSON in the default layout of the conversation's tool call
// lines: a space after each comma and colon, keys in order, and anything
// outside ASCII escaped.
func pyJSON(v any) string {
	switch x := v.(type) {
	case map[string]any:
		parts := make([]string, 0, len(x))
		for _, k := range slices.Sorted(maps.Keys(x)) {
			parts = append(parts, pyJSONString(k)+": "+pyJSON(x[k]))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case []any:
		parts := make([]string, 0, len(x))
		for _, item := range x {
			parts = append(parts, pyJSON(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case string:
		return pyJSONString(x)
	default:
		data, err := json.Marshal(x)
		if err != nil {
			return "null"
		}
		return string(data)
	}
}

// pyJSONString is s as a JSON string with everything outside ASCII escaped.
func pyJSONString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"':
			b.WriteString(`\"`)
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\b':
			b.WriteString(`\b`)
		case r == '\f':
			b.WriteString(`\f`)
		case r < 0x20 || r > 0x7f:
			if r > 0xffff {
				r -= 0x10000
				b.WriteString(`\u` + hex4(0xd800+(r>>10)) + `\u` + hex4(0xdc00+(r&0x3ff)))
			} else {
				b.WriteString(`\u` + hex4(r))
			}
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// hex4 is r as four lowercase hex digits.
func hex4(r rune) string {
	s := strconv.FormatInt(int64(r), 16)
	return strings.Repeat("0", 4-len(s)) + s
}
