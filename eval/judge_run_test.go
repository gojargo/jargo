package eval_test

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/eval"
)

// The bot's two turns in the conversation the run tests judge.
const (
	firstTurn  = "Let me check. Done, six o'clock."
	secondTurn = "You're welcome."
)

// turnOptions are the answers a run's turn is judged by.
var turnOptions = []string{"meets", "fails", "not_applicable"} //nolint:gochecknoglobals // fixed test data

// converse feeds the judge a booking: a tool call, a turn of two segments, and
// a second turn.
func converse(j *eval.ClassifierJudge) {
	j.AddUserMessage("Book a table at six.")
	j.AddToolCall(`book({"time": "6pm"})`)
	j.AddAssistantMessage("Let me check.")
	j.AddAssistantMessage("Done, six o'clock.")
	j.AddUserMessage("Thanks.")
	j.AddAssistantMessage("You're welcome.")
}

// turnAnswer is a turn's answer: its probability of meeting the criterion, the
// rest failing it.
func turnAnswer(meets float64) map[string]float64 {
	return map[string]float64{"meets": meets, "fails": 1 - meets, "not_applicable": 0}
}

// turnResult is the choice a turn's probabilities make.
func turnResult(probabilities map[string]float64) classifier.ChoiceResult {
	best := ""
	for _, o := range turnOptions {
		if best == "" || probabilities[o] > probabilities[best] {
			best = o
		}
	}
	return classifier.ChoiceResult{Choice: best, Probabilities: probabilities, Confidence: probabilities[best]}
}

// results is what the fake classifier answers.
type results = map[string]classifier.Result

// responder answers as a function of the state asked about.
type responder = func(map[string]any, []classifier.Named[classifier.Question]) (results, error)

// runAnswers answers a run's questions: the goal's with the probability of yes,
// and each turn's by its reply text. A failing goal ("goal") or reply fails its
// question.
func runAnswers(
	goal float64, turns map[string]map[string]float64, failing string,
) responder {
	return func(state map[string]any, questions []classifier.Named[classifier.Question]) (results, error) {
		reply, ok := state["latest_bot_reply"].(string)
		if !ok {
			if failing == "goal" {
				return nil, fmt.Errorf("%w: goal question failed", classifier.ErrClassifier)
			}
			return map[string]classifier.Result{"answer": classifier.YesNoResult{Probability: goal}}, nil
		}
		if reply == failing {
			return nil, fmt.Errorf("%w: turn question failed", classifier.ErrClassifier)
		}
		out := map[string]classifier.Result{}
		for _, q := range questions {
			out[q.Name] = turnResult(turns[reply])
		}
		return out, nil
	}
}

func runJudge(
	t *testing.T, respond responder,
	explainer *explainerLLM, opts ...func(*eval.ClassifierJudgeConfig),
) (*eval.ClassifierJudge, *fakeClassifier) {
	t.Helper()
	c := newFakeClassifier()
	c.respond = respond
	return newJudge(t, c, explainer, opts...), c
}

var polite = []eval.Criterion{{Name: "polite", Criterion: "is polite"}} //nolint:gochecknoglobals // fixed test data

func verdictsOf(vs []eval.JudgeVerdict) []string {
	out := make([]string, 0, len(vs))
	for _, v := range vs {
		out = append(out, v.Verdict)
	}
	return out
}

func explainBelow(v float64) func(*eval.ClassifierJudgeConfig) {
	return func(cfg *eval.ClassifierJudgeConfig) { cfg.ExplainBelow = &v }
}

func TestARunWithAFailedQuestionIsNotCached(t *testing.T) {
	replies := map[string]map[string]float64{firstTurn: turnAnswer(0.8), secondTurn: turnAnswer(0.9)}
	down := runAnswers(0.9, replies, "goal")
	up := runAnswers(0.9, replies, "")
	var failing atomic.Bool
	failing.Store(true)
	j, c := runJudge(t, func(s map[string]any, q []classifier.Named[classifier.Question]) (results, error) {
		if failing.Load() {
			return down(s, q)
		}
		return up(s, q)
	}, nil)
	converse(j)

	failed := j.EvaluateRun(t.Context(), polite, "a table is booked")
	failing.Store(false)
	judged := j.EvaluateRun(t.Context(), polite, "a table is booked")
	asked := len(c.questions())
	again := j.EvaluateRun(t.Context(), polite, "a table is booked")

	if failed.Goal.Verdict != eval.VerdictNone || !judged.Goal.Passed() {
		t.Fatalf("goals = %v then %v, want none then yes", failed.Goal.Verdict, judged.Goal.Verdict)
	}
	// A run that decided everything is cached.
	if len(c.questions()) != asked || !reflect.DeepEqual(again, judged) {
		t.Fatalf("the decided run was asked again: %d questions, then %d", asked, len(c.questions()))
	}
}

func TestTheGoalIsAskedOverTheWholeConversation(t *testing.T) {
	j, c := runJudge(t, runAnswers(0.9, map[string]map[string]float64{
		firstTurn: turnAnswer(0.8), secondTurn: turnAnswer(0.3),
	}, ""), nil)
	converse(j)

	verdicts := j.EvaluateRun(t.Context(), polite, "a table is booked")

	if !verdicts.Goal.Passed() {
		t.Fatalf("goal = %+v", verdicts.Goal)
	}
	var goal *asked
	for _, a := range c.questions() {
		if _, ok := a.state["latest_bot_reply"]; !ok {
			goal = &a
		}
	}
	if goal == nil {
		t.Fatal("the goal was not asked")
	}
	got, _ := json.Marshal(goal.state["conversation"])
	want := `[{"speaker":"user","text":"Book a table at six."},` +
		`{"speaker":"tool","text":"book({\"time\": \"6pm\"})"},` +
		`{"speaker":"bot","text":"Let me check. Done, six o'clock.","turn":1},` +
		`{"speaker":"user","text":"Thanks."},` +
		`{"speaker":"bot","text":"You're welcome.","turn":2}]`
	if string(got) != want {
		t.Fatalf("conversation = %s\nwant %s", got, want)
	}
}

func TestEachTurnIsJudgedAfterOnlyTheConversationBeforeIt(t *testing.T) {
	j, c := runJudge(t, runAnswers(0.9, map[string]map[string]float64{
		firstTurn: turnAnswer(0.8), secondTurn: turnAnswer(0.3),
	}, ""), nil)
	converse(j)

	verdicts := j.EvaluateRun(t.Context(), polite, "a table is booked")

	if got := verdictsOf(verdicts.Turns["polite"]); !slices.Equal(got, []string{"yes", "no"}) {
		t.Fatalf("turns = %v, want yes, no", got)
	}
	byReply := map[string]asked{}
	for _, a := range c.questions() {
		if reply, ok := a.state["latest_bot_reply"].(string); ok {
			byReply[reply] = a
		}
	}
	if len(byReply) != 2 {
		t.Fatalf("asked about %d turns, want 2", len(byReply))
	}
	got, _ := json.Marshal(byReply[firstTurn].state["conversation"])
	if want := `[{"speaker":"user","text":"Book a table at six."},` +
		`{"speaker":"tool","text":"book({\"time\": \"6pm\"})"}]`; string(got) != want {
		t.Fatalf("first turn's conversation = %s", got)
	}
	second := byReply[secondTurn]
	if convo, _ := second.state["conversation"].([]any); len(convo) != 4 {
		t.Fatalf("second turn's conversation = %v, want 4 entries", second.state["conversation"])
	}
	if len(second.questions) != 1 || second.questions[0].Name != "polite" {
		t.Fatalf("second turn's questions = %v", second.questions)
	}
}

func TestATurnTheCriterionDoesNotApplyToPasses(t *testing.T) {
	j, c := runJudge(t, runAnswers(0.9, map[string]map[string]float64{
		firstTurn:  {"meets": 0.3, "fails": 0.1, "not_applicable": 0.6},
		secondTurn: {"meets": 0.05, "fails": 0.9, "not_applicable": 0.05},
	}, ""), nil)
	converse(j)

	verdicts := j.EvaluateRun(t.Context(),
		[]eval.Criterion{{Name: "apology", Criterion: "when a time is taken, apologizes"}}, "booked")

	turns := verdicts.Turns["apology"]
	if len(turns) != 2 {
		t.Fatalf("turns = %+v", turns)
	}
	first, second := turns[0], turns[1]
	if first.Verdict != eval.VerdictYes || !near(*first.Confidence, 0.9) {
		t.Fatalf("first = %+v, want yes at 0.9", first)
	}
	if second.Verdict != eval.VerdictNo || !near(*second.Confidence, 0.9) {
		t.Fatalf("second = %+v, want no at 0.9", second)
	}
	for _, a := range c.questions() {
		if _, ok := a.state["latest_bot_reply"]; ok {
			if got := optionNames(t, a.questions[0].Question); !slices.Equal(got, turnOptions) {
				t.Fatalf("options = %v, want %v", got, turnOptions)
			}
			return
		}
	}
	t.Fatal("no turn was asked about")
}

func TestARunWithNoCriteriaAsksOnlyTheGoal(t *testing.T) {
	j, c := runJudge(t, runAnswers(0.9, nil, ""), nil)
	converse(j)
	if v := j.EvaluateRun(t.Context(), nil, "a table is booked"); !v.Goal.Passed() {
		t.Fatalf("goal = %+v", v.Goal)
	}
	if n := len(c.questions()); n != 1 {
		t.Fatalf("asked %d questions, want 1", n)
	}
}

func TestAFailedQuestionBlanksOnlyItsOwnVerdicts(t *testing.T) {
	j, _ := runJudge(t, runAnswers(0.9, map[string]map[string]float64{secondTurn: turnAnswer(0.8)}, firstTurn), nil)
	converse(j)
	verdicts := j.EvaluateRun(t.Context(), polite, "booked")
	if !verdicts.Goal.Passed() {
		t.Fatalf("goal = %+v", verdicts.Goal)
	}
	if got := verdictsOf(verdicts.Turns["polite"]); !slices.Equal(got, []string{"none", "yes"}) {
		t.Fatalf("turns = %v, want none, yes", got)
	}
}

func TestAFailedGoalQuestionGivesNoGoalVerdict(t *testing.T) {
	j, _ := runJudge(t, runAnswers(0.9, map[string]map[string]float64{
		firstTurn: turnAnswer(0.8), secondTurn: turnAnswer(0.8),
	}, "goal"), nil)
	converse(j)
	verdicts := j.EvaluateRun(t.Context(), polite, "booked")
	if verdicts.Goal.Verdict != eval.VerdictNone {
		t.Fatalf("goal = %+v, want none", verdicts.Goal)
	}
	if got := verdictsOf(verdicts.Turns["polite"]); !slices.Equal(got, []string{"yes", "yes"}) {
		t.Fatalf("turns = %v, want yes, yes", got)
	}
}

func TestTheExplainerIsAskedOnlyAboutTheVerdictsThatNeedAReason(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"turns": {"polite": {"2": "curt"}}}`}}
	j, _ := runJudge(t, runAnswers(0.95, map[string]map[string]float64{
		firstTurn: turnAnswer(0.9), secondTurn: turnAnswer(0.1),
	}, ""), explainer)
	converse(j)

	verdicts := j.EvaluateRun(t.Context(), polite, "booked")

	calls := explainer.recorded()
	if len(calls) != 1 {
		t.Fatalf("the explainer was asked %d times, want once", len(calls))
	}
	ask := calls[0].ask
	_, toExplain, _ := strings.Cut(ask, "verdicts to explain")
	if !strings.Contains(ask, "- polite, bot turn 2: fails the criterion") ||
		strings.Contains(toExplain, "bot turn 1:") || strings.Contains(ask, "- goal:") {
		t.Fatalf("ask = %s", ask)
	}
	if verdicts.Goal.Reason != "P(yes)=0.95" {
		t.Fatalf("goal reason = %q", verdicts.Goal.Reason)
	}
	turns := verdicts.Turns["polite"]
	if turns[0].Reason != "P(meets)=0.90, P(fails)=0.10, P(not_applicable)=0.00" || turns[1].Reason != "curt" {
		t.Fatalf("reasons = %q, %q", turns[0].Reason, turns[1].Reason)
	}
}

func TestTheSchemaNamesExactlyTheReasonsAskedFor(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"turns": {"polite": {"2": "curt"}}}`}}
	j, _ := runJudge(t, runAnswers(0.95, map[string]map[string]float64{
		firstTurn: turnAnswer(0.9), secondTurn: turnAnswer(0.1),
	}, ""), explainer)
	converse(j)
	j.EvaluateRun(t.Context(), polite, "booked")

	var schema struct {
		Required   []string `json:"required"`
		Properties struct {
			Turns struct {
				Required   []string `json:"required"`
				Properties map[string]struct {
					Required             []string `json:"required"`
					AdditionalProperties bool     `json:"additionalProperties"` //nolint:tagliatelle // JSON Schema's own key
				} `json:"properties"`
			} `json:"turns"`
		} `json:"properties"`
	}
	if err := json.Unmarshal([]byte(explainer.recorded()[0].schema), &schema); err != nil {
		t.Fatal(err)
	}
	turns := schema.Properties.Turns
	politeSchema := turns.Properties["polite"]
	if !slices.Equal(schema.Required, []string{"turns"}) || !slices.Equal(turns.Required, []string{"polite"}) ||
		!slices.Equal(politeSchema.Required, []string{"2"}) || politeSchema.AdditionalProperties {
		t.Fatalf("schema = %s", explainer.recorded()[0].schema)
	}
}

func TestAnAnswerThatLeavesAReasonOutIsAskedAgain(t *testing.T) {
	explainer := &explainerLLM{replies: []string{
		`{"goal": "booked", "turns": {"polite": {}}}`,
		`{"goal": "booked", "turns": {"polite": {"2": "curt"}}}`,
	}}
	j, _ := runJudge(t, runAnswers(0.6, map[string]map[string]float64{
		firstTurn: turnAnswer(0.9), secondTurn: turnAnswer(0.1),
	}, ""), explainer)
	converse(j)
	verdicts := j.EvaluateRun(t.Context(), polite, "booked")
	if n := len(explainer.recorded()); n != 2 {
		t.Fatalf("the explainer was asked %d times, want twice", n)
	}
	if verdicts.Goal.Reason != "booked" || verdicts.Turns["polite"][1].Reason != "curt" {
		t.Fatalf("reasons = %q, %q", verdicts.Goal.Reason, verdicts.Turns["polite"][1].Reason)
	}
}

func TestAnUnsurePassIsExplainedAsTheOutcomeTheClassifierChose(t *testing.T) {
	// The second answer fails, and the first one's reason stands.
	explainer := &explainerLLM{replies: []string{`{"turns": {"polite": {"1": "thanked the caller"}}}`}}
	j, _ := runJudge(t, runAnswers(0.95, map[string]map[string]float64{
		firstTurn:  turnAnswer(0.6),
		secondTurn: {"meets": 0.3, "fails": 0.1, "not_applicable": 0.6},
	}, ""), explainer, explainBelow(0.95))
	converse(j)

	verdicts := j.EvaluateRun(t.Context(), polite, "booked")

	ask := explainer.recorded()[0].ask
	if !strings.Contains(ask, "- polite, bot turn 1: meets the criterion") ||
		!strings.Contains(ask, "- polite, bot turn 2: passes because the criterion does not apply to it") {
		t.Fatalf("ask = %s", ask)
	}
	turns := verdicts.Turns["polite"]
	if turns[0].Reason != "thanked the caller" {
		t.Fatalf("first reason = %q", turns[0].Reason)
	}
	// A reason not given leaves the probabilities.
	if !strings.HasPrefix(turns[1].Reason, "P(meets)=") {
		t.Fatalf("second reason = %q", turns[1].Reason)
	}
}

func TestExplainBelowAboveOneExplainsEveryVerdictOfARun(t *testing.T) {
	explainer := &explainerLLM{replies: []string{
		`{"goal": "booked", "turns": {"polite": {"1": "thanked the caller", "2": "warm goodbye"}}}`,
	}}
	j, _ := runJudge(t, runAnswers(0.95, map[string]map[string]float64{
		firstTurn: turnAnswer(0.9), secondTurn: turnAnswer(0.9),
	}, ""), explainer, explainBelow(2))
	converse(j)

	verdicts := j.EvaluateRun(t.Context(), polite, "booked")

	if !strings.Contains(explainer.recorded()[0].ask, "- goal: achieved") || verdicts.Goal.Reason != "booked" {
		t.Fatalf("goal = %+v", verdicts.Goal)
	}
	p := 0.95
	if !reflect.DeepEqual(verdicts.Goal.Probabilities, map[string]float64{"yes": p, "no": 1 - p}) {
		t.Fatalf("goal probabilities = %v", verdicts.Goal.Probabilities)
	}
	turns := verdicts.Turns["polite"]
	if turns[0].Reason != "thanked the caller" || turns[1].Reason != "warm goodbye" {
		t.Fatalf("reasons = %q, %q", turns[0].Reason, turns[1].Reason)
	}
}

func TestARunIsExplainedFromTheWholeTranscript(t *testing.T) {
	explainer := &explainerLLM{replies: []string{`{"goal": "never booked"}`}}
	j, _ := runJudge(t, func(state map[string]any, questions []classifier.Named[classifier.Question]) (results, error) {
		if _, ok := state["latest_bot_reply"]; !ok {
			return map[string]classifier.Result{"answer": classifier.YesNoResult{Probability: 0.1}}, nil
		}
		out := map[string]classifier.Result{}
		for _, q := range questions {
			out[q.Name] = turnResult(turnAnswer(0.9))
		}
		return out, nil
	}, explainer)
	j.AddUserMessage("Book a table at six.")
	j.AddToolCall(`book({"time": "6pm"})`)
	j.AddAssistantMessage("Sure.")

	verdicts := j.EvaluateRun(t.Context(), polite, "a table is booked")

	if verdicts.Goal.Verdict != eval.VerdictNo || verdicts.Goal.Reason != "never booked" {
		t.Fatalf("goal = %+v", verdicts.Goal)
	}
	c := explainer.recorded()[0]
	for _, want := range []string{
		"User: Book a table at six.", `[tool call] book({"time": "6pm"})`, "Bot turn 1: Sure.", "- goal: not achieved",
	} {
		if !strings.Contains(c.ask, want) {
			t.Errorf("ask lacks %q:\n%s", want, c.ask)
		}
	}
	if !strings.HasPrefix(c.instruction, "You explain the verdicts a judge gave") {
		t.Fatalf("instruction = %q", c.instruction)
	}
}
