package eval

import (
	"reflect"
	"testing"
)

// asking is the turn verdicts a run asks the explainer about, by turn number.
func asking(name string, turns ...int) askedCriterion {
	c := askedCriterion{name: name}
	for _, n := range turns {
		c.turns = append(c.turns, askedTurn{number: n, verdict: JudgeVerdict{Verdict: VerdictNo}})
	}
	return c
}

func TestParseRunReasons(t *testing.T) {
	t.Run("the goal's reason and each asked turn's", func(t *testing.T) {
		out := parseRunReasons(`{"goal": "Berlin was named.", `+
			`"turns": {"Politeness": {"2": "curt"}, "brevity": {"1": "short"}}}`,
			[]askedCriterion{asking("politeness", 2), asking("brevity", 1)})
		want := map[string]map[int]string{"politeness": {2: "curt"}, "brevity": {1: "short"}}
		if out.goal != "Berlin was named." || !reflect.DeepEqual(out.turns, want) {
			t.Fatalf("got %+v", out)
		}
	})
	t.Run("a turn is the number in its key", func(t *testing.T) {
		out := parseRunReasons(`{"turns": {"politeness": {"bot turn 1": "warm", "Turn 2": "curt"}}}`,
			[]askedCriterion{asking("politeness", 1, 2)})
		want := map[string]map[int]string{"politeness": {1: "warm", 2: "curt"}}
		if !reflect.DeepEqual(out.turns, want) {
			t.Fatalf("got %+v", out.turns)
		}
	})
	t.Run("a reason not given is empty", func(t *testing.T) {
		out := parseRunReasons("```json\n{\"turns\": {\"politeness\": {\"1\": \"warm\"}}}\n```",
			[]askedCriterion{asking("politeness", 1, 2)})
		want := map[string]map[int]string{"politeness": {1: "warm", 2: ""}}
		if out.goal != "" || !reflect.DeepEqual(out.turns, want) {
			t.Fatalf("got %+v", out)
		}
	})
	t.Run("no json gives no reasons", func(t *testing.T) {
		out := parseRunReasons("no json here", []askedCriterion{asking("politeness", 1)})
		want := map[string]map[int]string{"politeness": {1: ""}}
		if out.goal != "" || !reflect.DeepEqual(out.turns, want) {
			t.Fatalf("got %+v", out)
		}
	})
}
