package eval

import (
	"slices"
	"testing"
)

// TestParseReason covers the reason read out of each shape of explainer
// answer: the JSON it is asked for, JSON wrapped in prose or fences, and a
// plain answer, which is the reason itself.
func TestParseReason(t *testing.T) {
	cases := map[string]struct {
		out  string
		want string
	}{
		"the json reason":     {out: `{"reason": "It mentions weather."}`, want: "It mentions weather."},
		"fenced json":         {out: "```json\n{\"reason\": \"ok\"}\n```", want: "ok"},
		"surrounded json":     {out: "Sure: {\"reason\": \"ok\"}\n\nLet me know if you need more!", want: "ok"},
		"a plain answer":      {out: "  It never mentions rain.\n", want: "It never mentions rain."},
		"fenced without lang": {out: "```\n{\"reason\": \"wrong\"}\n```", want: "wrong"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := parseReason(tc.out); got != tc.want {
				t.Fatalf("reason = %q, want %q", got, tc.want)
			}
		})
	}
}

// A turn in which nothing was said is not a turn: a blank user message or reply
// segment is left out of the conversation the judge keeps, and a tool call
// stays in it.
func TestATurnInWhichNothingWasSaidIsNotATurn(t *testing.T) {
	j := &EvalJudge{}
	j.AddUserMessage("Hi.")
	j.AddToolCall("lookup({})")
	j.AddAssistantMessage("  ")
	j.AddUserMessage("Hello?")
	j.AddUserMessage("")

	_, all := j.spoken()
	roles := make([]string, 0, len(all))
	for _, e := range all {
		roles = append(roles, e.role)
	}
	if want := []string{roleUser, roleTool, roleUser}; !slices.Equal(roles, want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
}
