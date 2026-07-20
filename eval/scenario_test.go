package eval_test

import (
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

func TestLoadValidScenario(t *testing.T) {
	path := writeScenario(t, `
name: greeting
turns:
  - user: "hello there"
    expect:
      - event: llm_started
      - event: llm_response
        text_contains: "hi"
        within_ms: 5000
      - event: llm_response
        judge: "greets the user warmly"
  - user: "what's the weather in Paris?"
    expect:
      - event: function_call
        function: get_weather
`)

	s, err := eval.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.Name != "greeting" || len(s.Turns) != 2 {
		t.Fatalf("unexpected scenario: %+v", s)
	}
	if s.Turns[0].User != "hello there" || len(s.Turns[0].Expect) != 3 {
		t.Fatalf("unexpected turn 1: %+v", s.Turns[0])
	}
	if s.Turns[0].Expect[1].TextContains != "hi" || s.Turns[0].Expect[1].WithinMS != 5000 {
		t.Fatalf("unexpected expectation: %+v", s.Turns[0].Expect[1])
	}
	if s.Turns[1].Expect[0].Event != eval.EventFunctionCall || s.Turns[1].Expect[0].Function != "get_weather" {
		t.Fatalf("unexpected function_call expectation: %+v", s.Turns[1].Expect[0])
	}
}

func TestLoadRejectsInvalid(t *testing.T) {
	cases := map[string]struct {
		body string
		want string
	}{
		"no name": {
			body: "turns:\n  - user: hi\n    expect:\n      - event: llm_started\n",
			want: "no name",
		},
		"no turns": {
			body: "name: empty\n",
			want: "no turns",
		},
		"empty user": {
			body: "name: x\nturns:\n  - user: \"\"\n    expect:\n      - event: llm_started\n",
			want: "no user input",
		},
		"no expectations": {
			body: "name: x\nturns:\n  - user: hi\n    expect: []\n",
			want: "no expectations",
		},
		"unknown event": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n      - event: teleport\n",
			want: "unknown event",
		},
		"function on wrong event": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n      - event: llm_response\n        function: foo\n",
			want: "function is only valid",
		},
		"judge on wrong event": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n      - event: function_call\n        judge: nice\n",
			want: "judge is only valid",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := eval.Load(writeScenario(t, tc.body))
			if err == nil {
				t.Fatal("expected an error, got nil")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.want)
			}
		})
	}
}
