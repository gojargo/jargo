package eval_test

import (
	"os"
	"path/filepath"
	"reflect"
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
        eval: "greets the user warmly"
  - user: "what's the weather in Paris?"
    expect:
      - event: function_call
        name: get_weather
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
	want := []eval.FunctionCall{{Name: "get_weather"}}
	if s.Turns[1].Expect[0].Event != eval.EventFunctionCall ||
		!reflect.DeepEqual(s.Turns[1].Expect[0].Calls, want) {
		t.Fatalf("unexpected function_call expectation: %+v", s.Turns[1].Expect[0])
	}
}

// The single-call `name:`/`args:` shorthand normalizes into Calls.
func TestLoadFunctionCallShorthand(t *testing.T) {
	s, err := eval.Load(writeScenario(t, `
name: one_call
turns:
  - user: "weather?"
    expect:
      - event: function_call
        name: get_weather
        args: {city: Paris}
`))
	if err != nil {
		t.Fatal(err)
	}
	want := []eval.FunctionCall{{Name: "get_weather", Args: map[string]any{"city": "Paris"}}}
	if !reflect.DeepEqual(s.Turns[0].Expect[0].Calls, want) {
		t.Fatalf("got %+v, want %+v", s.Turns[0].Expect[0].Calls, want)
	}
}

// Several calls in one turn go under `calls:`, where an entry is either a bare
// name or a {name, args} mapping.
func TestLoadFunctionCallsList(t *testing.T) {
	s, err := eval.Load(writeScenario(t, `
name: two_calls
turns:
  - user: "weather and food?"
    expect:
      - event: function_call
        calls:
          - get_weather
          - {name: get_restaurants, args: {city: Paris}}
`))
	if err != nil {
		t.Fatal(err)
	}
	want := []eval.FunctionCall{
		{Name: "get_weather"},
		{Name: "get_restaurants", Args: map[string]any{"city": "Paris"}},
	}
	if !reflect.DeepEqual(s.Turns[0].Expect[0].Calls, want) {
		t.Fatalf("got %+v, want %+v", s.Turns[0].Expect[0].Calls, want)
	}
}

// A bare function_call, with neither name nor calls, matches any single call.
func TestLoadBareFunctionCallMatchesAny(t *testing.T) {
	s, err := eval.Load(writeScenario(t,
		"name: bare\nturns: [{user: hi, expect: [{event: function_call}]}]\n"))
	if err != nil {
		t.Fatal(err)
	}
	want := []eval.FunctionCall{{}}
	if !reflect.DeepEqual(s.Turns[0].Expect[0].Calls, want) {
		t.Fatalf("got %+v, want %+v", s.Turns[0].Expect[0].Calls, want)
	}
}

// An absent expectation parses, and carries its own budget.
func TestLoadAbsentExpectation(t *testing.T) {
	s, err := eval.Load(writeScenario(t, `
name: absent
turns:
  - user: "x"
    expect:
      - event: llm_response
        eval: "answers"
      - event: llm_response
        absent: true
        within_ms: 5000
`))
	if err != nil {
		t.Fatal(err)
	}
	exp := s.Turns[0].Expect[1]
	if !exp.Absent || exp.WithinMS != 5000 {
		t.Fatalf("unexpected absent expectation: %+v", exp)
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
		"user and dtmf together": {
			body: "name: x\nturns:\n  - user: hi\n    dtmf: \"1\"\n    expect: [{event: llm_started}]\n",
			want: "user or dtmf, not both",
		},
		"invalid keypad key": {
			body: "name: x\nturns:\n  - dtmf: \"12x\"\n    expect: [{event: llm_started}]\n",
			want: `invalid keypad key "x"`,
		},
		"send_after with no input to schedule": {
			body: "name: x\nturns:\n  - send_after: {event: llm_started, delay_ms: 10}\n" +
				"    expect: [{event: llm_started}]\n",
			want: "send_after needs a user or dtmf turn",
		},
		"unknown event": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n      - event: teleport\n",
			want: "unknown event",
		},
		"name on wrong event": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n      - event: llm_response\n        name: foo\n",
			want: "name, args and calls need a function_call event",
		},
		"eval on wrong event": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n      - event: function_call\n        eval: nice\n",
			want: "eval is only valid",
		},
		"empty calls": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n      - event: function_call\n        calls: []\n",
			want: "calls must not be empty",
		},
		"absent with eval": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n" +
				"      - event: llm_response\n        absent: true\n        eval: repeats itself\n",
			want: "absent cannot be combined",
		},
		"absent with text_contains": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n" +
				"      - event: llm_response\n        absent: true\n        text_contains: again\n",
			want: "absent cannot be combined",
		},
		"absent with calls": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n" +
				"      - event: function_call\n        absent: true\n        name: hang_up\n",
			want: "absent cannot be combined",
		},
		"negative send_after delay": {
			body: "name: x\nturns:\n  - user: hi\n    send_after: {event: llm_started, delay_ms: -1}\n" +
				"    expect: [{event: llm_started}]\n",
			want: "delay_ms must be non-negative",
		},
		"send_after with neither event nor delay": {
			body: "name: x\nturns:\n  - user: hi\n    send_after: {delay_ms: 0}\n" +
				"    expect: [{event: llm_started}]\n",
			want: "needs an event or a positive delay_ms",
		},
		"absent must be boolean": {
			body: "name: x\nturns:\n  - user: hi\n    expect:\n" +
				"      - event: llm_response\n        absent: \"yes please\"\n",
			want: "cannot unmarshal",
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

// A turn's send_after parses, and a turn without one has none.
func TestLoadSendAfter(t *testing.T) {
	s, err := eval.Load(writeScenario(t, `
name: with_send_after
turns:
  - user: "first"
    expect: [{event: llm_started}]
  - send_after: {event: llm_started, delay_ms: 200}
    user: "second"
    expect: [{event: llm_started}]
`))
	if err != nil {
		t.Fatal(err)
	}
	if s.Turns[0].SendAfter != nil {
		t.Fatalf("turn 1 should have no schedule, got %+v", s.Turns[0].SendAfter)
	}
	want := eval.SendAfter{Event: eval.EventLLMStarted, DelayMS: 200}
	if s.Turns[1].SendAfter == nil || *s.Turns[1].SendAfter != want {
		t.Fatalf("got %+v, want %+v", s.Turns[1].SendAfter, want)
	}
}

// A send_after with only a delay is a pure time delay, with no event to anchor
// on.
func TestLoadSendAfterPureDelay(t *testing.T) {
	s, err := eval.Load(writeScenario(t, `
name: paced
turns:
  - user: "first"
    expect: [{event: llm_started}]
  - user: "second"
    send_after: {delay_ms: 500}
    expect: [{event: llm_started}]
`))
	if err != nil {
		t.Fatal(err)
	}
	want := eval.SendAfter{DelayMS: 500}
	if s.Turns[1].SendAfter == nil || *s.Turns[1].SendAfter != want {
		t.Fatalf("got %+v, want %+v", s.Turns[1].SendAfter, want)
	}
}

// A turn may have no input at all: it only waits and asserts, which is what a
// bot-first scenario needs.
func TestLoadObserveOnlyTurn(t *testing.T) {
	s, err := eval.Load(writeScenario(t, `
name: bot_first
turns:
  - expect:
      - event: llm_response
`))
	if err != nil {
		t.Fatal(err)
	}
	if s.Turns[0].User != "" || s.Turns[0].DTMF != "" {
		t.Fatalf("expected a turn with no input, got %+v", s.Turns[0])
	}
}

// A turn may also have no expectations: it only sends, with the assertion on a
// later turn.
func TestLoadSendOnlyTurn(t *testing.T) {
	s, err := eval.Load(writeScenario(t,
		"name: paced\nturns: [{user: hi}, {user: there, expect: [{event: llm_response}]}]\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Turns[0].Expect) != 0 {
		t.Fatalf("expected a turn with no expectations, got %+v", s.Turns[0])
	}
}

// A keypad sequence parses whether or not it is quoted: YAML reads bare digits
// as a number, and the digits are what matter.
func TestLoadDTMFTurn(t *testing.T) {
	for name, body := range map[string]string{
		"quoted":   "name: keys\nturns: [{dtmf: \"123#\", expect: [{event: user_transcription}]}]\n",
		"unquoted": "name: keys\nturns: [{dtmf: 123, expect: [{event: user_transcription}]}]\n",
	} {
		t.Run(name, func(t *testing.T) {
			s, err := eval.Load(writeScenario(t, body))
			if err != nil {
				t.Fatal(err)
			}
			want := "123"
			if name == "quoted" {
				want = "123#"
			}
			if s.Turns[0].DTMF != want {
				t.Fatalf("dtmf = %q, want %q", s.Turns[0].DTMF, want)
			}
		})
	}
}

// A scenario's context: seeds the conversation the bot starts from.
func TestLoadContext(t *testing.T) {
	s, err := eval.Load(writeScenario(t, `
name: seeded
context:
  - role: user
    text: "my name is Alex"
  - role: assistant
    text: "nice to meet you, Alex"
turns:
  - user: "what is my name?"
    expect: [{event: llm_response}]
`))
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Context) != 2 || s.Context[0].Text != "my name is Alex" {
		t.Fatalf("unexpected context: %+v", s.Context)
	}
}

// !include pulls any value from a separate file, so scenarios can share a block
// they all need. Paths resolve against the scenario file.
func TestLoadInclude(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "turns.yaml", `- user: "hello"
  expect: [{event: llm_response}]
`)
	writeFile(t, dir, "scenario.yaml", "name: shared\nturns: !include turns.yaml\n")

	s, err := eval.Load(filepath.Join(dir, "scenario.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Turns) != 1 || s.Turns[0].User != "hello" {
		t.Fatalf("the included turns should have been pulled in, got %+v", s.Turns)
	}
}

// A file that includes itself is reported rather than exhausting the stack.
func TestLoadIncludeCycleRejected(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "loop.yaml", "name: loop\nturns: !include loop.yaml\n")

	if _, err := eval.Load(filepath.Join(dir, "loop.yaml")); err == nil {
		t.Fatal("expected an error for the include cycle")
	} else if !strings.Contains(err.Error(), "nested too deep") {
		t.Fatalf("unexpected error: %v", err)
	}
}
