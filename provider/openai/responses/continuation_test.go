package responses

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// outputOf decodes a completed response's output array.
func outputOf(t *testing.T, raw string) []responseOutput {
	t.Helper()
	var out []responseOutput
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("decoding output: %v", err)
	}
	return out
}

// TestStartsWithResponseOutput checks which conversations count as continuing
// from a response's output: a message matches on role and text, a function call
// on its call id, and any other kind of item never matches.
func TestStartsWithResponseOutput(t *testing.T) {
	cases := []struct {
		name   string
		output string
		items  []inputItem
		want   bool
	}{
		{
			name:   "text message matches by role and text",
			output: `[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello!"}]}]`,
			items: []inputItem{
				{Type: itemMessage, Role: "assistant", Content: "Hello!"},
				{Type: itemMessage, Role: "user", Content: "hi"},
			},
			want: true,
		},
		{
			name: "function call matches by call id",
			output: `[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather",` +
				`"arguments":"{\"location\": \"SF\"}"}]`,
			items: []inputItem{
				{Type: itemFuncCall, CallID: "call_1", Name: "get_weather", Arguments: "{}"},
				{Type: itemFuncOutput, CallID: "call_1", Output: "sunny"},
			},
			want: true,
		},
		{
			name: "mixed output",
			output: `[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Let me check."}]},` +
				`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"}]`,
			items: []inputItem{
				{Type: itemMessage, Role: "assistant", Content: "Let me check."},
				{Type: itemFuncCall, CallID: "call_1", Name: "get_weather", Arguments: "{}"},
				{Type: itemFuncOutput, CallID: "call_1", Output: "sunny"},
			},
			want: true,
		},
		{
			name:   "role mismatch",
			output: `[{"type":"message","role":"assistant","content":[]}]`,
			items:  []inputItem{{Type: itemMessage, Role: "user", Content: "hi"}},
		},
		{
			name:   "text content mismatch",
			output: `[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello!"}]}]`,
			items:  []inputItem{{Type: itemMessage, Role: "assistant", Content: "Something completely different"}},
		},
		{
			name:   "call id mismatch",
			output: `[{"type":"function_call","call_id":"call_1","name":"f"}]`,
			items:  []inputItem{{Type: itemFuncCall, CallID: "call_999", Name: "f"}},
		},
		{
			name: "too few items",
			output: `[{"type":"message","role":"assistant","content":[]},` +
				`{"type":"function_call","call_id":"call_1","name":"f"}]`,
			items: []inputItem{{Type: itemMessage, Role: "assistant", Content: "hi"}},
		},
		{name: "empty output, no items", output: `[]`, want: true},
		{
			name:   "empty output, some items",
			output: `[]`,
			items:  []inputItem{{Type: itemMessage, Role: "user"}},
			want:   true,
		},
		{
			name:   "unknown output type rejects",
			output: `[{"type":"unknown_thing","data":"something"}]`,
			items:  []inputItem{{Type: itemMessage, Role: "assistant", Content: "hi"}},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := startsWithResponseOutput(c.items, outputOf(t, c.output)); got != c.want {
				t.Errorf("startsWithResponseOutput = %v, want %v", got, c.want)
			}
		})
	}
}

// completedWith is a response.completed event carrying the given output.
func completedWith(id, output string) string {
	return `{"type":"response.completed","response":{"id":"` + id + `","output":` + output + `}}`
}

// textOutput is the output of a response that only said text.
func textOutput(text string) string {
	b, _ := json.Marshal(text)
	return `[{"type":"message","role":"assistant","content":[{"type":"output_text","text":` +
		string(b) + `}]}]`
}

// twoTurns runs a first turn over convo, lets edit change the conversation the
// way the turn's aftermath does, runs a second turn, and returns the second
// request.
func twoTurns(
	t *testing.T, first []string, convo *frames.LLMContext, edit func(*frames.LLMContext),
) map[string]any {
	t.Helper()
	srv := newFakeResponsesWS(t, [][]string{first, {
		`{"type":"response.created","response":{"id":"resp_2"}}`,
		`{"type":"response.output_text.delta","delta":"x"}`,
		completedWith("resp_2", textOutput("x")),
	}})
	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL()})
	t.Cleanup(svc.disconnect)
	discard := func(string) error { return nil }
	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("first Generate: %v", err)
	}
	edit(convo)
	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("second Generate: %v", err)
	}
	return srv.turn(1)
}

// textTurn is a first turn whose response only said text.
func textTurn(text string) []string {
	b, _ := json.Marshal(text)
	return []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_text.delta","delta":` + string(b) + `}`,
		completedWith("resp_1", textOutput(text)),
	}
}

// callTurn is a first turn whose response only called a function.
func callTurn(callID string) []string {
	return []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call",` +
			`"id":"fc_1","call_id":"` + callID + `","name":"get_weather"}}`,
		completedWith("resp_1", `[{"type":"function_call","id":"fc_1","call_id":"`+callID+
			`","name":"get_weather","arguments":"{}"}]`),
	}
}

// userConvo is a conversation holding one user message.
func userConvo(text string) *frames.LLMContext {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage(text)
	return convo
}

// TestContinuationSendsOnlyWhatIsNew checks a conversation holding the previous
// response's own output continues from it, sending only the new message.
func TestContinuationSendsOnlyWhatIsNew(t *testing.T) {
	req := twoTurns(t, textTurn("hello"), userConvo("hi"), func(c *frames.LLMContext) {
		c.AddAssistantMessage("hello")
		c.AddUserMessage("how are you?")
	})
	if req["previous_response_id"] != "resp_1" {
		t.Errorf("previous_response_id = %v, want resp_1", req["previous_response_id"])
	}
	input, _ := req["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input = %v, want only the new user message", input)
	}
	if item, _ := input[0].(map[string]any); item["content"] != "how are you?" {
		t.Errorf("input = %v, want only the new user message", input)
	}
}

// TestContinuationSameLengthSendsFull checks a conversation that did not grow
// is sent whole.
func TestContinuationSameLengthSendsFull(t *testing.T) {
	first := []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		completedWith("resp_1", `[]`),
	}
	req := twoTurns(t, first, userConvo("hi"), func(*frames.LLMContext) {})
	if _, ok := req["previous_response_id"]; ok {
		t.Errorf("request = %v, want the full context", req)
	}
}

// TestContinuationOutputMismatchSendsFull checks the items after the previous
// input must be the response's own output: anything else in their place sends
// the full context.
func TestContinuationOutputMismatchSendsFull(t *testing.T) {
	req := twoTurns(t, textTurn("hello"), userConvo("hi"), func(c *frames.LLMContext) {
		c.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "something unexpected"})
		c.AddUserMessage("how are you?")
	})
	if _, ok := req["previous_response_id"]; ok {
		t.Errorf("request = %v, want the full context", req)
	}
	if input, _ := req["input"].([]any); len(input) != 3 {
		t.Errorf("input = %v, want the full context of three items", input)
	}
}

// TestContinuationAfterAnInterruptedReplySendsFull checks the voice case: the
// response completed, the user barged in while it was spoken, and the
// conversation holds only the words heard. Continuing from the server's copy
// would leave the model believing the user heard the whole reply.
func TestContinuationAfterAnInterruptedReplySendsFull(t *testing.T) {
	full := "Sure, the options are A, B, C and D. Which one?"
	req := twoTurns(t, textTurn(full), userConvo("what can I order?"), func(c *frames.LLMContext) {
		c.AddAssistantMessage("Sure, the options are A")
		c.AddUserMessage("wait, what?")
	})
	if _, ok := req["previous_response_id"]; ok {
		t.Errorf("request = %v, want the full context, with the reply as it was heard", req)
	}
}

// TestContinuationCallIDMismatchSendsFull checks a function call whose id is
// not the one the response made sends the full context.
func TestContinuationCallIDMismatchSendsFull(t *testing.T) {
	req := twoTurns(t, callTurn("call_1"), userConvo("hi"), func(c *frames.LLMContext) {
		c.AddAssistantToolCall(frames.ToolCall{ID: "call_999", Name: "get_weather", Args: json.RawMessage(`{}`)})
		c.AddToolResult(frames.ToolResult{ID: "call_999", Content: "sunny"})
	})
	if _, ok := req["previous_response_id"]; ok {
		t.Errorf("request = %v, want the full context", req)
	}
}

// TestContinuationFunctionCallMatchesByCallID checks a function call the
// response made continues from it, sending only the result.
func TestContinuationFunctionCallMatchesByCallID(t *testing.T) {
	req := twoTurns(t, callTurn("call_1"), userConvo("hi"), func(c *frames.LLMContext) {
		c.AddAssistantToolCall(frames.ToolCall{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{}`)})
		c.AddToolResult(frames.ToolResult{ID: "call_1", Content: "sunny"})
	})
	if req["previous_response_id"] != "resp_1" {
		t.Errorf("previous_response_id = %v, want resp_1", req["previous_response_id"])
	}
	input, _ := req["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input = %v, want only the function call output", input)
	}
	if item, _ := input[0].(map[string]any); item["type"] != itemFuncOutput {
		t.Errorf("input = %v, want only the function call output", input)
	}
}
