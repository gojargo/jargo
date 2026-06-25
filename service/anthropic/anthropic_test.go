package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

func TestToToolsMapsSchema(t *testing.T) {
	tools := []frames.Tool{{
		Name:        "get_weather",
		Description: "Get the weather",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
	}}
	out := toTools(tools)
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	b, err := json.Marshal(out[0])
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{`"name":"get_weather"`, `"Get the weather"`, `"location"`, `"required":["location"]`} {
		if !strings.Contains(s, want) {
			t.Fatalf("tool JSON %s missing %q", s, want)
		}
	}
}

func TestToMessagesBuildsToolTurns(t *testing.T) {
	msgs := []frames.Message{
		{Role: frames.RoleUser, Text: "weather?"},
		{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{
			{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`)},
		}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{
			{ID: "c1", Name: "get_weather", Content: "sunny", IsError: false},
		}},
		{Role: frames.RoleAssistant, Text: "It is sunny."},
	}
	out := toMessages(msgs)
	if len(out) != 4 {
		t.Fatalf("len = %d, want 4", len(out))
	}
	b, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{`"tool_use"`, `"c1"`, `"get_weather"`, `"tool_result"`, `"tool_use_id":"c1"`, `sunny`} {
		if !strings.Contains(s, want) {
			t.Fatalf("messages JSON missing %q:\n%s", want, s)
		}
	}
}
