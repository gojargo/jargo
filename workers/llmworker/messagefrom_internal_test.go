package llmworker

import (
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// An activation's messages keep what a universal message can carry: text given
// as parts, the calls an assistant made, and the result a tool message carries.
func TestActivationMessagesKeepMoreThanPlainText(t *testing.T) {
	cases := map[string]struct {
		raw  map[string]any
		want frames.Message
	}{
		"text parts": {
			raw: map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "Book a table"},
				map[string]any{"type": "text", "text": "for two"},
			}},
			want: frames.Message{Role: frames.RoleUser, Text: "Book a table for two"},
		},
		"tool calls": {
			raw: map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
				"id": "call_1", "type": "function",
				"function": map[string]any{"name": "book", "arguments": `{"people":2}`},
			}}},
			want: frames.Message{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{
				{ID: "call_1", Name: "book", Args: []byte(`{"people":2}`)},
			}},
		},
		"tool result": {
			raw: map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "booked"},
			want: frames.Message{Role: frames.RoleUser, ToolResults: []frames.ToolResult{
				{ID: "call_1", Content: "booked"},
			}},
		},
	}
	for name, tc := range cases {
		if got := messageFrom(tc.raw); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: message = %+v, want %+v", name, got, tc.want)
		}
	}
}

// Messages put into the arguments as maps, rather than decoded off the bus,
// are read too.
func TestActivationArgsFromTypedMessages(t *testing.T) {
	args := map[string]any{"messages": []map[string]any{{"role": "user", "content": "hi"}}}
	if got := ActivationArgsFrom(args); len(got.Messages) != 1 || got.Messages[0]["content"] != "hi" {
		t.Errorf("messages = %+v, want the one given", got.Messages)
	}
}
