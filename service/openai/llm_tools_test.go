package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// fakeSink collects the text and tool calls reported during a tool-capable
// generation.
type fakeSink struct {
	text  strings.Builder
	calls []frames.ToolCall
}

func (f *fakeSink) Text(t string) error          { f.text.WriteString(t); return nil }
func (f *fakeSink) Tool(c frames.ToolCall) error { f.calls = append(f.calls, c); return nil }

// toolDelta builds a streamed tool-call delta for the given index.
func toolDelta(index int, id, name, args string) toolCallDelta {
	d := toolCallDelta{Index: index, ID: id}
	d.Function.Name = name
	d.Function.Arguments = args
	return d
}

func TestToMessagesToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("weather in Paris?")
	convo.AddAssistantToolCalls("", []frames.ToolCall{
		{ID: "call_a", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`)},
	})
	convo.AddToolResults([]frames.ToolResult{
		{ID: "call_a", Name: "get_weather", Content: "sunny, 20C"},
	})

	msgs := toMessages(convo)
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages (system, user, assistant, tool), got %d", len(msgs))
	}
	if msgs[0].Role != "system" || msgs[1].Role != "user" {
		t.Fatalf("unexpected leading roles: %q, %q", msgs[0].Role, msgs[1].Role)
	}

	asst := msgs[2]
	if asst.Role != "assistant" || len(asst.ToolCalls) != 1 {
		t.Fatalf("assistant tool message malformed: %+v", asst)
	}
	tc := asst.ToolCalls[0]
	if tc.ID != "call_a" || tc.Type != "function" || tc.Function.Name != "get_weather" {
		t.Errorf("tool_call fields wrong: %+v", tc)
	}
	if tc.Function.Arguments != `{"location":"Paris"}` {
		t.Errorf("tool_call arguments = %q", tc.Function.Arguments)
	}

	res := msgs[3]
	if res.Role != "tool" || res.ToolCallID != "call_a" || res.Content != "sunny, 20C" {
		t.Errorf("tool result message wrong: %+v", res)
	}
}

func TestToMessagesEmptyArgsDefaults(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddAssistantToolCalls("", []frames.ToolCall{{ID: "c1", Name: "now"}})
	msgs := toMessages(convo)
	if got := msgs[0].ToolCalls[0].Function.Arguments; got != "{}" {
		t.Errorf("empty args should default to {}, got %q", got)
	}
}

func TestToTools(t *testing.T) {
	out := toTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "Look up the weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})
	if len(out) != 1 {
		t.Fatalf("want 1 tool, got %d", len(out))
	}
	if out[0].Type != "function" || out[0].Function.Name != "get_weather" {
		t.Errorf("tool shape wrong: %+v", out[0])
	}
	if string(out[0].Function.Parameters) != `{"type":"object"}` {
		t.Errorf("parameters not passed through: %s", out[0].Function.Parameters)
	}
}

func TestToolCoalescer(t *testing.T) {
	c := &toolCoalescer{calls: map[int]*toolAccumulator{}}
	sink := &fakeSink{}

	// A text preamble, then two tool calls whose name/arguments arrive in
	// fragments across deltas (as OpenAI streams them).
	mustAdd(t, c, sink, chatDelta{Content: "Let me check. "})
	mustAdd(t, c, sink, chatDelta{ToolCalls: []toolCallDelta{toolDelta(0, "call_a", "get_weather", `{"loc`)}})
	mustAdd(t, c, sink, chatDelta{ToolCalls: []toolCallDelta{toolDelta(0, "", "", `ation":"Paris"}`)}})
	mustAdd(t, c, sink, chatDelta{ToolCalls: []toolCallDelta{toolDelta(1, "call_b", "get_time", "")}})
	if err := c.emit(sink); err != nil {
		t.Fatalf("emit: %v", err)
	}

	if sink.text.String() != "Let me check. " {
		t.Errorf("text = %q", sink.text.String())
	}
	if len(sink.calls) != 2 {
		t.Fatalf("want 2 calls, got %d", len(sink.calls))
	}
	if sink.calls[0].ID != "call_a" || sink.calls[0].Name != "get_weather" ||
		string(sink.calls[0].Args) != `{"location":"Paris"}` {
		t.Errorf("first call wrong: %+v (args %s)", sink.calls[0], sink.calls[0].Args)
	}
	// A call with no streamed arguments defaults to an empty JSON object.
	if sink.calls[1].ID != "call_b" || sink.calls[1].Name != "get_time" ||
		string(sink.calls[1].Args) != "{}" {
		t.Errorf("second call wrong: %+v (args %s)", sink.calls[1], sink.calls[1].Args)
	}
}

func mustAdd(t *testing.T, c *toolCoalescer, sink *fakeSink, d chatDelta) {
	t.Helper()
	if err := c.add(d, sink); err != nil {
		t.Fatalf("add: %v", err)
	}
}
