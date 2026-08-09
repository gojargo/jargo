package chat

import (
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
