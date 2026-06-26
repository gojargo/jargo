package frames_test

import (
	"context"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// keep is a summarize callback that records what it was handed and returns a
// fixed summary.
func keep(
	prior *string, dropped *[]frames.Message, out string,
) func(context.Context, string, []frames.Message) (string, error) {
	return func(_ context.Context, p string, d []frames.Message) (string, error) {
		if prior != nil {
			*prior = p
		}
		if dropped != nil {
			*dropped = d
		}
		return out, nil
	}
}

func TestCompactFoldsOldTurnsIntoSummary(t *testing.T) {
	c := frames.NewLLMContext("base system")
	c.AddUserMessage("u1")
	c.AddAssistantMessage("a1")
	c.AddUserMessage("u2")
	c.AddAssistantMessage("a2")
	c.AddUserMessage("u3")
	c.AddAssistantMessage("a3")

	var gotPrior string
	var gotDropped []frames.Message
	ok, err := c.Compact(context.Background(), 2, keep(&gotPrior, &gotDropped, "SUMMARY"))
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !ok {
		t.Fatal("Compact returned false, want true")
	}
	if gotPrior != "" {
		t.Fatalf("prior summary = %q, want empty on first compaction", gotPrior)
	}
	// keepRecent=2 cuts at index 4 (u3), dropping the first four messages.
	if len(gotDropped) != 4 || gotDropped[0].Text != "u1" || gotDropped[3].Text != "a2" {
		t.Fatalf("dropped = %+v, want first four messages u1..a2", gotDropped)
	}
	msgs := c.Messages()
	if len(msgs) != 2 || msgs[0].Text != "u3" || msgs[1].Text != "a3" {
		t.Fatalf("remaining messages = %+v, want [u3 a3]", msgs)
	}
	if c.Summary() != "SUMMARY" {
		t.Fatalf("Summary() = %q, want SUMMARY", c.Summary())
	}
	sys := c.System()
	if !strings.Contains(sys, "base system") || !strings.Contains(sys, "SUMMARY") {
		t.Fatalf("System() = %q, want it to compose base prompt and summary", sys)
	}
}

func TestCompactIsCumulative(t *testing.T) {
	c := frames.NewLLMContext("sys")
	for _, s := range []string{"u1", "a1", "u2", "a2"} {
		if strings.HasPrefix(s, "u") {
			c.AddUserMessage(s)
		} else {
			c.AddAssistantMessage(s)
		}
	}
	if ok, err := c.Compact(context.Background(), 2, keep(nil, nil, "S1")); err != nil || !ok {
		t.Fatalf("first Compact: ok=%v err=%v", ok, err)
	}

	// Grow the conversation, then compact again — the prior summary must be fed
	// to the summarizer so it can extend it.
	c.AddUserMessage("u3")
	c.AddAssistantMessage("a3")
	var gotPrior string
	if ok, err := c.Compact(context.Background(), 2, keep(&gotPrior, nil, "S2")); err != nil || !ok {
		t.Fatalf("second Compact: ok=%v err=%v", ok, err)
	}
	if gotPrior != "S1" {
		t.Fatalf("prior summary on second compaction = %q, want S1", gotPrior)
	}
	if c.Summary() != "S2" {
		t.Fatalf("Summary() = %q, want S2", c.Summary())
	}
}

func TestCompactCutsOnCleanTurnBoundary(t *testing.T) {
	// A tool exchange sits in the middle; the cut must not orphan the tool result
	// from its tool call, so the preserved tail starts at a plain user turn.
	c := frames.NewLLMContext("sys")
	c.AddUserMessage("u1")
	c.AddAssistantToolCalls("", []frames.ToolCall{{ID: "t1", Name: "lookup"}})
	c.AddToolResults([]frames.ToolResult{{ID: "t1", Name: "lookup", Content: "42"}})
	c.AddAssistantMessage("a1")
	c.AddUserMessage("u2")
	c.AddAssistantMessage("a2")

	var gotDropped []frames.Message
	ok, err := c.Compact(context.Background(), 2, keep(nil, &gotDropped, "S"))
	if err != nil || !ok {
		t.Fatalf("Compact: ok=%v err=%v", ok, err)
	}
	// The whole tool exchange is inside the dropped prefix...
	if len(gotDropped) != 4 {
		t.Fatalf("dropped %d messages, want 4 (the full tool exchange)", len(gotDropped))
	}
	// ...and the tail begins with a plain user message, no stray tool result.
	msgs := c.Messages()
	if len(msgs) != 2 || msgs[0].Role != frames.RoleUser || len(msgs[0].ToolResults) != 0 || msgs[0].Text != "u2" {
		t.Fatalf("tail = %+v, want it to start at the clean user turn u2", msgs)
	}
}

func TestCompactRefusesToOrphanToolResult(t *testing.T) {
	// The only droppable boundary would split a tool exchange, so Compact must
	// decline rather than produce an invalid message list.
	c := frames.NewLLMContext("sys")
	c.AddUserMessage("u1")
	c.AddAssistantToolCalls("", []frames.ToolCall{{ID: "t1", Name: "lookup"}})
	c.AddToolResults([]frames.ToolResult{{ID: "t1", Name: "lookup", Content: "42"}})
	c.AddAssistantMessage("a1")

	called := false
	ok, err := c.Compact(context.Background(), 1, func(context.Context, string, []frames.Message) (string, error) {
		called = true
		return "S", nil
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if ok || called {
		t.Fatalf("Compact compacted (ok=%v called=%v), want a no-op", ok, called)
	}
	if c.Summary() != "" || len(c.Messages()) != 4 {
		t.Fatalf("context changed: summary=%q messages=%d", c.Summary(), len(c.Messages()))
	}
}

func TestCompactNoopWhenShort(t *testing.T) {
	c := frames.NewLLMContext("sys")
	c.AddUserMessage("u1")
	c.AddAssistantMessage("a1")

	ok, err := c.Compact(context.Background(), 8, keep(nil, nil, "S"))
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if ok {
		t.Fatal("Compact returned true, want false (nothing to compact)")
	}
}

func TestCompactSkipsOnEmptySummary(t *testing.T) {
	c := frames.NewLLMContext("sys")
	c.AddUserMessage("u1")
	c.AddAssistantMessage("a1")
	c.AddUserMessage("u2")
	c.AddAssistantMessage("a2")

	ok, err := c.Compact(context.Background(), 2, keep(nil, nil, "   "))
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if ok {
		t.Fatal("Compact applied a blank summary, want it skipped")
	}
	if len(c.Messages()) != 4 || c.Summary() != "" {
		t.Fatalf("context changed on blank summary: messages=%d summary=%q", len(c.Messages()), c.Summary())
	}
}

func TestSystemWithoutSummaryIsUnchanged(t *testing.T) {
	c := frames.NewLLMContext("just the base")
	if c.System() != "just the base" {
		t.Fatalf("System() = %q, want the base prompt verbatim before any summary", c.System())
	}
}
