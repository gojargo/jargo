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

// TestContextMutators covers the setters a mid-session reconfiguration uses:
// swapping the system prompt, the toolset, and the transient recall block.
func TestContextMutators(t *testing.T) {
	c := frames.NewLLMContext("original")

	c.SetSystem("replaced")
	if got := c.System(); !strings.Contains(got, "replaced") {
		t.Errorf("System() = %q, want the replacement prompt", got)
	}
	if strings.Contains(c.System(), "original") {
		t.Error("System() still contains the original prompt")
	}

	tools := []frames.Tool{{Name: "get_weather", Description: "look up weather"}}
	c.SetTools(tools)
	got := c.Tools()
	if len(got) != 1 || got[0].Name != "get_weather" {
		t.Fatalf("Tools() = %+v, want the tool that was set", got)
	}
	// Tools returns a copy, so a caller cannot reach into the context.
	got[0].Name = "mutated"
	if c.Tools()[0].Name != "get_weather" {
		t.Error("Tools() handed out the backing slice; a caller mutated the context")
	}
}

// TestContextRecall checks retrieved memories are folded into the system prompt
// and can be cleared, since a memory processor refreshes them every turn.
func TestContextRecall(t *testing.T) {
	c := frames.NewLLMContext("be brief")
	if got := c.Recall(); got != "" {
		t.Errorf("Recall() = %q, want empty before anything is recalled", got)
	}

	c.SetRecall("the user prefers window seats")
	if got := c.Recall(); got != "the user prefers window seats" {
		t.Errorf("Recall() = %q", got)
	}
	sys := c.System()
	if !strings.Contains(sys, "be brief") || !strings.Contains(sys, "window seats") {
		t.Errorf("System() = %q, want both the prompt and the recall", sys)
	}

	c.SetRecall("")
	if strings.Contains(c.System(), "window seats") {
		t.Error("clearing recall should drop it from the system prompt")
	}
}

// TestReplaceLastAssistantText checks the aggregator can keep an in-progress
// assistant turn in sync with the words actually spoken, and that it refuses to
// touch anything else.
func TestReplaceLastAssistantText(t *testing.T) {
	t.Run("plain assistant message", func(t *testing.T) {
		c := frames.NewLLMContext("")
		c.AddUserMessage("hello")
		c.AddAssistantMessage("full response")

		if !c.ReplaceLastAssistantText("interrupted here") {
			t.Fatal("want the replacement to apply")
		}
		msgs := c.Messages()
		if got := msgs[len(msgs)-1].Text; got != "interrupted here" {
			t.Errorf("text = %q, want the replacement", got)
		}
	})

	t.Run("empty context", func(t *testing.T) {
		if frames.NewLLMContext("").ReplaceLastAssistantText("x") {
			t.Error("want false with no messages to replace")
		}
	})

	t.Run("last message is the user's", func(t *testing.T) {
		c := frames.NewLLMContext("")
		c.AddAssistantMessage("earlier")
		c.AddUserMessage("hello")
		if c.ReplaceLastAssistantText("x") {
			t.Error("want false when the last message is not the assistant's")
		}
	})

	t.Run("assistant turn carrying tool calls", func(t *testing.T) {
		c := frames.NewLLMContext("")
		c.AddAssistantToolCalls("thinking", []frames.ToolCall{{ID: "a", Name: "one"}})
		if c.ReplaceLastAssistantText("x") {
			t.Error("want false for a tool-call turn; its text is not spoken output")
		}
	})
}

// TestEstimatedTokens checks the compaction trigger counts every part of the
// context, not just the messages.
func TestEstimatedTokens(t *testing.T) {
	empty := frames.NewLLMContext("")
	if got := empty.EstimatedTokens(); got != 0 {
		t.Errorf("EstimatedTokens() = %d, want 0 for an empty context", got)
	}

	c := frames.NewLLMContext(strings.Repeat("a", 40)) // ~10 tokens
	base := c.EstimatedTokens()
	if base == 0 {
		t.Fatal("the system prompt should count toward the estimate")
	}

	c.AddUserMessage(strings.Repeat("b", 40))
	withMsg := c.EstimatedTokens()
	if withMsg <= base {
		t.Errorf("EstimatedTokens() = %d, want it to grow past %d once a message is added", withMsg, base)
	}

	c.SetRecall(strings.Repeat("c", 40))
	if c.EstimatedTokens() <= withMsg {
		t.Error("recall should count toward the estimate; it is sent to the model")
	}

	// Tool calls and results are part of what the model reads.
	before := c.EstimatedTokens()
	c.AddAssistantToolCalls("", []frames.ToolCall{{ID: "a", Name: "get_weather", Args: []byte(`{"city":"Paris"}`)}})
	c.AddToolResults([]frames.ToolResult{{ID: "a", Name: "get_weather", Content: strings.Repeat("d", 40)}})
	if c.EstimatedTokens() <= before {
		t.Error("tool calls and results should count toward the estimate")
	}
}
