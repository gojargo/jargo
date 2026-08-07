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
	c.AddAssistantToolCall(frames.ToolCall{ID: "t1", Name: "lookup"})
	c.AddToolResult(frames.ToolResult{ID: "t1", Name: "lookup", Content: "42"})
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
	c.AddAssistantToolCall(frames.ToolCall{ID: "t1", Name: "lookup"})
	c.AddToolResult(frames.ToolResult{ID: "t1", Name: "lookup", Content: "42"})
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
		c.AddAssistantToolCall(frames.ToolCall{ID: "a", Name: "one"})
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
	c.AddAssistantToolCall(frames.ToolCall{ID: "a", Name: "get_weather", Args: []byte(`{"city":"Paris"}`)})
	c.AddToolResult(frames.ToolResult{ID: "a", Name: "get_weather", Content: strings.Repeat("d", 40)})
	if c.EstimatedTokens() <= before {
		t.Error("tool calls and results should count toward the estimate")
	}
}

// TestContextToolChoice checks the default and the setter, which back the
// LLMSetToolChoiceFrame the aggregator applies.
func TestContextToolChoice(t *testing.T) {
	c := frames.NewLLMContext("system")
	if got := c.ToolChoice(); got != frames.ToolChoiceAuto {
		t.Errorf("ToolChoice() = %q, want auto by default", got)
	}
	c.SetToolChoice(frames.ToolChoiceNone)
	if got := c.ToolChoice(); got != frames.ToolChoiceNone {
		t.Errorf("ToolChoice() = %q, want none", got)
	}
}

// TestContextSetMessages checks SetMessages replaces the conversation, leaves the
// system prompt alone, and copies its input so a later caller mutation cannot
// reach into the context.
func TestContextSetMessages(t *testing.T) {
	c := frames.NewLLMContext("system")
	c.AddUserMessage("old")

	in := []frames.Message{{Role: frames.RoleUser, Text: "new"}}
	c.SetMessages(in)

	got := c.Messages()
	if len(got) != 1 || got[0].Text != "new" {
		t.Fatalf("Messages() = %+v, want the conversation replaced", got)
	}
	if c.System() != "system" {
		t.Errorf("System() = %q, want it untouched", c.System())
	}

	in[0].Text = "mutated by the caller"
	if again := c.Messages(); again[0].Text != "new" {
		t.Error("SetMessages should copy its input, not alias the caller's slice")
	}
}

// TestUpdateToolResult checks a result settles into the placeholder its call
// wrote, wherever that placeholder has ended up. Appending instead would put a
// slow tool's result after whatever landed while it ran, separating it from the
// call it answers.
func TestUpdateToolResult(t *testing.T) {
	c := frames.NewLLMContext("system")
	c.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather"})
	c.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: "IN_PROGRESS"})
	c.AddUserMessage("actually, what time is it?")

	if !c.UpdateToolResult("c1", "sunny") {
		t.Fatal("UpdateToolResult reported no result to update")
	}
	msgs := c.Messages()
	if len(msgs) != 3 {
		t.Fatalf("messages = %+v, want the update to replace rather than append", msgs)
	}
	if msgs[1].ToolResults[0].Content != "sunny" {
		t.Errorf("msg[1] = %+v, want the placeholder replaced in place", msgs[1])
	}
	if msgs[2].Text != "actually, what time is it?" {
		t.Errorf("msg[2] = %+v, want the later turn undisturbed", msgs[2])
	}

	if c.UpdateToolResult("nope", "x") {
		t.Error("UpdateToolResult should report false for a call with no result message")
	}
}

// TestAddAssistantToolCallWritesOneCallPerMessage checks each call is its own
// message. That is what lets each be followed straight away by the message
// answering it, so the conversation is valid at every instant rather than only
// once the whole batch has reported.
func TestAddAssistantToolCallWritesOneCallPerMessage(t *testing.T) {
	c := frames.NewLLMContext("system")
	c.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "a"})
	c.AddToolResult(frames.ToolResult{ID: "c1", Content: "ra"})
	c.AddAssistantToolCall(frames.ToolCall{ID: "c2", Name: "b"})
	c.AddToolResult(frames.ToolResult{ID: "c2", Content: "rb"})

	msgs := c.Messages()
	if len(msgs) != 4 {
		t.Fatalf("messages = %+v, want a message each for both calls and both results", msgs)
	}
	for i, id := range []string{"c1", "c2"} {
		call, res := msgs[i*2], msgs[i*2+1]
		if call.Role != frames.RoleAssistant || len(call.ToolCalls) != 1 || call.ToolCalls[0].ID != id {
			t.Fatalf("msg[%d] = %+v, want the single assistant call %s", i*2, call, id)
		}
		if res.Role != frames.RoleUser || len(res.ToolResults) != 1 || res.ToolResults[0].ID != id {
			t.Fatalf("msg[%d] = %+v, want the single result for %s", i*2+1, res, id)
		}
	}
}

// TestAddMessage checks a message is appended as it stands, which is how the
// developer messages an asynchronous tool reports through reach the context.
func TestAddMessage(t *testing.T) {
	c := frames.NewLLMContext("system")
	c.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "an update"})

	msgs := c.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleDeveloper || msgs[0].Text != "an update" {
		t.Fatalf("messages = %+v, want the developer message as it stands", msgs)
	}
}

// TestCompactCutsPastDeveloperMessages checks a developer message is never taken
// for the start of a user turn. Cutting there would strand it from the tool call
// whose progress it reports.
func TestCompactCutsPastDeveloperMessages(t *testing.T) {
	c := frames.NewLLMContext("system")
	c.AddUserMessage("u1")
	c.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "watch"})
	c.AddMessage(frames.NewAsyncToolStartedMessage("c1"))
	c.AddUserMessage("u2")
	c.AddMessage(frames.NewAsyncToolFinalMessage("c1", "done"))
	c.AddAssistantMessage("a1")

	var dropped []frames.Message
	ok, err := c.Compact(context.Background(), 2, keep(nil, &dropped, "S"))
	if err != nil || !ok {
		t.Fatalf("Compact: ok=%v err=%v", ok, err)
	}
	// The only clean cut point is u2: neither the developer message nor the
	// async-tool started message, which is a tool result, may start a turn.
	msgs := c.Messages()
	if len(msgs) != 3 {
		t.Fatalf("tail = %+v, want the cut at the one plain user turn", msgs)
	}
	if msgs[0].Role != frames.RoleUser || msgs[0].Text != "u2" || len(msgs[0].ToolResults) != 0 {
		t.Fatalf("tail starts at %+v, want the plain user turn u2", msgs[0])
	}
	if msgs[1].Role != frames.RoleDeveloper {
		t.Fatalf("msg[1] = %+v, want the developer message kept with the tail", msgs[1])
	}
	if len(dropped) != 3 {
		t.Fatalf("dropped %d messages, want the three before u2", len(dropped))
	}
}
