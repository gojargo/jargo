package frames_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

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

// TestMessagesDoesNotAliasToolResults is the regression test for a data race in
// the tool-result design. A tool result is rewritten in place once its call
// reports, so a caller holding a shallow copy of the messages would be reading
// an array being written under it. Messages must hand out a copy deep enough
// that no later update can reach it.
func TestMessagesDoesNotAliasToolResults(t *testing.T) {
	c := frames.NewLLMContext("system")
	c.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather", Args: []byte(`{}`)})
	c.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: "IN_PROGRESS"})

	snapshot := c.Messages()
	c.UpdateToolResult("c1", "sunny")

	if got := snapshot[1].ToolResults[0].Content; got != "IN_PROGRESS" {
		t.Errorf("snapshot result = %q, want the value it held when it was taken", got)
	}
	if got := c.Messages()[1].ToolResults[0].Content; got != "sunny" {
		t.Errorf("context result = %q, want the update", got)
	}

	// The tool calls are copied too, for the same reason.
	snapshot[0].ToolCalls[0].Name = "mutated by the caller"
	if got := c.Messages()[0].ToolCalls[0].Name; got != "get_weather" {
		t.Errorf("context call = %q, want it untouched by a caller's copy", got)
	}
}

// TestSetMessagesDoesNotAliasTheCaller checks the write path too: a caller that
// keeps the slice it passed must not be able to reach into the conversation.
func TestSetMessagesDoesNotAliasTheCaller(t *testing.T) {
	c := frames.NewLLMContext("system")
	in := []frames.Message{{
		Role:        frames.RoleUser,
		ToolResults: []frames.ToolResult{{ID: "c1", Content: "sunny"}},
	}}
	c.SetMessages(in)

	in[0].ToolResults[0].Content = "mutated by the caller"
	if got := c.Messages()[0].ToolResults[0].Content; got != "sunny" {
		t.Errorf("context result = %q, want it untouched by the caller's slice", got)
	}
}
