package frames_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestMessagesForKeepsUniversalAndItsOwn checks a provider is sent every
// universal message plus its own, and none written for another provider.
func TestMessagesForKeepsUniversalAndItsOwn(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.NewLLMSpecificMessage("anthropic", "anthropic's own"))
	convo.AddMessage(frames.NewLLMSpecificMessage("google", "gemini's own"))
	convo.AddAssistantMessage("hi")
	convo.AddMessage(frames.NewLLMSpecificMessage("openai", "openai's own"))

	got := convo.MessagesFor("openai")
	if len(got) != 3 {
		t.Fatalf("messages = %+v, want the two universal ones and openai's", got)
	}
	if got[0].Text != "hello" || got[1].Text != "hi" {
		t.Errorf("messages = %+v, want the universal ones kept", got[:2])
	}
	if !got[2].IsLLMSpecific() || got[2].Native != "openai's own" {
		t.Errorf("message 2 = %+v, want openai's own", got[2])
	}
}

// TestMessagesForLeavesOthersOut checks a provider sent none of its own still
// gets the universal conversation.
func TestMessagesForLeavesOthersOut(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.NewLLMSpecificMessage("anthropic", "anthropic's own"))

	got := convo.MessagesFor("google")
	if len(got) != 1 || got[0].Text != "hello" {
		t.Errorf("messages = %+v, want only the universal one", got)
	}
}

// TestMessagesReturnsEverything checks the plain reader is unfiltered: it is
// what the conversation holds, not what one provider is sent.
func TestMessagesReturnsEverything(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.NewLLMSpecificMessage("anthropic", "anthropic's own"))

	if got := convo.Messages(); len(got) != 2 {
		t.Errorf("messages = %+v, want both", got)
	}
}

// TestMessagesForCopies checks the caller is given a copy: an adapter rewrites
// what it is handed, and the conversation must not change under it.
func TestMessagesForCopies(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")

	got := convo.MessagesFor("openai")
	got[0].Text = "rewritten"

	if convo.Messages()[0].Text != "hello" {
		t.Error("the conversation changed under the caller, want it left as it was")
	}
}

// TestEstimatedTokensCountsNativeMessages checks a message in a provider's own
// format is not counted as nothing, which would let a conversation of them grow
// past the point where it should have been compacted.
func TestEstimatedTokensCountsNativeMessages(t *testing.T) {
	convo := frames.NewLLMContext("")
	empty := convo.EstimatedTokens()

	convo.AddMessage(frames.NewLLMSpecificMessage("openai", "a message long enough to count"))
	if convo.EstimatedTokens() <= empty {
		t.Error("a provider's own message counted as nothing, want it counted")
	}
}

// TestCompactDoesNotCutOnANativeMessage checks compaction never treats a
// message it cannot read as a turn boundary, even when its role happens to read
// as a user turn. Cutting there would drop a prefix on a point nothing verified
// was a clean one.
func TestCompactDoesNotCutOnANativeMessage(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("one")
	convo.AddAssistantMessage("a")
	// The only candidate boundary in range, and one this cannot read.
	native := frames.NewLLMSpecificMessage("openai", "native")
	native.Role = frames.RoleUser
	convo.AddMessage(native)
	convo.AddAssistantMessage("b")

	compacted, err := convo.Compact(
		t.Context(), 1,
		func(context.Context, string, []frames.Message) (string, error) { return "summary", nil },
	)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if compacted {
		t.Error("compaction cut on a message it cannot read, want it to find no boundary")
	}
	if got := convo.Messages(); len(got) != 4 {
		t.Errorf("messages = %d, want all four kept", len(got))
	}
}
