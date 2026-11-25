package anthropic

import (
	"encoding/json"
	"sync"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm/adapter"
)

// aggregatorThought builds a thought the way the assistant aggregator writes
// one when the Anthropic service asks for its reasoning back.
func aggregatorThought(text, signature string) frames.Message {
	return frames.NewLLMSpecificMessage("anthropic", map[string]any{
		"type": "thought", "text": text, "signature": signature,
	})
}

// TestThoughtConvertedToThinkingBlock checks a signed thought, as the assistant
// aggregator writes it, becomes a thinking block merged into the assistant
// response it preceded, with the thinking block first.
func TestThoughtConvertedToThinkingBlock(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("What's on the menu?")
	convo.AddMessage(aggregatorThought("Let me check.", "sig"))
	convo.AddAssistantMessage("Espresso and tea.")

	p := adapterParamsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant)
	content := p.Messages[1].Content
	if len(content) != 2 {
		t.Fatalf("content = %+v, want the thinking block and the text", content)
	}
	if th := content[0].OfThinking; th == nil || th.Thinking != "Let me check." || th.Signature != "sig" {
		t.Errorf("block 0 = %+v, want the thinking block", content[0])
	}
	if tx := content[1].OfText; tx == nil || tx.Text != "Espresso and tea." {
		t.Errorf("block 1 = %+v, want the response text", content[1])
	}
}

// TestThoughtWithEmptyTextPreserved checks a signed thought with no text still
// round-trips: a model set to omit its reasoning returns thinking blocks whose
// text is empty, and Anthropic requires them to be passed back.
func TestThoughtWithEmptyTextPreserved(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("What's on the menu?")
	convo.AddMessage(aggregatorThought("", "sig"))
	convo.AddAssistantMessage("Espresso and tea.")

	p := adapterParamsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant)
	content := p.Messages[1].Content
	if len(content) != 2 || content[0].OfThinking == nil ||
		content[0].OfThinking.Thinking != "" || content[0].OfThinking.Signature != "sig" {
		t.Fatalf("content = %+v, want an empty thinking block then the text", content)
	}
}

// TestThoughtWithoutSignatureDropped checks a thought with no signature, which
// cannot be round-tripped, is skipped and leaves the turns around it intact.
func TestThoughtWithoutSignatureDropped(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("What's on the menu?")
	convo.AddMessage(aggregatorThought("Let me check.", ""))
	convo.AddAssistantMessage("Espresso and tea.")

	p := adapterParamsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant)
	if len(p.Messages[1].Content) != 1 || textOf(p.Messages[1]) != "Espresso and tea." {
		t.Errorf("assistant = %+v, want only the response text", p.Messages[1].Content)
	}
}

// TestThoughtBeforeToolCallMergesIntoTheToolTurn checks reasoning that preceded
// a tool call is sent back in the same assistant turn as the call, ahead of it,
// which is where Anthropic requires it when thinking runs alongside tools.
func TestThoughtBeforeToolCallMergesIntoTheToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Order a latte")
	convo.AddMessage(aggregatorThought("I should place the order.", "sig"))
	convo.AddAssistantToolCall(frames.ToolCall{
		ID: "call_1", Name: "place_order", Args: json.RawMessage(`{"drink":"latte"}`),
	})
	convo.AddToolResult(frames.ToolResult{ID: "call_1", Name: "place_order", Content: `{"ok":true}`})

	p := adapterParamsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages,
		sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant, sdk.MessageParamRoleUser)
	content := p.Messages[1].Content
	if len(content) != 2 || content[0].OfThinking == nil || content[1].OfToolUse == nil {
		t.Fatalf("assistant = %+v, want the thinking block then the tool use", content)
	}
}

// nativeUserMessage builds a native message whose block slice has room to grow,
// which is what would let a merge append into the conversation's own array.
func nativeUserMessage(text string) frames.Message {
	content := make([]sdk.ContentBlockParamUnion, 1, 8)
	content[0] = sdk.NewTextBlock(text)
	return adapter.CreateLLMSpecificMessage(&Adapter{},
		sdk.MessageParam{Role: sdk.MessageParamRoleUser, Content: content})
}

// TestConversionDoesNotMutateNativeMessages checks a native message kept in the
// conversation comes out of a conversion exactly as it went in, even when the
// conversion merges it with its neighbors and marks it for caching.
func TestConversionDoesNotMutateNativeMessages(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(nativeUserMessage("native"))
	convo.AddUserMessage("after")
	before, err := json.Marshal(convo.Messages()[0].Native)
	if err != nil {
		t.Fatal(err)
	}

	p := adapterParamsOf(t, convo, adapter.Options{EnablePromptCaching: true})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser)
	if got := textOf(p.Messages[0]); got != "nativeafter" {
		t.Errorf("merged text = %q, want both messages", got)
	}

	native, ok := convo.Messages()[0].Native.(sdk.MessageParam)
	if !ok {
		t.Fatalf("native = %T, want the message it was written with", convo.Messages()[0].Native)
	}
	after, err := json.Marshal(native)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Errorf("native message changed by the conversion:\nbefore %s\nafter  %s", before, after)
	}
	// The block the merge appended past the slice's length would sit in the
	// spare capacity of the conversation's own array.
	if spare := native.Content[:2][1]; spare.OfText != nil {
		t.Errorf("the conversation's block array was written into: %+v", spare)
	}
}

// TestDeepCopyKeepsTheWireForm checks the copy taken of a native message sends
// exactly what the original would have.
func TestDeepCopyKeepsTheWireForm(t *testing.T) {
	msg := sdk.NewAssistantMessage(
		sdk.NewThinkingBlock("sig", "reasoning"),
		sdk.NewTextBlock("hello"),
		sdk.NewToolUseBlock("call_1", map[string]any{"drink": "latte"}, "place_order"),
	)
	copied, err := deepCopy(msg)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(msg)
	got, _ := json.Marshal(copied)
	if string(got) != string(want) {
		t.Errorf("copy = %s, want %s", got, want)
	}
	if copied.Content[1].OfText == msg.Content[1].OfText {
		t.Error("the copy shares a content block with the original")
	}
}

// TestConcurrentConversionsOfNativeMessages runs several conversions of one
// conversation at once, as a one-shot inference beside a generation does. Run
// under -race it fails if a conversion writes into a message the conversation
// owns.
func TestConcurrentConversionsOfNativeMessages(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(nativeUserMessage("native"))
	convo.AddUserMessage("after")
	convo.AddMessage(adapter.CreateLLMSpecificMessage(&Adapter{},
		sdk.NewAssistantMessage(sdk.NewTextBlock("native reply"))))
	convo.AddAssistantMessage("and more")
	convo.AddUserMessage("last")

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			p, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{EnablePromptCaching: true})
			if err != nil {
				t.Error(err)
				return
			}
			if len(p.Messages) != 3 || textOf(p.Messages[0]) != "nativeafter" {
				t.Errorf("messages = %+v, want three merged turns", p.Messages)
			}
		})
	}
	wg.Wait()
}
