package anthropic

import (
	"bytes"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// capturedWarnings swaps the default logger for one writing into a buffer and
// returns a reader for what was logged, restoring the logger when the test ends.
func capturedWarnings(t *testing.T) func() string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return buf.String
}

// paramsOf converts a conversation and fails the test if the conversion did.
func paramsOf(t *testing.T, convo *frames.LLMContext, opts adapter.Options) Params {
	t.Helper()
	p, err := (&Adapter{}).LLMInvocationParams(convo, opts)
	if err != nil {
		t.Fatalf("LLMInvocationParams: %v", err)
	}
	return p
}

// wantRoles fails unless the messages carry exactly the given roles in order.
func wantRoles(t *testing.T, msgs []sdk.MessageParam, want ...sdk.MessageParamRole) {
	t.Helper()
	if len(msgs) != len(want) {
		got := make([]sdk.MessageParamRole, 0, len(msgs))
		for _, m := range msgs {
			got = append(got, m.Role)
		}
		t.Fatalf("roles = %v, want %v", got, want)
	}
	for i := range want {
		if msgs[i].Role != want[i] {
			t.Errorf("message %d role = %q, want %q", i, msgs[i].Role, want[i])
		}
	}
}

// textOf returns the text blocks of a message, joined, which is what most of
// these tests compare.
func textOf(m sdk.MessageParam) string {
	var b strings.Builder
	for _, blk := range m.Content {
		if blk.OfText != nil {
			b.WriteString(blk.OfText.Text)
		}
	}
	return b.String()
}

func TestIDForLLMSpecificMessages(t *testing.T) {
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "anthropic" {
		t.Errorf("id = %q, want %q", got, "anthropic")
	}
}

// TestStandardMessagesConverted checks the conversation reaches the request with
// the system prompt carried beside it rather than in it.
func TestStandardMessagesConverted(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	convo.AddAssistantMessage("Hi there")

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant)
	if len(p.System) != 1 || p.System[0].Text != "You are helpful." {
		t.Errorf("system = %+v, want the conversation's prompt beside the messages", p.System)
	}
	if got := textOf(p.Messages[0]); got != "Hello" {
		t.Errorf("message 0 = %q, want %q", got, "Hello")
	}
}

// TestSystemMessagesInTheListBecomeUser checks a system message said partway
// through the conversation enters as the user. Anthropic has no system input
// role, and dropping the message would lose what it said.
func TestSystemMessagesInTheListBecomeUser(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	convo.AddAssistantMessage("Hi")
	convo.AddMessage(frames.Message{Role: frames.RoleSystem, Text: "Be concise."})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages,
		sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant, sdk.MessageParamRoleUser)
	if got := textOf(p.Messages[2]); got != "Be concise." {
		t.Errorf("message 2 = %q, want what the system message said", got)
	}
}

// TestDeveloperMessagesBecomeUser checks a developer message enters as the user,
// which is what carries an asynchronous tool's late results to a provider with
// no developer role.
func TestDeveloperMessagesBecomeUser(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	convo.AddAssistantMessage("Hi")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "a tool reported late"})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages,
		sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant, sdk.MessageParamRoleUser)
	if got := textOf(p.Messages[2]); got != "a tool reported late" {
		t.Errorf("message 2 = %q, want what the developer message carried", got)
	}
}

// TestConsecutiveSameRoleMerged checks two messages of one role become one
// carrying both contents, because Anthropic requires the roles to alternate.
func TestConsecutiveSameRoleMerged(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	convo.AddUserMessage("Are you there?")
	convo.AddAssistantMessage("Hi")

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant)
	if len(p.Messages[0].Content) != 2 {
		t.Fatalf("content = %+v, want both messages kept as blocks", p.Messages[0].Content)
	}
	if got := textOf(p.Messages[0]); got != "HelloAre you there?" {
		t.Errorf("merged text = %q, want both contents in turn", got)
	}
}

// TestEmptyTextGetsAPlaceholder checks a turn that produced no text is still
// sent: Anthropic rejects a message with no content.
func TestEmptyTextGetsAPlaceholder(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("")

	p := paramsOf(t, convo, adapter.Options{})
	if got := textOf(p.Messages[0]); got != emptyText {
		t.Errorf("text = %q, want %q", got, emptyText)
	}
}

// TestSystemOnlyConversationBecomesAUserMessage checks a conversation carrying
// nothing but a system prompt is sent as a user message. Sending the prompt
// beside an empty message list is a request Anthropic rejects.
func TestSystemOnlyConversationBecomesAUserMessage(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")

	p := paramsOf(t, convo, adapter.Options{})
	if len(p.System) != 0 {
		t.Errorf("system = %+v, want none: it went into the messages", p.System)
	}
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser)
	if got := textOf(p.Messages[0]); got != "You are helpful." {
		t.Errorf("message = %q, want the prompt sent as a user message", got)
	}
}

// TestSystemOnlyConversationDoesNotMutateTheContext checks the conversation is
// left as it was: the prompt is still its prompt, not a message.
func TestSystemOnlyConversationDoesNotMutateTheContext(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	paramsOf(t, convo, adapter.Options{})

	if got := convo.System(); got != "You are helpful." {
		t.Errorf("system = %q, want the conversation's prompt untouched", got)
	}
	if got := convo.Messages(); len(got) != 0 {
		t.Errorf("messages = %+v, want the conversation still empty", got)
	}
}

// TestSystemInstructionReplacesTheContextPrompt checks that an instruction given
// for one call stands in place of the conversation's own. Anthropic has one
// field for it, so the two cannot both be sent.
func TestSystemInstructionReplacesTheContextPrompt(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be concise."})

	if len(p.System) != 1 || p.System[0].Text != "Be concise." {
		t.Errorf("system = %+v, want the call's instruction", p.System)
	}
	if got := warnings(); !strings.Contains(got, "using the instruction") {
		t.Errorf("warnings = %q, want the conflict reported", got)
	}
}

// TestSystemInstructionOnly checks an instruction given for a call, with no
// prompt on the conversation, is sent on its own and reported as no conflict.
func TestSystemInstructionOnly(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be helpful."})

	if len(p.System) != 1 || p.System[0].Text != "Be helpful." {
		t.Errorf("system = %+v, want the instruction", p.System)
	}
	if got := warnings(); got != "" {
		t.Errorf("warnings = %q, want none", got)
	}
}

// TestDeveloperMessageWithInstructionDoesNotWarn checks the conflict is about
// the system prompt alone: a developer message carries no instruction to clash
// with.
func TestDeveloperMessageWithInstructionDoesNotWarn(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	paramsOf(t, convo, adapter.Options{SystemInstruction: "Be concise."})

	if got := warnings(); got != "" {
		t.Errorf("warnings = %q, want none", got)
	}
}

// TestEnsureLastMessageIsUser checks the trailing user turn a model without
// assistant-prefill support needs, and that nothing is added when the
// conversation already ends the right way.
func TestEnsureLastMessageIsUser(t *testing.T) {
	user := sdk.NewUserMessage(sdk.NewTextBlock("hi"))
	assistant := sdk.NewAssistantMessage(sdk.NewTextBlock("hello"))

	got := EnsureLastMessageIsUser([]sdk.MessageParam{user, assistant})
	if len(got) != 3 || got[2].Role != sdk.MessageParamRoleUser {
		t.Fatalf("ending on assistant should append a user message; got %d messages", len(got))
	}

	got = EnsureLastMessageIsUser([]sdk.MessageParam{assistant, user})
	if len(got) != 2 {
		t.Fatalf("ending on user should be unchanged; got %d messages", len(got))
	}

	if got := EnsureLastMessageIsUser(nil); len(got) != 0 {
		t.Fatalf("empty list should stay empty; got %d messages", len(got))
	}
}

// TestEnsureLastMessageIsUserLeavesAToolResult checks a conversation ending on
// a tool result is left alone: a tool result is a user message already.
func TestEnsureLastMessageIsUserLeavesAToolResult(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("weather?")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: "sunny"})

	p := paramsOf(t, convo, adapter.Options{EnsureLastMessageIsUser: true})
	wantRoles(t, p.Messages,
		sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant, sdk.MessageParamRoleUser)
}

// TestMalformedToolCallIsAConversionError checks a tool call whose arguments are
// not JSON is reported as a conversion failure rather than sent: Anthropic takes
// a tool-use input as an object, so there is nothing to send.
func TestMalformedToolCallIsAConversionError(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "f", Args: []byte("not json")})

	_, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err == nil {
		t.Fatal("LLMInvocationParams succeeded, want a conversion error")
	}
	var convErr *adapter.ConversionError
	if !errors.As(err, &convErr) {
		t.Fatalf("err = %v, want an adapter.ConversionError", err)
	}
}

// TestToMessagesBuildsToolTurns checks a tool turn reaches the request as a
// tool-use block and the tool-result block answering it.
func TestToMessagesBuildsToolTurns(t *testing.T) {
	msgs := []frames.Message{
		{Role: frames.RoleUser, Text: "weather?"},
		{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{
			{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`)},
		}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{
			{ID: "c1", Name: "get_weather", Content: "sunny"},
		}},
		{Role: frames.RoleAssistant, Text: "It is sunny."},
	}
	out, err := ToMessages(msgs)
	if err != nil {
		t.Fatalf("ToMessages: %v", err)
	}
	if len(out) != 4 {
		t.Fatalf("len = %d, want 4", len(out))
	}
	b, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{
		`"tool_use"`, `"c1"`, `"get_weather"`, `"tool_result"`, `"tool_use_id":"c1"`, `sunny`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("messages JSON missing %q:\n%s", want, s)
		}
	}
}

// TestToProviderToolsFormatMapsSchema checks an advertised tool reaches the
// request with its JSON-Schema properties and required list.
func TestToProviderToolsFormatMapsSchema(t *testing.T) {
	tools := []frames.Tool{{
		Name:        "get_weather",
		Description: "Get the weather",
		Parameters: json.RawMessage(
			`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`,
		),
	}}
	out := (&Adapter{}).ToProviderToolsFormat(frames.ToolsSchema{Standard: tools})
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	b, err := json.Marshal(out[0])
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{
		`"name":"get_weather"`, `"Get the weather"`, `"location"`, `"required":["location"]`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("tool JSON %s missing %q", s, want)
		}
	}
}

// TestCacheControlMarksTheLastTwoUserMessages checks the prompt is cached up to
// the most recent user message and looked up from the one before it. Marking
// only the last would write a cache every turn and never read one back.
func TestCacheControlMarksTheLastTwoUserMessages(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("one")
	convo.AddAssistantMessage("a")
	convo.AddUserMessage("two")
	convo.AddAssistantMessage("b")
	convo.AddUserMessage("three")

	p := paramsOf(t, convo, adapter.Options{EnablePromptCaching: true})
	marked := make([]int, 0, 2)
	for i, m := range p.Messages {
		for _, blk := range m.Content {
			if blk.OfText != nil && blk.OfText.CacheControl.Type != "" {
				marked = append(marked, i)
			}
		}
	}
	// Messages 0, 2 and 4 are the user turns; the last two are 2 and 4.
	if len(marked) != 2 || marked[0] != 2 || marked[1] != 4 {
		t.Errorf("marked messages %v, want the last two user turns (2 and 4)", marked)
	}
}

// TestCacheControlIsOffByRequest checks nothing is marked when caching is not
// asked for.
func TestCacheControlIsOffByRequest(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("one")

	p := paramsOf(t, convo, adapter.Options{})
	for _, m := range p.Messages {
		for _, blk := range m.Content {
			if blk.OfText != nil && blk.OfText.CacheControl.Type != "" {
				t.Errorf("message %+v is marked, want no cache control", m)
			}
		}
	}
	if len(p.System) != 1 || p.System[0].CacheControl.Type != "" {
		t.Errorf("system = %+v, want it unmarked", p.System)
	}
}

// TestCacheControlDoesNotMutateTheSource checks the marking leaves the caller's
// messages alone: the blocks are shared with whoever converted the conversation.
func TestCacheControlDoesNotMutateTheSource(t *testing.T) {
	msgs, err := ToMessages([]frames.Message{{Role: frames.RoleUser, Text: "hello"}})
	if err != nil {
		t.Fatalf("ToMessages: %v", err)
	}
	WithCacheControlMarkers(msgs)
	if msgs[0].Content[0].OfText.CacheControl.Type != "" {
		t.Error("the source message was marked, want it left as it was")
	}
}

// TestSystemPromptKeepsTheCachedPrefixStable checks the cache breakpoint falls
// on the part of the prompt that survives between turns. The recalled context a
// memory service refreshes every turn sits outside it, so the cached prefix
// stays byte-identical and is read back rather than rewritten.
func TestSystemPromptKeepsTheCachedPrefixStable(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("hi")
	convo.SetRecall("the user likes tea")

	p := paramsOf(t, convo, adapter.Options{EnablePromptCaching: true})
	if len(p.System) != 2 {
		t.Fatalf("system = %+v, want the prompt split at its first volatile point", p.System)
	}
	if p.System[0].Text != "be helpful" || p.System[0].CacheControl.Type == "" {
		t.Errorf("block 0 = %+v, want the stable prompt marked", p.System[0])
	}
	if p.System[1].Text != "the user likes tea" || p.System[1].CacheControl.Type != "" {
		t.Errorf("block 1 = %+v, want the recalled context outside the breakpoint", p.System[1])
	}
}

// TestThoughtBecomesAThinkingBlock checks a reasoning block kept in the
// conversation is handed back as a thinking block, which is what lets the model
// carry its reasoning across a turn.
func TestThoughtBecomesAThinkingBlock(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(NewThought(Thought{Text: "let me think", Signature: "sig"}))

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser, sdk.MessageParamRoleAssistant)
	blk := p.Messages[1].Content[0]
	if blk.OfThinking == nil {
		t.Fatalf("block = %+v, want a thinking block", blk)
	}
	if blk.OfThinking.Thinking != "let me think" || blk.OfThinking.Signature != "sig" {
		t.Errorf("thinking block = %+v, want the thought it was written with", blk.OfThinking)
	}
}

// TestThoughtWithEmptyTextIsKept checks a model set to omit its reasoning still
// round-trips: the block carries no text but the signature is what matters.
func TestThoughtWithEmptyTextIsKept(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(NewThought(Thought{Signature: "sig"}))

	p := paramsOf(t, convo, adapter.Options{})
	if len(p.Messages) != 1 || p.Messages[0].Content[0].OfThinking == nil {
		t.Fatalf("messages = %+v, want the thinking block kept", p.Messages)
	}
}

// TestThoughtWithoutSignatureIsDropped checks a thought that cannot be
// round-tripped is left out rather than sent: Anthropic decrypts a thinking
// block by its signature, and refuses one without.
func TestThoughtWithoutSignatureIsDropped(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(NewThought(Thought{Text: "unsigned"}))

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser)
}

// TestNativeAnthropicMessagePassedThrough checks a message already written in
// Anthropic's own format is sent as it stands.
func TestNativeAnthropicMessagePassedThrough(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(adapter.CreateLLMSpecificMessage(
		&Adapter{}, sdk.NewAssistantMessage(sdk.NewTextBlock("written for anthropic")),
	))

	p := paramsOf(t, convo, adapter.Options{})
	if len(p.Messages) != 1 || textOf(p.Messages[0]) != "written for anthropic" {
		t.Errorf("messages = %+v, want the provider's own message", p.Messages)
	}
}

// TestAnotherProvidersMessageIsLeftOut checks a message written for a different
// provider never reaches this one.
func TestAnotherProvidersMessageIsLeftOut(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.NewLLMSpecificMessage("openai", "not for anthropic"))

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, sdk.MessageParamRoleUser)
}
