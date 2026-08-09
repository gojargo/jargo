package openai

import (
	"bytes"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// capturedWarnings swaps the default logger for one writing into a buffer and
// returns a reader for what was logged, restoring the logger when the test ends.
// The conflict between an instruction and a conversation's own prompt is
// reported rather than returned, so reading the log is how a test sees it.
func capturedWarnings(t *testing.T) func() string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return buf.String
}

// roles returns the role of each message, which is what most of these tests
// compare.
func roles(msgs []Message) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Role)
	}
	return out
}

// wantRoles fails unless the messages carry exactly the given roles in order.
func wantRoles(t *testing.T, msgs []Message, want ...string) {
	t.Helper()
	got := roles(msgs)
	if len(got) != len(want) {
		t.Fatalf("roles = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("message %d role = %q, want %q", i, got[i], want[i])
		}
	}
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

func TestIDForLLMSpecificMessages(t *testing.T) {
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "openai" {
		t.Errorf("id = %q, want %q", got, "openai")
	}
}

// TestStandardMessagesConverted checks the conversation reaches the request in
// the order it was said, with each role as it was.
func TestStandardMessagesConverted(t *testing.T) {
	convo := frames.NewLLMContext("You are a helpful assistant.")
	convo.AddUserMessage("Hello, how are you?")
	convo.AddAssistantMessage("I'm doing well, thank you for asking!")

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, RoleSystem, RoleUser, RoleAssistant)

	want := []string{
		"You are a helpful assistant.",
		"Hello, how are you?",
		"I'm doing well, thank you for asking!",
	}
	for i, content := range want {
		if p.Messages[i].Content != content {
			t.Errorf("message %d content = %q, want %q", i, p.Messages[i].Content, content)
		}
	}
}

// TestSystemMessagesPreservedThroughout checks a system message said partway
// through the conversation stays where it was said. OpenAI takes the system
// prompt as a message like any other, so there is nothing to hoist out.
func TestSystemMessagesPreservedThroughout(t *testing.T) {
	convo := frames.NewLLMContext("You are a helpful assistant.")
	convo.AddUserMessage("Hello!")
	convo.AddAssistantMessage("Hi there!")
	convo.AddMessage(frames.Message{Role: frames.RoleSystem, Text: "Remember to be concise."})
	convo.AddUserMessage("Tell me about Go.")
	convo.AddMessage(frames.Message{Role: frames.RoleSystem, Text: "Use simple language."})
	convo.AddAssistantMessage("Go is a programming language.")

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages,
		RoleSystem, RoleUser, RoleAssistant, RoleSystem, RoleUser, RoleSystem, RoleAssistant)

	if p.Messages[3].Content != "Remember to be concise." {
		t.Errorf("message 3 = %q, want the system message said midway", p.Messages[3].Content)
	}
	if p.Messages[5].Content != "Use simple language." {
		t.Errorf("message 5 = %q, want the later system message", p.Messages[5].Content)
	}
}

// TestSystemInstructionOnly checks an instruction given for one call, with no
// prompt on the conversation, leads the messages.
func TestSystemInstructionOnly(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be helpful."})
	wantRoles(t, p.Messages, RoleSystem, RoleUser)
	if p.Messages[0].Content != "Be helpful." {
		t.Errorf("leading message = %q, want the instruction", p.Messages[0].Content)
	}
}

// TestContextSystemOnly checks the conversation's own prompt leads the messages
// when no instruction was given for the call.
func TestContextSystemOnly(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, RoleSystem, RoleUser)
	if p.Messages[0].Content != "You are helpful." {
		t.Errorf("leading message = %q, want the conversation's prompt", p.Messages[0].Content)
	}
}

// TestBothSystemInstructionAndContextSystemWarns checks that an instruction
// given for a call alongside a prompt on the conversation is reported and that
// both are sent: OpenAI takes the prompt as a message, so an instruction can
// supplement it rather than having to replace it.
func TestBothSystemInstructionAndContextSystemWarns(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be concise."})

	if got := warnings(); !strings.Contains(got, "may be unintended") {
		t.Errorf("warnings = %q, want the conflict reported as possibly unintended", got)
	}
	wantRoles(t, p.Messages, RoleSystem, RoleSystem, RoleUser)
	if p.Messages[0].Content != "Be concise." {
		t.Errorf("message 0 = %q, want the call's instruction first", p.Messages[0].Content)
	}
	if p.Messages[1].Content != "You are helpful." {
		t.Errorf("message 1 = %q, want the conversation's prompt kept", p.Messages[1].Content)
	}
}

// TestDeveloperMessageWithInstructionDoesNotWarn checks the conflict is about
// the system prompt alone: a developer message is not one, and carries no
// instruction to clash with.
func TestDeveloperMessageWithInstructionDoesNotWarn(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be concise."})

	if got := warnings(); got != "" {
		t.Errorf("warnings = %q, want none", got)
	}
	wantRoles(t, p.Messages, RoleSystem, RoleDeveloper, RoleUser)
}

// TestConflictWarningFiresOnlyOnce checks the report is made once for the
// adapter rather than on every generation of a session.
func TestConflictWarningFiresOnlyOnce(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	a := &Adapter{}
	for range 2 {
		if _, err := a.LLMInvocationParams(
			convo, adapter.Options{SystemInstruction: "Be concise."},
		); err != nil {
			t.Fatalf("LLMInvocationParams: %v", err)
		}
	}

	if got := strings.Count(warnings(), "may be unintended"); got != 1 {
		t.Errorf("reported %d times, want once", got)
	}
}

// TestDeveloperMessagesConvertedToUser checks the role an asynchronous tool's
// late results travel under: kept as it is for an endpoint that has the
// developer role, and sent as a user message for one that does not, which is
// what stops such an endpoint rejecting the turn. The conversation itself is
// left as it was either way: the conversion is how this endpoint is addressed,
// not a change to what was said.
func TestDeveloperMessagesConvertedToUser(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddUserMessage("Hello")

	kept := paramsOf(t, convo, adapter.Options{})
	if kept.Messages[0].Role != RoleDeveloper {
		t.Errorf("role = %q, want it left as the developer role", kept.Messages[0].Role)
	}

	converted := paramsOf(t, convo, adapter.Options{ConvertDeveloperToUser: true})
	if converted.Messages[0].Role != RoleUser {
		t.Errorf("role = %q, want the developer message sent as a user message",
			converted.Messages[0].Role)
	}
	if converted.Messages[0].Content != "Extra context." {
		t.Errorf("content = %q, want what the developer message carried",
			converted.Messages[0].Content)
	}
	if got := convo.Messages()[0].Role; got != frames.RoleDeveloper {
		t.Errorf("the conversation now reads %q, want the developer message untouched", got)
	}
}

// TestDeveloperConversionDoesNotAffectOtherRoles checks the conversion reaches
// the developer role and nothing else.
func TestDeveloperConversionDoesNotAffectOtherRoles(t *testing.T) {
	convo := frames.NewLLMContext("System prompt.")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Dev guidance."})
	convo.AddUserMessage("Hello")
	convo.AddAssistantMessage("Hi")

	p := paramsOf(t, convo, adapter.Options{ConvertDeveloperToUser: true})
	wantRoles(t, p.Messages, RoleSystem, RoleUser, RoleUser, RoleAssistant)
	if p.Messages[1].Content != "Dev guidance." {
		t.Errorf("content = %q, want what the developer message carried", p.Messages[1].Content)
	}
}

// TestToolTurn checks a tool turn reaches the request as an assistant message
// carrying the call and a "tool" message carrying its result.
func TestToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("weather in Paris?")
	convo.AddAssistantToolCall(frames.ToolCall{
		ID: "call_a", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`),
	})
	convo.AddToolResult(frames.ToolResult{ID: "call_a", Name: "get_weather", Content: "sunny, 20C"})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, RoleSystem, RoleUser, RoleAssistant, RoleTool)

	asst := p.Messages[2]
	if len(asst.ToolCalls) != 1 {
		t.Fatalf("assistant tool message malformed: %+v", asst)
	}
	tc := asst.ToolCalls[0]
	if tc.ID != "call_a" || tc.Type != toolTypeFunction || tc.Function.Name != "get_weather" {
		t.Errorf("tool_call fields wrong: %+v", tc)
	}
	if tc.Function.Arguments != `{"location":"Paris"}` {
		t.Errorf("tool_call arguments = %q", tc.Function.Arguments)
	}

	res := p.Messages[3]
	if res.ToolCallID != "call_a" || res.Content != "sunny, 20C" {
		t.Errorf("tool result message wrong: %+v", res)
	}
}

// TestToolCallEmptyArgsDefaults checks a call the model made with no arguments
// goes out as an empty object rather than an empty string, which is not JSON.
func TestToolCallEmptyArgsDefaults(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "now"})

	p := paramsOf(t, convo, adapter.Options{})
	if got := p.Messages[0].ToolCalls[0].Function.Arguments; got != "{}" {
		t.Errorf("empty args should default to {}, got %q", got)
	}
}

// TestToProviderToolsFormat checks an advertised tool reaches the request as an
// OpenAI function tool with its schema passed through untouched.
func TestToProviderToolsFormat(t *testing.T) {
	out := (&Adapter{}).ToProviderToolsFormat([]frames.Tool{{
		Name:        "get_weather",
		Description: "Look up the weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})
	if len(out) != 1 {
		t.Fatalf("want 1 tool, got %d", len(out))
	}
	if out[0].Type != toolTypeFunction || out[0].Function.Name != "get_weather" {
		t.Errorf("tool shape wrong: %+v", out[0])
	}
	if string(out[0].Function.Parameters) != `{"type":"object"}` {
		t.Errorf("parameters not passed through: %s", out[0].Function.Parameters)
	}
}

// TestToolsAdvertisedOnParams checks the toolset and the choice travel together:
// a conversation with no tools states no choice either, which would otherwise
// pin a model to a toolset it was not given.
func TestToolsAdvertisedOnParams(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")

	if p := paramsOf(t, convo, adapter.Options{}); p.Tools != nil || p.ToolChoice != "" {
		t.Errorf("tools = %+v, choice = %q, want neither sent", p.Tools, p.ToolChoice)
	}

	convo.SetTools([]frames.Tool{{Name: "now"}})
	convo.SetToolChoice(frames.ToolChoiceRequired)
	p := paramsOf(t, convo, adapter.Options{})
	if len(p.Tools) != 1 || p.Tools[0].Function.Name != "now" {
		t.Errorf("tools = %+v, want the advertised one", p.Tools)
	}
	if p.ToolChoice != string(frames.ToolChoiceRequired) {
		t.Errorf("choice = %q, want %q", p.ToolChoice, frames.ToolChoiceRequired)
	}
}

// TestMessagesForLogging checks a trace is given the conversation in the shape
// that goes on the wire.
func TestMessagesForLogging(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("hello")

	got := (&Adapter{}).MessagesForLogging(convo)
	if len(got) != 2 {
		t.Fatalf("logged %d messages, want 2", len(got))
	}
	if got[0]["role"] != RoleSystem || got[0]["content"] != "be helpful" {
		t.Errorf("logged message 0 = %v, want the system prompt", got[0])
	}
	if got[1]["role"] != RoleUser || got[1]["content"] != "hello" {
		t.Errorf("logged message 1 = %v, want the user message", got[1])
	}
}

// TestMessageExtraMergedOverModeledFields checks a field an endpoint reads that
// OpenAI's schema has no place for is carried, and that it wins over a modeled
// field of the same name.
func TestMessageExtraMergedOverModeledFields(t *testing.T) {
	raw, err := json.Marshal(Message{
		Role:    RoleAssistant,
		Content: "partial",
		Extra:   map[string]any{"prefix": true, "content": "overridden"},
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got["prefix"] != true {
		t.Errorf("prefix = %v, want the extra field carried", got["prefix"])
	}
	if got["content"] != "overridden" {
		t.Errorf("content = %v, want the extra field to win", got["content"])
	}
	if got["role"] != RoleAssistant {
		t.Errorf("role = %v, want the modeled field kept", got["role"])
	}
}

// TestLLMSpecificMessagePassedThrough checks a message already written in this
// format is sent as it stands, which is what an application uses to say
// something the universal conversation has no representation for.
func TestLLMSpecificMessagePassedThrough(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(adapter.CreateLLMSpecificMessage(
		&Adapter{}, Message{Role: RoleAssistant, Content: "written for openai"},
	))

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, RoleUser, RoleAssistant)
	if p.Messages[1].Content != "written for openai" {
		t.Errorf("message 1 = %+v, want the provider's own message", p.Messages[1])
	}
}

// TestAnotherProvidersMessageIsLeftOut checks a message written for a different
// provider never reaches this one, so a conversation carrying one can still be
// sent here.
func TestAnotherProvidersMessageIsLeftOut(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.NewLLMSpecificMessage("anthropic", "not for openai"))

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Messages, RoleUser)
}

// TestLLMSpecificMessageOfTheWrongTypeFails checks a message written for this
// provider but holding something it cannot read is reported rather than sent.
func TestLLMSpecificMessageOfTheWrongTypeFails(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.NewLLMSpecificMessage("openai", "a bare string"))

	_, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err == nil {
		t.Fatal("LLMInvocationParams succeeded, want a conversion error")
	}
	var convErr *adapter.ConversionError
	if !errors.As(err, &convErr) {
		t.Fatalf("err = %v, want an adapter.ConversionError", err)
	}
}
