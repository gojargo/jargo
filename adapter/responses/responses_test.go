package responses

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

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

// TestSystemRoleBecomesDeveloper checks a system message said partway through
// the conversation enters as a developer message. The Responses API has no
// system role: developer is the role it reserves for instructions given out of
// band, which is what a system message is.
func TestSystemRoleBecomesDeveloper(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.Message{Role: frames.RoleSystem, Text: "be concise"})

	items := paramsOf(t, convo, adapter.Options{}).Input
	if len(items) != 2 {
		t.Fatalf("items = %+v, want both messages", items)
	}
	if items[1].Role != RoleDeveloper || items[1].Content != "be concise" {
		t.Errorf("item 1 = %+v, want the system message sent as a developer message", items[1])
	}
}

// TestDeveloperRoleKept checks a developer message stays one: it is already the
// role the API takes.
func TestDeveloperRoleKept(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "a tool reported late"})

	items := paramsOf(t, convo, adapter.Options{}).Input
	if items[0].Role != RoleDeveloper {
		t.Errorf("role = %q, want it left as the developer role", items[0].Role)
	}
}

// TestSystemInstructionReplacesTheContextPrompt checks that an instruction given
// for one call stands in place of the conversation's own: the API has one
// instructions field, so the two cannot both be sent.
func TestSystemInstructionReplacesTheContextPrompt(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	warnings := capturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "be concise"})

	if p.Instructions != "be concise" {
		t.Errorf("instructions = %q, want the call's own", p.Instructions)
	}
	if got := warnings(); !strings.Contains(got, "using the instruction") {
		t.Errorf("warnings = %q, want the conflict reported", got)
	}
}

// TestInstructionsBecomeADeveloperMessageWhenNothingSaid checks a conversation
// with nothing said yet carries the instruction as a developer message. The API
// requires at least one input item when instructions are given, so sending them
// beside an empty list is a request it refuses.
func TestInstructionsBecomeADeveloperMessageWhenNothingSaid(t *testing.T) {
	convo := frames.NewLLMContext("be brief")

	p := paramsOf(t, convo, adapter.Options{})
	if p.Instructions != "" {
		t.Errorf("instructions = %q, want none: they went into the input", p.Instructions)
	}
	if len(p.Input) != 1 {
		t.Fatalf("input = %+v, want the instruction as its one item", p.Input)
	}
	if p.Input[0].Role != RoleDeveloper || p.Input[0].Content != "be brief" {
		t.Errorf("item = %+v, want the instruction as a developer message", p.Input[0])
	}
}

// TestNoInstructionsAndNothingSaid checks an empty conversation with no prompt
// stays empty rather than gaining a message with nothing in it.
func TestNoInstructionsAndNothingSaid(t *testing.T) {
	p := paramsOf(t, frames.NewLLMContext(""), adapter.Options{})
	if len(p.Input) != 0 || p.Instructions != "" {
		t.Errorf("params = %+v, want nothing to send", p)
	}
}

func TestIDForLLMSpecificMessages(t *testing.T) {
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "openai" {
		t.Errorf("id = %q, want %q", got, "openai")
	}
}

// TestToInputMessages checks a plain conversation becomes message items, and
// that the system prompt is lifted out into the instructions field rather than
// traveling as a message.
func TestToInputMessages(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	convo.AddAssistantMessage("hi there")

	p := paramsOf(t, convo, adapter.Options{})
	items, instructions := p.Input, p.Instructions
	if instructions != "be brief" {
		t.Errorf("instructions = %q, want the system prompt", instructions)
	}
	if len(items) != 2 {
		t.Fatalf("items = %+v, want the two non-system messages", items)
	}
	if items[0].Type != ItemMessage || items[0].Role != "user" || items[0].Content != "hello" {
		t.Errorf("item 0 = %+v, want the user message", items[0])
	}
	if items[1].Role != "assistant" || items[1].Content != "hi there" {
		t.Errorf("item 1 = %+v, want the assistant message", items[1])
	}
}

// TestToInputToolTurn checks a tool exchange renders as the function_call and
// function_call_output items the Responses API expects, rather than the
// chat-completions message shapes.
func TestToInputToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("weather?")
	convo.AddAssistantMessage("let me check")
	convo.AddAssistantToolCall(frames.ToolCall{
		ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city":"Paris"}`),
	})
	convo.AddToolResult(frames.ToolResult{ID: "call_1", Content: "sunny"})

	items := paramsOf(t, convo, adapter.Options{}).Input
	if len(items) != 4 {
		t.Fatalf("items = %+v, want user, assistant text, function call, output", items)
	}
	if items[1].Type != ItemMessage || items[1].Content != "let me check" {
		t.Errorf("item 1 = %+v, want the assistant's text alongside the call", items[1])
	}
	if items[2].Type != ItemFuncCall || items[2].CallID != "call_1" || items[2].Name != "get_weather" {
		t.Errorf("item 2 = %+v, want the function call", items[2])
	}
	if items[2].Arguments != `{"city":"Paris"}` {
		t.Errorf("arguments = %q, want the call's JSON", items[2].Arguments)
	}
	if items[3].Type != ItemFuncOutput || items[3].CallID != "call_1" || items[3].Output != "sunny" {
		t.Errorf("item 3 = %+v, want the function output", items[3])
	}
}

// TestToInputEmptyArguments checks a call with no arguments sends an empty
// object, which the API requires, rather than an empty string.
func TestToInputEmptyArguments(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "now"})
	items := paramsOf(t, convo, adapter.Options{}).Input
	if len(items) != 1 || items[0].Arguments != "{}" {
		t.Errorf("items = %+v, want a call with empty-object arguments", items)
	}
}

// TestToProviderToolsFormat checks the Responses API's flattened tool shape: the function
// fields sit on the tool rather than nested under a "function" key.
func TestToProviderToolsFormat(t *testing.T) {
	convo := frames.NewLLMContext("")
	if got := paramsOf(t, convo, adapter.Options{}).Tools; got != nil {
		t.Errorf("buildTools with no tools = %+v, want nil", got)
	}

	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "look it up",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})
	tools := paramsOf(t, convo, adapter.Options{}).Tools
	if len(tools) != 1 {
		t.Fatalf("tools = %+v, want one", tools)
	}
	if tools[0].Type != "function" || tools[0].Name != "get_weather" {
		t.Errorf("tool = %+v, want a flattened function tool", tools[0])
	}

	raw, err := json.Marshal(tools[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, nested := m["function"]; nested {
		t.Errorf("tool = %s, want the function fields flattened onto it", raw)
	}
	if m["name"] != "get_weather" {
		t.Errorf("tool name = %v, want it at the top level", m["name"])
	}
}
