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

// TestContextPromptSentBesideTheInstruction checks the conversation's own
// system prompt and the call's instruction both reach the model: the prompt as a
// developer message at the head of the input, the instruction in the
// instructions field. The conflict is reported, once per adapter.
func TestContextPromptSentBesideTheInstruction(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)
	a := &Adapter{}

	for range 2 {
		p, err := a.LLMInvocationParams(convo, adapter.Options{SystemInstruction: "Be concise."})
		if err != nil {
			t.Fatalf("LLMInvocationParams: %v", err)
		}
		if p.Instructions != "Be concise." {
			t.Errorf("instructions = %q, want the call's instruction", p.Instructions)
		}
		if len(p.Input) != 2 {
			t.Fatalf("input = %+v, want the prompt and the user message", p.Input)
		}
		if p.Input[0].Role != RoleDeveloper || p.Input[0].Content != "You are helpful." {
			t.Errorf("item 0 = %+v, want the conversation's prompt as a developer message", p.Input[0])
		}
	}
	if got := warnings(); strings.Count(got, "Both are being sent") != 1 {
		t.Errorf("warnings = %q, want the conflict reported once", got)
	}
}

// TestContextPromptWithoutInstructionNoWarning checks a conversation's prompt
// with no instruction for the call raises no warning: there is no conflict.
func TestContextPromptWithoutInstructionNoWarning(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	paramsOf(t, convo, adapter.Options{})
	if got := warnings(); got != "" {
		t.Errorf("warnings = %q, want none", got)
	}
}

// TestDeveloperMessageWithInstructionNoWarning checks a developer message is no
// system prompt, so it does not conflict with the call's instruction.
func TestDeveloperMessageWithInstructionNoWarning(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be concise."})
	if got := warnings(); got != "" {
		t.Errorf("warnings = %q, want none", got)
	}
	if p.Input[0].Role != RoleDeveloper || p.Instructions != "Be concise." {
		t.Errorf("params = %+v, want the developer message kept and the instruction beside it", p)
	}
}

// TestNonInitialSystemMessageNoWarning checks a system message said partway
// through is converted without a warning.
func TestNonInitialSystemMessageNoWarning(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	convo.AddMessage(frames.Message{Role: frames.RoleSystem, Text: "New instruction"})
	warnings := capturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be helpful."})
	if got := warnings(); got != "" {
		t.Errorf("warnings = %q, want none", got)
	}
	if p.Input[1].Role != RoleDeveloper || p.Input[1].Content != "New instruction" {
		t.Errorf("item 1 = %+v, want the system message as a developer message", p.Input[1])
	}
}

// TestInstructionWithMessagesSetsInstructions checks an instruction for a
// conversation with something in it goes in the instructions field.
func TestInstructionWithMessagesSetsInstructions(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be helpful."})
	if p.Instructions != "Be helpful." {
		t.Errorf("instructions = %q, want the call's instruction", p.Instructions)
	}
	if len(p.Input) != 1 || p.Input[0].Role != RoleUser {
		t.Errorf("input = %+v, want the user message alone", p.Input)
	}
}

// TestInstructionsBecomeADeveloperMessageWhenNothingSaid checks an empty
// conversation carries the instruction as a developer message. The API requires
// at least one input item when instructions are given, so sending them beside
// an empty list is a request it refuses.
func TestInstructionsBecomeADeveloperMessageWhenNothingSaid(t *testing.T) {
	p := paramsOf(t, frames.NewLLMContext(""), adapter.Options{SystemInstruction: "Be helpful."})
	if p.Instructions != "" {
		t.Errorf("instructions = %q, want none: they went into the input", p.Instructions)
	}
	if len(p.Input) != 1 {
		t.Fatalf("input = %+v, want the instruction as its one item", p.Input)
	}
	if p.Input[0].Role != RoleDeveloper || p.Input[0].Content != "Be helpful." {
		t.Errorf("item = %+v, want the instruction as a developer message", p.Input[0])
	}
}

// TestNoInstructionOmitsInstructions checks no instruction for the call leaves
// the instructions field empty, whatever the conversation holds.
func TestNoInstructionOmitsInstructions(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hi")
	if p := paramsOf(t, convo, adapter.Options{}); p.Instructions != "" {
		t.Errorf("instructions = %q, want none", p.Instructions)
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
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "openai_responses" {
		t.Errorf("id = %q, want %q", got, "openai_responses")
	}
}

// TestLLMSpecificMessagePassthrough checks a message written for this adapter
// is sent as it is, and one written for the chat-completions adapter is left
// out rather than failing the request.
func TestLLMSpecificMessagePassthrough(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	convo.AddMessage(frames.NewLLMSpecificMessage("openai_responses",
		InputItem{Type: ItemFuncCall, CallID: "x", Name: "foo", Arguments: "{}"}))
	convo.AddMessage(frames.NewLLMSpecificMessage("openai",
		map[string]any{"role": "user", "content": "chat only"}))

	items := paramsOf(t, convo, adapter.Options{}).Input
	if len(items) != 2 {
		t.Fatalf("input = %+v, want the user message and the native item", items)
	}
	if items[0].Role != RoleUser || items[1].Type != ItemFuncCall {
		t.Errorf("input = %+v, want the user message then the function call", items)
	}
}

// TestToInputMessages checks a plain conversation becomes message items, with
// the system prompt as the developer message at their head.
func TestToInputMessages(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	convo.AddAssistantMessage("hi there")

	p := paramsOf(t, convo, adapter.Options{})
	items := p.Input
	if p.Instructions != "" {
		t.Errorf("instructions = %q, want none: the prompt is in the input", p.Instructions)
	}
	if len(items) != 3 {
		t.Fatalf("items = %+v, want the prompt and the two messages", items)
	}
	if items[0].Role != RoleDeveloper || items[0].Content != "be brief" {
		t.Errorf("item 0 = %+v, want the system prompt as a developer message", items[0])
	}
	if items[1].Type != ItemMessage || items[1].Role != "user" || items[1].Content != "hello" {
		t.Errorf("item 1 = %+v, want the user message", items[1])
	}
	if items[2].Role != "assistant" || items[2].Content != "hi there" {
		t.Errorf("item 2 = %+v, want the assistant message", items[2])
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
