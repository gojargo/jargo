package novasonic

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

func TestIDForLLMSpecificMessages(t *testing.T) {
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "aws-nova-sonic" {
		t.Errorf("id = %q, want aws-nova-sonic", got)
	}
}

func params(t *testing.T, convo *frames.LLMContext, instruction string) Params {
	t.Helper()
	p, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{SystemInstruction: instruction})
	if err != nil {
		t.Fatalf("params: %v", err)
	}
	return p
}

// captureWarnings routes the default logger to a buffer until the returned
// function is called.
func captureWarnings(t *testing.T) (*bytes.Buffer, func()) {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	return &buf, func() { slog.SetDefault(prev) }
}

// Ports of TestAWSNovaSonicGetLLMInvocationParams in
// tests/test_get_llm_invocation_params.py. jargo keeps a conversation's system
// prompt beside its messages, so upstream's initial system message is the
// conversation's prompt here.

func TestSystemMessageExtractedAsInstruction(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	p := params(t, convo, "")
	if p.SystemInstruction != "You are helpful." || len(p.Messages) != 1 {
		t.Errorf("params = %+v, want the prompt as the instruction and one message", p)
	}
}

func TestDeveloperMessageBecomesUser(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddUserMessage("Hello")
	p := params(t, convo, "")
	if p.SystemInstruction != "" || len(p.Messages) != 2 {
		t.Errorf("params = %+v, want no instruction and both messages", p)
	}
	if p.Messages[0].Role != RoleUser {
		t.Errorf("role = %q, want the developer message as the user's", p.Messages[0].Role)
	}
}

func TestSubsequentDeveloperMessageBecomesUser(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddAssistantMessage("Hi")
	p := params(t, convo, "")
	if p.SystemInstruction != "You are helpful." || len(p.Messages) != 2 {
		t.Errorf("params = %+v, want the instruction, the developer message and the assistant's", p)
	}
}

func TestBothSystemInstructionAndSystemMessageWarns(t *testing.T) {
	buf, restore := captureWarnings(t)
	defer restore()
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	p := params(t, convo, "Be concise.")
	if p.SystemInstruction != "Be concise." {
		t.Errorf("instruction = %q, want the one given", p.SystemInstruction)
	}
	if n := strings.Count(buf.String(), "level=WARN"); n != 1 {
		t.Errorf("warnings = %d, want one: %q", n, buf.String())
	}
}

func TestBothSystemInstructionAndDeveloperMessageNoWarning(t *testing.T) {
	buf, restore := captureWarnings(t)
	defer restore()
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddUserMessage("Hello")
	p := params(t, convo, "Be concise.")
	if p.SystemInstruction != "Be concise." {
		t.Errorf("instruction = %q, want the one given", p.SystemInstruction)
	}
	if buf.Len() != 0 {
		t.Errorf("warned %q, want nothing", buf.String())
	}
}

func TestSystemInstructionOnly(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	if p := params(t, convo, "Be concise."); p.SystemInstruction != "Be concise." {
		t.Errorf("instruction = %q, want the one given", p.SystemInstruction)
	}
}

// The tests below are not ports of upstream tests.

// A tool is declared as a toolSpec carrying its object schema as JSON text.
func TestToolsAreToolSpecsWithASchemaAsText(t *testing.T) {
	got := (&Adapter{}).ToProviderToolsFormat(frames.ToolsSchema{Standard: []frames.Tool{{
		Name:        "get_weather",
		Description: "Get the weather.",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"location":{"type":"string"}},"required":["location"]}`),
	}}})
	if len(got) != 1 {
		t.Fatalf("tools = %v, want one", got)
	}
	spec, _ := got[0]["toolSpec"].(map[string]any)
	if spec["name"] != "get_weather" || spec["description"] != "Get the weather." {
		t.Errorf("spec = %v", spec)
	}
	input, _ := spec["inputSchema"].(map[string]any)
	text, _ := input["json"].(string)
	var schema map[string]any
	if err := json.Unmarshal([]byte(text), &schema); err != nil {
		t.Fatalf("schema %q: %v", text, err)
	}
	want := map[string]any{
		"type":       "object",
		"properties": map[string]any{"location": map[string]any{"type": "string"}},
		"required":   []any{"location"},
	}
	if !reflect.DeepEqual(schema, want) {
		t.Errorf("schema = %v, want %v", schema, want)
	}
}

// What has no words is left out of the history: a tool call, a tool result.
func TestToolTrafficIsLeftOutOfTheHistory(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Weather?")
	convo.AddMessage(frames.Message{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{
		{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{}`)},
	}})
	convo.AddMessage(frames.Message{Role: frames.RoleUser, ToolResults: []frames.ToolResult{
		{ID: "c1", Content: `{"temp":20}`},
	}})
	convo.AddAssistantMessage("It is 20 degrees.")
	p := params(t, convo, "")
	want := []HistoryMessage{{Role: RoleUser, Text: "Weather?"}, {Role: RoleAssistant, Text: "It is 20 degrees."}}
	if !reflect.DeepEqual(p.Messages, want) {
		t.Errorf("history = %+v, want %+v", p.Messages, want)
	}
}

// The tools are the conversation's own when it has any, else the configured
// ones, and never nil.
func TestTheSessionToolsAreSettledLikeARealtimeSessions(t *testing.T) {
	a := &Adapter{}
	service := frames.ToolsSchema{Standard: []frames.Tool{{Name: "sample"}}}
	p, err := a.LLMInvocationParams(frames.NewLLMContext(""), adapter.Options{ServiceTools: service})
	if err != nil || len(p.Tools) != 1 {
		t.Fatalf("tools = %v (%v), want the configured one", p.Tools, err)
	}
	p, _ = a.LLMInvocationParams(frames.NewLLMContext(""), adapter.Options{})
	if p.Tools == nil || len(p.Tools) != 0 {
		t.Errorf("tools = %#v, want an empty list", p.Tools)
	}
}
