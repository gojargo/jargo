package gemini

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

// wantRoles fails unless the contents carry exactly the given roles in order.
func wantRoles(t *testing.T, contents []map[string]any, want ...string) {
	t.Helper()
	if len(contents) != len(want) {
		t.Fatalf("contents = %v, want %d of them with roles %v", contents, len(want), want)
	}
	for i := range want {
		if got := contents[i][keyRole]; got != want[i] {
			t.Errorf("content %d role = %v, want %q", i, got, want[i])
		}
	}
}

// textAt returns the text of a content's single text part.
func textAt(t *testing.T, c map[string]any) string {
	t.Helper()
	parts, ok := c[keyParts].([]map[string]any)
	if !ok || len(parts) == 0 {
		t.Fatalf("content %v carries no parts", c)
	}
	text, _ := parts[0][keyText].(string)
	return text
}

func TestIDForLLMSpecificMessages(t *testing.T) {
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "google" {
		t.Errorf("id = %q, want %q", got, "google")
	}
}

// TestStandardMessagesConverted checks the conversation reaches the request with
// the assistant as the model and the instruction carried beside the contents.
func TestStandardMessagesConverted(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	convo.AddAssistantMessage("Hi there")

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Contents, roleUser, roleModel)
	if p.SystemInstruction != "You are helpful." {
		t.Errorf("instruction = %q, want the conversation's prompt", p.SystemInstruction)
	}
}

// TestSystemAndDeveloperMessagesBecomeUser checks that a system or developer
// message said partway through the conversation enters as the user, Gemini
// having neither input role.
func TestSystemAndDeveloperMessagesBecomeUser(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	convo.AddMessage(frames.Message{Role: frames.RoleSystem, Text: "Be concise."})
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "a tool reported late"})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Contents, roleUser, roleUser, roleUser)
	if got := textAt(t, p.Contents[1]); got != "Be concise." {
		t.Errorf("content 1 = %q, want what the system message said", got)
	}
	if got := textAt(t, p.Contents[2]); got != "a tool reported late" {
		t.Errorf("content 2 = %q, want what the developer message carried", got)
	}
}

// TestSystemOnlyConversationBecomesAUserMessage checks a conversation carrying
// nothing but a system prompt is sent as a user message rather than beside an
// empty content list, which Gemini rejects.
func TestSystemOnlyConversationBecomesAUserMessage(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")

	p := paramsOf(t, convo, adapter.Options{})
	if p.SystemInstruction != "" {
		t.Errorf("instruction = %q, want none: it went into the contents", p.SystemInstruction)
	}
	wantRoles(t, p.Contents, roleUser)
	if got := textAt(t, p.Contents[0]); got != "You are helpful." {
		t.Errorf("content = %q, want the prompt sent as a user message", got)
	}
}

// TestSystemInstructionReplacesTheContextPrompt checks that an instruction given
// for one call stands in place of the conversation's own: Gemini has one field
// for it, so the two cannot both be sent.
func TestSystemInstructionReplacesTheContextPrompt(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	warnings := capturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be concise."})

	if p.SystemInstruction != "Be concise." {
		t.Errorf("instruction = %q, want the call's own", p.SystemInstruction)
	}
	if got := warnings(); !strings.Contains(got, "using the instruction") {
		t.Errorf("warnings = %q, want the conflict reported", got)
	}
}

// TestSystemRepeatedWhenOnlyToolTurns checks a conversation of nothing but tool
// turns is given the instruction again as a user message. Gemini reads the
// instruction as framing rather than as something to answer, so without it the
// model has no prose to reply to.
func TestSystemRepeatedWhenOnlyToolTurns(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: "sunny"})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Contents, roleModel, roleUser, roleUser)
	if got := textAt(t, p.Contents[2]); got != "You are helpful." {
		t.Errorf("trailing content = %q, want the instruction said again", got)
	}
	if p.SystemInstruction != "You are helpful." {
		t.Errorf("instruction = %q, want it still carried beside the contents", p.SystemInstruction)
	}
}

// TestSystemNotRepeatedWhenProseIsPresent checks the repeat above is only for a
// conversation with nothing to answer.
func TestSystemNotRepeatedWhenProseIsPresent(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("weather?")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: "sunny"})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Contents, roleUser, roleModel, roleUser)
}

// TestEmptyContentsDropped checks a turn that carries nothing is left out:
// Gemini rejects a content with no parts.
func TestEmptyContentsDropped(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.Message{Role: frames.RoleAssistant, ToolCalls: nil, Text: ""})

	p := paramsOf(t, convo, adapter.Options{})
	for _, c := range p.Contents {
		parts, ok := c[keyParts].([]map[string]any)
		if !ok || len(parts) == 0 {
			t.Errorf("content %v carries no parts, want it dropped", c)
		}
	}
}

// TestToolResultNamedAfterItsCall checks a result is named after the call it
// answers, which is what an asynchronous tool's late result carries only an id
// for.
func TestToolResultNamedAfterItsCall(t *testing.T) {
	msgs := []frames.Message{
		{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{{ID: "c1", Name: "get_weather"}}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "c1", Content: "sunny"}}},
	}
	b, err := json.Marshal(ToContents(msgs))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"name":"get_weather"`) {
		t.Errorf("contents = %s, want the result named after its call", b)
	}
}

// TestUnnamedToolResultGetsAPlaceholder checks a result whose call is nowhere in
// the conversation still carries a name, which Gemini requires.
func TestUnnamedToolResultGetsAPlaceholder(t *testing.T) {
	msgs := []frames.Message{
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "gone", Content: "sunny"}}},
	}
	b, err := json.Marshal(ToContents(msgs))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"name":"`+unnamedToolResult+`"`) {
		t.Errorf("contents = %s, want the placeholder name", b)
	}
}

func TestToContentsToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("weather in Paris?")
	convo.AddAssistantToolCall(frames.ToolCall{
		ID: "call_0", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`),
	})
	convo.AddToolResult(frames.ToolResult{ID: "call_0", Name: "get_weather", Content: "sunny"})

	b, err := json.Marshal(ToContents(convo.Messages()))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(b)
	// user turn + tool-result turn use role "user"; the assistant tool-call turn
	// uses role "model". Map keys marshal in alphabetical order.
	wants := []string{
		`"role":"user"`,
		`"role":"model"`,
		`"functionCall":{"args":{"location":"Paris"},"id":"call_0","name":"get_weather"}`,
		`"functionResponse":{"id":"call_0","name":"get_weather","response":{"value":"sunny"}}`,
	}
	for _, want := range wants {
		if !strings.Contains(got, want) {
			t.Errorf("contents missing %s\nin: %s", want, got)
		}
	}
}

func TestFunctionResponseDict(t *testing.T) {
	// A JSON object passes through unchanged.
	raw, ok := FunctionResponseDict(`{"temp":20}`).(json.RawMessage)
	if !ok || string(raw) != `{"temp":20}` {
		t.Errorf("object should pass through, got %v", FunctionResponseDict(`{"temp":20}`))
	}
	// A plain string is wrapped under "value".
	m, ok := FunctionResponseDict("sunny").(map[string]any)
	if !ok || m["value"] != "sunny" {
		t.Errorf("non-object should wrap as {value}, got %v", FunctionResponseDict("sunny"))
	}
}

func TestToProviderToolsFormatStripsAdditionalProperties(t *testing.T) {
	out := (&Adapter{}).ToProviderToolsFormat([]frames.Tool{{
		Name:        "get_weather",
		Description: "Look up the weather",
		Parameters:  json.RawMessage(`{"type":"object","additionalProperties":false,"properties":{"loc":{"type":"string"}}}`),
	}})
	b, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(b)
	if strings.Contains(got, "additionalProperties") {
		t.Errorf("additionalProperties should be stripped: %s", got)
	}
	if !strings.Contains(got, `"functionDeclarations"`) || !strings.Contains(got, `"name":"get_weather"`) {
		t.Errorf("declaration shape wrong: %s", got)
	}
}
