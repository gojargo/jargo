package realtime

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
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "openai-realtime" {
		t.Errorf("id = %q, want openai-realtime", got)
	}
}

// weatherTool is the tool upstream's adapter tests describe.
func weatherTool() frames.Tool {
	return frames.Tool{
		Name:        "get_weather",
		Description: "Get the weather in a given location",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"location":{"type":"string","description":"The city, e.g. San Francisco"},
			"format":{"type":"string","enum":["celsius","fahrenheit"],"description":"The temperature unit to use."}
		},"required":["location","format"]}`),
	}
}

// asJSON normalizes a value through its JSON encoding, for comparison.
func asJSON(t *testing.T, v any) any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return out
}

// weatherExpected is the tool in the Realtime API's form.
const weatherExpected = `{
	"type": "function",
	"name": "get_weather",
	"description": "Get the weather in a given location",
	"parameters": {
		"type": "object",
		"properties": {
			"location": {"type": "string", "description": "The city, e.g. San Francisco"},
			"format": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit to use."}
		},
		"required": ["location", "format"]
	}
}`

// Ports of the OpenAI Realtime cases of tests/test_function_calling_adapters.py.

func TestOpenAIRealtimeAdapter(t *testing.T) {
	got := (&Adapter{}).ToProviderToolsFormat(frames.ToolsSchema{Standard: []frames.Tool{weatherTool()}})
	var want any
	_ = json.Unmarshal([]byte("["+weatherExpected+"]"), &want)
	if g := asJSON(t, got); !reflect.DeepEqual(g, want) {
		t.Errorf("tools = %v, want %v", g, want)
	}
}

func TestOpenAIRealtimeAdapterWithCustomTools(t *testing.T) {
	search := map[string]any{"type": "tool_search"}
	got := (&Adapter{}).ToProviderToolsFormat(frames.ToolsSchema{
		Standard: []frames.Tool{weatherTool()},
		Custom:   map[frames.AdapterType][]any{frames.AdapterTypeOpenAI: {search}},
	})
	var want any
	_ = json.Unmarshal([]byte("["+weatherExpected+`, {"type": "tool_search"}]`), &want)
	if g := asJSON(t, got); !reflect.DeepEqual(g, want) {
		t.Errorf("tools = %v, want %v", g, want)
	}
}

// Ports of TestOpenAIRealtimeGetLLMInvocationParams. jargo keeps a
// conversation's system prompt beside its messages, so upstream's initial
// system message is the conversation's prompt here.

func params(t *testing.T, convo *frames.LLMContext, instruction string) Params {
	t.Helper()
	p, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{SystemInstruction: instruction})
	if err != nil {
		t.Fatalf("params: %v", err)
	}
	return p
}

func TestSystemMessageExtractedAsInstruction(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	p := params(t, convo, "")
	if p.SystemInstruction != "You are helpful." || len(p.Messages) != 1 {
		t.Errorf("params = %+v, want the prompt as the instruction and one item", p)
	}
}

func TestDeveloperMessageBecomesUser(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	convo.AddUserMessage("Hello")
	p := params(t, convo, "")
	// Developer converted to user, then packed with the other user message.
	if p.SystemInstruction != "" || len(p.Messages) != 1 {
		t.Errorf("params = %+v, want no instruction and one packed item", p)
	}
}

func TestSubsequentDeveloperMessageBecomesUser(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Extra context."})
	p := params(t, convo, "")
	if p.SystemInstruction != "You are helpful." || len(p.Messages) != 1 {
		t.Fatalf("params = %+v, want the instruction and one item", p)
	}
	if p.Messages[0].Role != "user" {
		t.Errorf("role = %q, want the developer message as the user's", p.Messages[0].Role)
	}
}

func TestEmptyMessages(t *testing.T) {
	p := params(t, frames.NewLLMContext(""), "")
	if len(p.Messages) != 0 || p.SystemInstruction != "" {
		t.Errorf("params = %+v, want nothing", p)
	}
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

// Port of TestRealtimePackedHistoryKeepsNonAscii: a history packed into one
// text item keeps non-ASCII text readable.
func TestPackedHistoryIsNotEscaped(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("मेरा ऑर्डर कहाँ है?")
	convo.AddAssistantMessage("आपका ऑर्डर कल पहुँचेगा।")
	p := params(t, convo, "")
	text := p.Messages[0].Content[0].Text
	if !strings.Contains(text, "मेरा ऑर्डर कहाँ है?") || !strings.Contains(text, "आपका ऑर्डर कल पहुँचेगा।") {
		t.Errorf("packed text = %q, want the history readable", text)
	}
}

// The tests below are not ports of upstream tests.

// A single user message goes in as it stands, as an input_text item.
func TestASingleUserMessageGoesInAsItStands(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	p := params(t, convo, "")
	item := p.Messages[0]
	if item.Type != "message" || item.Role != "user" || item.ID == "" ||
		len(item.Content) != 1 || item.Content[0].Type != "input_text" || item.Content[0].Text != "Hello" {
		t.Errorf("item = %+v, want the user's text", item)
	}
}

// The session's tools are the conversation's own when it has any, else the
// configured ones, with the built-in tools riding along.
func TestTheSessionToolsAreSettledLikeARealtimeSessions(t *testing.T) {
	a := &Adapter{}
	a.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "delegate"}})
	names := func(convo *frames.LLMContext, service frames.ToolsSchema) []string {
		p, err := a.LLMInvocationParams(convo, adapter.Options{ServiceTools: service})
		if err != nil {
			t.Fatalf("params: %v", err)
		}
		var out []string
		for _, tool := range p.Tools {
			name, _ := tool["name"].(string)
			out = append(out, name)
		}
		return out
	}
	service := frames.ToolsSchema{Standard: []frames.Tool{{Name: "sample"}}}
	withTools := frames.NewLLMContext("")
	withTools.SetTools([]frames.Tool{{Name: "from_context"}})
	if got := names(withTools, service); !reflect.DeepEqual(got, []string{"from_context", "delegate"}) {
		t.Errorf("tools = %v, want the conversation's and the built-in", got)
	}
	if got := names(frames.NewLLMContext(""), service); !reflect.DeepEqual(got, []string{"sample", "delegate"}) {
		t.Errorf("tools = %v, want the configured ones and the built-in", got)
	}
}
