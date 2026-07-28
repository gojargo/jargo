package responses

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
)

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: Config{}, Valid: false},
		{Name: "API key only", Cfg: Config{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "OpenAIResponsesLLM", NewLLM(Config{APIKey: "k"}))
	providertest.Service(t, "OpenAIResponsesHTTPLLM", NewHTTPLLM(Config{APIKey: "k"}))
}

// TestBuildInputMessages checks a plain conversation becomes message items, and
// that the system prompt is lifted out into the instructions field rather than
// traveling as a message.
func TestBuildInputMessages(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	convo.AddAssistantMessage("hi there")

	items, instructions := Config{}.buildInput(convo)
	if instructions != "be brief" {
		t.Errorf("instructions = %q, want the system prompt", instructions)
	}
	if len(items) != 2 {
		t.Fatalf("items = %+v, want the two non-system messages", items)
	}
	if items[0].Type != itemMessage || items[0].Role != "user" || items[0].Content != "hello" {
		t.Errorf("item 0 = %+v, want the user message", items[0])
	}
	if items[1].Role != "assistant" || items[1].Content != "hi there" {
		t.Errorf("item 1 = %+v, want the assistant message", items[1])
	}
}

// TestBuildInputConfiguredInstructions checks the configured prompt is used when
// the context carries no system message, and yields to one that does.
func TestBuildInputConfiguredInstructions(t *testing.T) {
	cfg := Config{Instructions: "from config"}

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	if _, got := cfg.buildInput(convo); got != "from config" {
		t.Errorf("instructions = %q, want the configured prompt", got)
	}

	withSystem := frames.NewLLMContext("from context")
	if _, got := cfg.buildInput(withSystem); got != "from context" {
		t.Errorf("instructions = %q, want the context's system message to win", got)
	}
}

// TestBuildInputToolTurn checks a tool exchange renders as the function_call and
// function_call_output items the Responses API expects, rather than the
// chat-completions message shapes.
func TestBuildInputToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("weather?")
	convo.AddAssistantToolCalls("let me check", []frames.ToolCall{
		{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city":"Paris"}`)},
	})
	convo.AddToolResults([]frames.ToolResult{{ID: "call_1", Content: "sunny"}})

	items, _ := Config{}.buildInput(convo)
	if len(items) != 4 {
		t.Fatalf("items = %+v, want user, assistant text, function call, output", items)
	}
	if items[1].Type != itemMessage || items[1].Content != "let me check" {
		t.Errorf("item 1 = %+v, want the assistant's text alongside the call", items[1])
	}
	if items[2].Type != itemFuncCall || items[2].CallID != "call_1" || items[2].Name != "get_weather" {
		t.Errorf("item 2 = %+v, want the function call", items[2])
	}
	if items[2].Arguments != `{"city":"Paris"}` {
		t.Errorf("arguments = %q, want the call's JSON", items[2].Arguments)
	}
	if items[3].Type != itemFuncOutput || items[3].CallID != "call_1" || items[3].Output != "sunny" {
		t.Errorf("item 3 = %+v, want the function output", items[3])
	}
}

// TestBuildInputEmptyArguments checks a call with no arguments sends an empty
// object, which the API requires, rather than an empty string.
func TestBuildInputEmptyArguments(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddAssistantToolCalls("", []frames.ToolCall{{ID: "c1", Name: "now"}})
	items, _ := Config{}.buildInput(convo)
	if len(items) != 1 || items[0].Arguments != "{}" {
		t.Errorf("items = %+v, want a call with empty-object arguments", items)
	}
}

// TestBuildTools checks the Responses API's flattened tool shape: the function
// fields sit on the tool rather than nested under a "function" key.
func TestBuildTools(t *testing.T) {
	convo := frames.NewLLMContext("")
	if got := buildTools(convo); got != nil {
		t.Errorf("buildTools with no tools = %+v, want nil", got)
	}

	convo.SetTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "look it up",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}})
	tools := buildTools(convo)
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

// TestEncodeBodyExtraOverrides checks configured extra fields reach the request
// and win over the modeled ones.
func TestEncodeBodyExtraOverrides(t *testing.T) {
	req := request{Model: "gpt-4.1", Stream: true}
	raw, err := encodeBody(req, map[string]any{"model": "gpt-5", "custom": 7})
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["model"] != "gpt-5" {
		t.Errorf("model = %v, want the extra field to win", m["model"])
	}
	if m["custom"] != float64(7) {
		t.Errorf("custom = %v, want the extra field carried through", m["custom"])
	}
}

// TestNewRequestOmitsUnset checks the optional controls are absent from the wire
// when unset, leaving the API defaults in place.
func TestNewRequestOmitsUnset(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")

	raw, err := encodeBody(Config{Model: "gpt-4.1"}.newRequest(convo, false), nil)
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, key := range []string{
		"temperature", "top_p", "max_output_tokens", "service_tier", "tools",
		"instructions", "previous_response_id",
	} {
		if _, ok := m[key]; ok {
			t.Errorf("%s is present when unset, want it omitted", key)
		}
	}
	if m["store"] != false {
		t.Errorf("store = %v, want false so conversations are not retained", m["store"])
	}
	if m["stream"] != true {
		t.Errorf("stream = %v, want true", m["stream"])
	}
}
