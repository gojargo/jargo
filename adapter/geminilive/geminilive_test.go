package geminilive_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/gemini"
	"github.com/gojargo/jargo/adapter/geminilive"
	"github.com/gojargo/jargo/frames"
)

// partText is the text of a content's first part.
func partText(t *testing.T, content map[string]any) string {
	t.Helper()
	parts, _ := content["parts"].([]map[string]any)
	if len(parts) == 0 {
		t.Fatalf("content %v has no parts", content)
	}
	text, _ := parts[0]["text"].(string)
	return text
}

// Messages with no tool calls are converted just like Gemini's.
func TestStandardMessagesConvertedToGeminiFormat(t *testing.T) {
	convo := frames.NewLLMContext("You are a helpful assistant.")
	convo.AddUserMessage("Hello, how are you?")
	convo.AddAssistantMessage("I'm doing well, thank you for asking!")

	got, err := (&geminilive.Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("live: %v", err)
	}
	want, err := (&gemini.Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("gemini: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("params = %+v, want Gemini's %+v", got, want)
	}
}

// Messages written for Gemini Live are kept, and those written for Gemini are
// left out.
func TestLLMSpecificMessageFiltering(t *testing.T) {
	content := func(text string) map[string]any {
		return map[string]any{"role": "model", "parts": []map[string]any{{"text": text}}}
	}
	convo := frames.NewLLMContext("")
	convo.AddMessage(adapter.CreateLLMSpecificMessage(&gemini.Adapter{}, content("Gemini specific response")))
	convo.AddMessage(adapter.CreateLLMSpecificMessage(&geminilive.Adapter{}, content("Gemini Live specific response")))

	params, err := (&geminilive.Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("params: %v", err)
	}
	if len(params.Contents) != 1 {
		t.Fatalf("contents = %v, want one", params.Contents)
	}
	if got := partText(t, params.Contents[0]); got != "Gemini Live specific response" {
		t.Errorf("text = %q, want the Gemini Live message", got)
	}
}

// A tool call and its result are summarized as text: the call as a model turn,
// its result as a user turn.
func TestToolCallAndResultSummarizedAsText(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("What's the weather?")
	convo.AddAssistantToolCall(frames.ToolCall{
		ID: "call-1", Name: "get_weather", Args: json.RawMessage(`{"location": "Paris"}`),
	})
	convo.AddToolResult(frames.ToolResult{ID: "call-1", Content: `{"conditions": "nice"}`})

	params, err := (&geminilive.Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("params: %v", err)
	}
	if len(params.Contents) != 3 {
		t.Fatalf("contents = %v, want three", params.Contents)
	}

	call := params.Contents[1]
	if call["role"] != "model" {
		t.Errorf("call role = %v, want model", call["role"])
	}
	parts, _ := call["parts"].([]map[string]any)
	if _, ok := parts[0]["functionCall"]; ok {
		t.Error("the call was sent as a functionCall, want text")
	}
	if text := partText(t, call); !strings.Contains(text, "get_weather") || !strings.Contains(text, "Paris") {
		t.Errorf("call text = %q, want the function and its args", text)
	}

	result := params.Contents[2]
	if result["role"] != "user" {
		t.Errorf("result role = %v, want user", result["role"])
	}
	parts, _ = result["parts"].([]map[string]any)
	if _, ok := parts[0]["functionResponse"]; ok {
		t.Error("the result was sent as a functionResponse, want text")
	}
	if text := partText(t, result); !strings.Contains(text, "get_weather") || !strings.Contains(text, "nice") {
		t.Errorf("result text = %q, want the function and what it returned", text)
	}
}

// A message calling several functions is summarized as one message.
func TestParallelToolCallsSummarizedTogether(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{
		{ID: "call-1", Name: "get_weather", Args: json.RawMessage(`{}`)},
		{ID: "call-2", Name: "get_restaurant", Args: json.RawMessage(`{}`)},
	}})

	params, err := (&geminilive.Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("params: %v", err)
	}
	if len(params.Contents) != 1 {
		t.Fatalf("contents = %v, want one", params.Contents)
	}
	parts, _ := params.Contents[0]["parts"].([]map[string]any)
	if len(parts) != 1 {
		t.Fatalf("parts = %v, want one", parts)
	}
	text, _ := parts[0]["text"].(string)
	if !strings.Contains(text, "get_weather") || !strings.Contains(text, "get_restaurant") {
		t.Errorf("text = %q, want both functions", text)
	}
}

// A result whose call is not in the conversation is named with the fallback.
// (Not a port of an upstream test: it covers the fallback name upstream uses.)
func TestAResultWithoutItsCallIsNamedWithTheFallback(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddToolResult(frames.ToolResult{ID: "lost", Content: "42"})

	got := geminilive.ConvertToolCallsToText(convo.MessagesFor("gemini-live"))
	if len(got) != 1 || got[0].Text != "[Function tool_call_result returned 42]" {
		t.Errorf("messages = %+v, want the fallback name", got)
	}
}

// The session's tools are the conversation's own when it has any, else the
// configured ones, with the built-in tools riding along.
func TestTheSessionToolsAreSettledLikeARealtimeSessions(t *testing.T) {
	a := &geminilive.Adapter{}
	a.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "delegate"}})
	service := frames.ToolsSchema{Standard: []frames.Tool{{Name: "sample"}}}

	declared := func(convo *frames.LLMContext, service frames.ToolsSchema) []string {
		t.Helper()
		params, err := a.LLMInvocationParams(convo, adapter.Options{ServiceTools: service})
		if err != nil {
			t.Fatalf("params: %v", err)
		}
		var names []string
		for _, group := range params.Tools {
			decls, _ := group["functionDeclarations"].([]map[string]any)
			for _, d := range decls {
				name, _ := d["name"].(string)
				names = append(names, name)
			}
		}
		return names
	}

	withTools := frames.NewLLMContext("")
	withTools.SetTools([]frames.Tool{{Name: "from_context"}})
	if got := declared(withTools, service); !reflect.DeepEqual(got, []string{"from_context", "delegate"}) {
		t.Errorf("tools = %v, want the conversation's and the built-in", got)
	}
	if got := declared(frames.NewLLMContext(""), service); !reflect.DeepEqual(got, []string{"sample", "delegate"}) {
		t.Errorf("tools = %v, want the configured ones and the built-in", got)
	}
	if got := declared(frames.NewLLMContext(""), frames.ToolsSchema{}); !reflect.DeepEqual(got, []string{"delegate"}) {
		t.Errorf("tools = %v, want the built-in alone", got)
	}
}
