package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/gojargo/jargo/frames"
)

func TestToToolsMapsSchema(t *testing.T) {
	tools := []frames.Tool{{
		Name:        "get_weather",
		Description: "Get the weather",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
	}}
	out := toTools(tools)
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	b, err := json.Marshal(out[0])
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{`"name":"get_weather"`, `"Get the weather"`, `"location"`, `"required":["location"]`} {
		if !strings.Contains(s, want) {
			t.Fatalf("tool JSON %s missing %q", s, want)
		}
	}
}

func TestNewParamsAppliesSampling(t *testing.T) {
	temp, topP := 0.3, 0.9
	topK := int64(40)
	s := NewLLM(Config{Temperature: &temp, TopP: &topP, TopK: &topK})
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hi")
	b, err := json.Marshal(s.newParams(convo))
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	for _, want := range []string{`"temperature":0.3`, `"top_p":0.9`, `"top_k":40`} {
		if !strings.Contains(got, want) {
			t.Fatalf("params JSON %s missing %q", got, want)
		}
	}
}

func TestNewParamsOmitsUnsetSampling(t *testing.T) {
	s := NewLLM(Config{})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	b, err := json.Marshal(s.newParams(convo))
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	for _, absent := range []string{"temperature", "top_p", "top_k"} {
		if strings.Contains(got, absent) {
			t.Fatalf("params JSON %s should omit %q when unset", got, absent)
		}
	}
}

func TestToUsageMapsFields(t *testing.T) {
	u := toUsage(sdk.Usage{
		InputTokens:              100,
		OutputTokens:             20,
		CacheReadInputTokens:     80,
		CacheCreationInputTokens: 10,
	})
	if u.PromptTokens != 100 || u.CompletionTokens != 20 ||
		u.CacheReadTokens != 80 || u.CacheCreationTokens != 10 || u.TotalTokens != 120 {
		t.Fatalf("usage = %+v", u)
	}
}

func TestToMessagesBuildsToolTurns(t *testing.T) {
	msgs := []frames.Message{
		{Role: frames.RoleUser, Text: "weather?"},
		{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{
			{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`)},
		}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{
			{ID: "c1", Name: "get_weather", Content: "sunny", IsError: false},
		}},
		{Role: frames.RoleAssistant, Text: "It is sunny."},
	}
	out := toMessages(msgs)
	if len(out) != 4 {
		t.Fatalf("len = %d, want 4", len(out))
	}
	b, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{`"tool_use"`, `"c1"`, `"get_weather"`, `"tool_result"`, `"tool_use_id":"c1"`, `sunny`} {
		if !strings.Contains(s, want) {
			t.Fatalf("messages JSON missing %q:\n%s", want, s)
		}
	}
}
