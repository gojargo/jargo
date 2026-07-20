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

func TestSupportsPrefill(t *testing.T) {
	cases := map[string]bool{
		// Direct model ids that still support prefill.
		"claude-haiku-4-5":           true,
		"claude-haiku-4-5-20251001":  true,
		"claude-sonnet-4-5":          true,
		"claude-3-5-sonnet-20241022": true,
		"claude-opus-4-1":            true,
		// Direct ids that dropped prefill.
		"claude-opus-4-8":   false,
		"claude-sonnet-4-6": false,
		// Bedrock ids (region-prefixed) are matched as substrings.
		"us.anthropic.claude-3-5-haiku-20241022-v1:0": true,
		"us.anthropic.claude-sonnet-4-6-v1:0":         false,
		"us.anthropic.claude-opus-4-8-v1:0":           false,
		// Non-Claude models are unaffected (nothing is injected).
		"amazon.titan-text-express-v1": true,
		"":                             true,
	}
	for model, want := range cases {
		if got := supportsPrefill(model); got != want {
			t.Errorf("supportsPrefill(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestEnsureLastMessageIsUser(t *testing.T) {
	user := sdk.NewUserMessage(sdk.NewTextBlock("hi"))
	assistant := sdk.NewAssistantMessage(sdk.NewTextBlock("hello"))

	got := ensureLastMessageIsUser([]sdk.MessageParam{user, assistant})
	if len(got) != 3 || got[2].Role != sdk.MessageParamRoleUser {
		t.Fatalf("ending on assistant should append a user message; got %d messages", len(got))
	}

	got = ensureLastMessageIsUser([]sdk.MessageParam{assistant, user})
	if len(got) != 2 {
		t.Fatalf("ending on user should be unchanged; got %d messages", len(got))
	}

	if got := ensureLastMessageIsUser(nil); len(got) != 0 {
		t.Fatalf("empty list should stay empty; got %d messages", len(got))
	}
}

func TestNewParamsPrefillFixup(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.AddUserMessage("hi")
	convo.AddAssistantMessage("hello") // context ends on an assistant message

	// A no-prefill model gets a trailing user message injected.
	noPrefill := NewLLM(Config{Model: "claude-opus-4-8"}).newParams(convo).Messages
	if n := len(noPrefill); n != 3 || noPrefill[n-1].Role != sdk.MessageParamRoleUser {
		t.Fatalf("no-prefill model: want 3 messages ending in a user turn, got %d", n)
	}

	// A prefill-supported model keeps the assistant message last.
	prefill := NewLLM(Config{Model: "claude-haiku-4-5"}).newParams(convo).Messages
	if n := len(prefill); n != 2 || prefill[n-1].Role != sdk.MessageParamRoleAssistant {
		t.Fatalf("prefill model: want the assistant message to stay last, got %d messages", n)
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
