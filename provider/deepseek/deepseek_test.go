package deepseek_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/deepseek"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// TestNewLLM checks the DeepSeek shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "DeepSeekLLM", "deepseek-flash", compatLLM)
}

// compatLLM builds the service from the shared OpenAI-compatible config alone,
// which is what the shared assertions are written against.
func compatLLM(cfg chat.LLMConfig) *chat.LLMService {
	return deepseek.NewLLM(deepseek.LLMConfig{LLMConfig: cfg})
}

// DeepSeek's chat API has no JSON schema response format, so an inference
// given one runs without it.
func TestDeepSeekCannotEnforceAResponseSchema(t *testing.T) {
	if compatLLM(chat.LLMConfig{APIKey: "k"}).SupportsResponseSchema() {
		t.Error("DeepSeek reports it can enforce a response schema")
	}
}

// TestDisablesThinkingByDefault checks a service left at its defaults turns
// thinking off, and leaves out the OpenAI parameters DeepSeek does not support.
func TestDisablesThinkingByDefault(t *testing.T) {
	seed, maxCompletion := 7, 100
	body := generate(t, deepseek.LLMConfig{
		Seed:                &seed,
		MaxCompletionTokens: &maxCompletion,
		ServiceTier:         "flex",
	}, plainConvo())

	thinking, ok := body["thinking"].(map[string]any)
	if !ok || len(thinking) != 1 || thinking["type"] != "disabled" {
		t.Errorf("thinking = %v, want {type: disabled}", body["thinking"])
	}
	for _, field := range []string{"seed", "max_completion_tokens", "service_tier"} {
		if v, present := body[field]; present {
			t.Errorf("%s = %v, want it left out", field, v)
		}
	}
}

// TestThinkingLeftToDeepSeek checks an empty thinking configuration sends no
// thinking field, which leaves the choice to DeepSeek's own default.
func TestThinkingLeftToDeepSeek(t *testing.T) {
	body := generate(t, deepseek.LLMConfig{Thinking: &deepseek.ThinkingConfig{}}, plainConvo())
	if v, present := body["thinking"]; present {
		t.Errorf("thinking = %v, want it left out", v)
	}
}

// TestThinkingKeepsTheCallersExtra checks the thinking field is sent alongside
// the caller's own extra fields, and that the caller's map is not written to.
func TestThinkingKeepsTheCallersExtra(t *testing.T) {
	extra := map[string]any{"user_field": float64(1)}
	body := generate(t, deepseek.LLMConfig{
		Extra:    extra,
		Thinking: &deepseek.ThinkingConfig{Type: "enabled"},
	}, plainConvo())

	thinking, ok := body["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" {
		t.Errorf("thinking = %v, want {type: enabled}", body["thinking"])
	}
	if body["user_field"] != float64(1) {
		t.Errorf("user_field = %v, want the caller's extra field", body["user_field"])
	}
	if _, written := extra["thinking"]; written {
		t.Error("the caller's Extra map was written to")
	}
}

// TestThinkingFromTheCallersExtraWins checks a thinking field the caller set in
// Extra wins over the service's own, as Extra does over every other parameter.
func TestThinkingFromTheCallersExtraWins(t *testing.T) {
	body := generate(t, deepseek.LLMConfig{
		Extra: map[string]any{"thinking": map[string]any{"type": "enabled"}},
	}, plainConvo())

	thinking, ok := body["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" {
		t.Errorf("thinking = %v, want the caller's {type: enabled}", body["thinking"])
	}
}

// TestToolCallTurnCarriesReasoningContent checks the request that follows a
// tool call sends reasoning_content on the assistant message that made it.
// Without it DeepSeek refuses the request in thinking mode.
func TestToolCallTurnCarriesReasoningContent(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("Weather in LA?")
	convo.AddAssistantToolCall(frames.ToolCall{
		ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"location": "LA"}`),
	})
	convo.AddToolResult(frames.ToolResult{ID: "call_1", Name: "get_weather", Content: `{"temperature": "75"}`})

	body := generate(t, deepseek.LLMConfig{}, convo)

	msgs, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("messages = %v, want a list", body["messages"])
	}
	for _, raw := range msgs {
		m, ok := raw.(map[string]any)
		if !ok {
			t.Fatalf("message = %v, want an object", raw)
		}
		v, present := m["reasoning_content"]
		if m["role"] == chat.RoleAssistant {
			if !present || v != "" {
				t.Errorf("assistant message = %v, want an empty reasoning_content", m)
			}
		} else if present {
			t.Errorf("%s message = %v, want no reasoning_content", m["role"], m)
		}
	}
}

// plainConvo is a conversation of one user message.
func plainConvo() *frames.LLMContext {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	return convo
}

// generate runs one generation against a fake endpoint and reports the request
// body it received.
func generate(t *testing.T, cfg deepseek.LLMConfig, convo *frames.LLMContext) map[string]any {
	t.Helper()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	cfg.APIKey = "k"
	cfg.BaseURL = srv.URL
	if err := deepseek.NewLLM(cfg).Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return body
}
