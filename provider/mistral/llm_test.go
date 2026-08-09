package mistral_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/mistral"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// sentBody runs one generation against a recording endpoint and returns the
// request body Mistral would have received.
func sentBody(t *testing.T, cfg chat.LLMConfig, convo *frames.LLMContext) map[string]any {
	t.Helper()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	if err := mistral.NewLLM(cfg).Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return body
}

// messages returns the messages of a recorded request body.
func messages(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	raw, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("body has no messages: %v", body)
	}
	out := make([]map[string]any, 0, len(raw))
	for _, m := range raw {
		msg, ok := m.(map[string]any)
		if !ok {
			t.Fatalf("message is not an object: %v", m)
		}
		out = append(out, msg)
	}
	return out
}

// TestSeedIsSentUnderMistralsName checks the sampling seed reaches Mistral under
// the name it reads, and only that one.
func TestSeedIsSentUnderMistralsName(t *testing.T) {
	seed := 42
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")

	body := sentBody(t, chat.LLMConfig{APIKey: "k", Seed: &seed}, convo)
	if body["random_seed"] != float64(seed) {
		t.Errorf("random_seed = %v, want %d", body["random_seed"], seed)
	}
	if _, ok := body["seed"]; ok {
		t.Errorf("body carries a seed field Mistral does not read: %v", body)
	}

	// A caller's own extra fields survive the rewrite.
	body = sentBody(t, chat.LLMConfig{
		APIKey: "k", Seed: &seed, Extra: map[string]any{"safe_prompt": true},
	}, convo)
	if body["safe_prompt"] != true || body["random_seed"] != float64(seed) {
		t.Errorf("body = %v, want both the caller's field and the renamed seed", body)
	}

	// The caller's config is not rewritten under them.
	cfg := chat.LLMConfig{APIKey: "k", Seed: &seed}
	mistral.NewLLM(cfg)
	if cfg.Seed == nil || *cfg.Seed != seed {
		t.Error("the constructor emptied the seed on the caller's own config")
	}
}

// TestConversationIsShapedForMistral checks the constraints Mistral puts on a
// history end to end: the developer role it has no place for, the assistant
// message a tool result must be followed by, and the prefix flag that message
// then needs.
func TestConversationIsShapedForMistral(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("weather in Paris?")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "a tool reported late"})
	convo.AddAssistantToolCall(frames.ToolCall{ID: "call_a", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "call_a", Name: "get_weather", Content: "sunny"})

	msgs := messages(t, sentBody(t, chat.LLMConfig{APIKey: "k"}, convo))
	if len(msgs) != 6 {
		t.Fatalf("messages = %v, want the five converted plus the inserted assistant message", msgs)
	}
	if msgs[2]["role"] != chat.RoleUser {
		t.Errorf("developer message role = %v, want it sent as a user message", msgs[2]["role"])
	}
	if msgs[4]["role"] != chat.RoleTool {
		t.Errorf("message 4 role = %v, want the tool result", msgs[4]["role"])
	}
	last := msgs[5]
	if last["role"] != chat.RoleAssistant || last["content"] != " " {
		t.Errorf("last message = %v, want a minimal assistant message after the tool result", last)
	}
	if last["prefix"] != true {
		t.Errorf("last message = %v, want it marked as a prefix", last)
	}
}
