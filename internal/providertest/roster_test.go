package providertest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/cerebras"
	"github.com/gojargo/jargo/provider/deepseek"
	"github.com/gojargo/jargo/provider/fireworks"
	"github.com/gojargo/jargo/provider/groq"
	"github.com/gojargo/jargo/provider/inception"
	"github.com/gojargo/jargo/provider/mistral"
	"github.com/gojargo/jargo/provider/nebius"
	"github.com/gojargo/jargo/provider/novita"
	"github.com/gojargo/jargo/provider/nvidia"
	"github.com/gojargo/jargo/provider/ollama"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/openrouter"
	"github.com/gojargo/jargo/provider/perplexity"
	"github.com/gojargo/jargo/provider/qwen"
	"github.com/gojargo/jargo/provider/sambanova"
	"github.com/gojargo/jargo/provider/sarvam"
	"github.com/gojargo/jargo/provider/together"
	"github.com/gojargo/jargo/provider/xai/grok"
)

// TestDeveloperRoleRoster pins, for every OpenAI-compatible provider, whether
// its endpoint has a developer role. The role carries the late results of an
// asynchronous tool, and an endpoint without it rejects the turn outright, so
// the answer is a property of the endpoint rather than a preference: it belongs
// in one list where it can be read off and corrected.
func TestDeveloperRoleRoster(t *testing.T) {
	roster := []struct {
		name  string
		build CompatLLMBuilder
		// role is what a developer message is expected to be sent under.
		role string
	}{
		// Cerebras maps the developer role to its own developer instruction
		// layer, which sits above user instructions in the prompt hierarchy, so
		// the message is sent as written.
		{"cerebras", cerebras.NewLLM, chat.RoleDeveloper},
		{"deepseek", deepseek.NewLLM, chat.RoleUser},
		{"inception", inception.NewLLM, chat.RoleUser},
		{"mistral", mistral.NewLLM, chat.RoleUser},
		{"nebius", nebius.NewLLM, chat.RoleUser},
		{"ollama", ollama.NewLLM, chat.RoleUser},
		{"openrouter", openrouter.NewLLM, chat.RoleUser},
		{"perplexity", perplexity.NewLLM, chat.RoleUser},
		{"qwen", qwen.NewLLM, chat.RoleUser},
		{"sambanova", sambanova.NewLLM, chat.RoleUser},
		{"sarvam", sarvam.NewLLM, chat.RoleUser},
		{"together", together.NewLLM, chat.RoleUser},

		{"openai", chat.NewLLM, chat.RoleDeveloper},
		{"fireworks", fireworks.NewLLM, chat.RoleDeveloper},
		{"groq", groq.NewLLM, chat.RoleDeveloper},
		{"novita", novita.NewLLM, chat.RoleDeveloper},
		{"nvidia", nvidia.NewLLM, chat.RoleDeveloper},
		{"grok", grok.NewLLM, chat.RoleDeveloper},
	}

	for _, p := range roster {
		t.Run(p.name, func(t *testing.T) {
			if got := developerRoleAsSent(t, p.build); got != p.role {
				t.Errorf("a developer message was sent as %q, want %q", got, p.role)
			}
		})
	}
}

// developerRoleAsSent returns the role a developer message travels under once
// the service has converted the conversation for its endpoint.
func developerRoleAsSent(t *testing.T, build CompatLLMBuilder) string {
	t.Helper()

	// Content is decoded loosely: an endpoint that merges two same-role messages
	// sends the result as a list of parts rather than a string, so the shape
	// varies across the roster even though the question asked of it does not.
	var body struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("what is the score?")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "a tool reported late"})

	svc := build(chat.LLMConfig{APIKey: "k", BaseURL: srv.URL})
	if err := svc.Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for _, m := range body.Messages {
		if slices.Contains(contentTexts(m.Content), "a tool reported late") {
			return m.Role
		}
	}
	t.Fatalf("the developer message never reached the endpoint: %+v", body.Messages)
	return ""
}

// contentTexts returns every text a message's content carries, whether it was
// sent as a plain string or as a list of parts.
func contentTexts(raw json.RawMessage) []string {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []string{text}
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return nil
	}
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		out = append(out, p.Text)
	}
	return out
}
