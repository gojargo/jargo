package chat

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// bodyOf runs one generation against a recording server and returns the request
// body the service sent.
func bodyOf(t *testing.T, c Compat, cfg LLMConfig, convo *frames.LLMContext) map[string]any {
	t.Helper()
	srv := newLLMServer(t, sse(contentChunk("ok")))
	cfg.BaseURL = srv.URL
	svc := NewCompatLLM(c, cfg)
	if err := svc.Generate(t.Context(), convo, func(string) error { return nil }); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return srv.body
}

// messagesOf returns the role/content pairs of the messages in a request body.
func messagesOf(t *testing.T, body map[string]any) []map[string]any {
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

// TestShapeMessagesRewritesWhatIsSent checks the hook sees the converted
// conversation and decides what actually goes out.
func TestShapeMessagesRewritesWhatIsSent(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")

	var saw []string
	body := bodyOf(t, Compat{
		Name:         "ShapedLLM",
		DefaultModel: "m",
		ShapeMessages: func(msgs []Message) []Message {
			for _, m := range msgs {
				saw = append(saw, m.Role)
			}
			return append(msgs, Message{Role: RoleAssistant, Content: "partial"})
		},
	}, LLMConfig{APIKey: "k"}, convo)

	if len(saw) != 2 || saw[0] != RoleSystem || saw[1] != RoleUser {
		t.Errorf("the hook saw %v, want the converted conversation", saw)
	}
	msgs := messagesOf(t, body)
	if len(msgs) != 3 || msgs[2]["role"] != RoleAssistant || msgs[2]["content"] != "partial" {
		t.Errorf("messages sent = %v, want the hook's rewrite", msgs)
	}
}

// TestMessageExtraIsMerged checks a field OpenAI's schema has no place for is
// encoded alongside the modeled ones, and that it wins where the names collide.
func TestMessageExtraIsMerged(t *testing.T) {
	raw, err := json.Marshal(Message{
		Role:    RoleAssistant,
		Content: "half a thought",
		Extra:   map[string]any{"prefix": true},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["prefix"] != true || got["role"] != RoleAssistant || got["content"] != "half a thought" {
		t.Errorf("encoded message = %v, want the extra field alongside the modeled ones", got)
	}
	if _, ok := got["extra"]; ok {
		t.Error("the extra map was encoded as a field of its own")
	}

	raw, err = json.Marshal(Message{Role: RoleAssistant, Extra: map[string]any{"role": "user"}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["role"] != RoleUser {
		t.Errorf("role = %v, want the extra field to win over the modeled one", got["role"])
	}
}

// TestPlainCompatEndpointSendsTheSameBodyAsBefore pins what an endpoint that
// declares no departures sends, so that serving the ones that do departs from
// nothing here.
func TestPlainCompatEndpointSendsTheSameBodyAsBefore(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "aside"})

	body := bodyOf(t,
		Compat{Name: "PlainLLM", DefaultModel: "default-model"},
		LLMConfig{APIKey: "k"},
		convo,
	)

	want := map[string]bool{"model": true, "messages": true, "stream": true, "stream_options": true}
	for k := range body {
		if !want[k] {
			t.Errorf("body carries an unexpected field %q: %v", k, body)
		}
	}
	if body["model"] != "default-model" || body["stream"] != true {
		t.Errorf("body = %v, want the model and the streaming flag", body)
	}
	msgs := messagesOf(t, body)
	if len(msgs) != 3 {
		t.Fatalf("messages = %v, want the system prompt and both turns", msgs)
	}
	roles := []string{RoleSystem, RoleUser, RoleDeveloper}
	for i, role := range roles {
		if msgs[i]["role"] != role {
			t.Errorf("message %d role = %v, want %q", i, msgs[i]["role"], role)
		}
		if _, ok := msgs[i]["prefix"]; ok {
			t.Errorf("message %d carries a field no endpoint asked for: %v", i, msgs[i])
		}
	}
}
