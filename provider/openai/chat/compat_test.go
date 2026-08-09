package chat

import (
	"testing"

	"github.com/gojargo/jargo/adapter"
	openaiadapter "github.com/gojargo/jargo/adapter/openai"
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

// shapingAdapter converts the conversation as OpenAI does and then appends a
// message of its own, standing in for an endpoint that constrains the shape of
// a conversation beyond what OpenAI's schema says.
type shapingAdapter struct {
	openaiadapter.Adapter
	saw *[]string
}

func (a *shapingAdapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (openaiadapter.Params, error) {
	p, err := a.Adapter.LLMInvocationParams(convo, opts)
	if err != nil {
		return openaiadapter.Params{}, err
	}
	for _, m := range p.Messages {
		*a.saw = append(*a.saw, m.Role)
	}
	p.Messages = append(p.Messages, Message{Role: RoleAssistant, Content: "partial"})
	return p, nil
}

// TestAdapterDecidesWhatIsSent checks the endpoint's own adapter sees the
// converted conversation and decides what actually goes out.
func TestAdapterDecidesWhatIsSent(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")

	var saw []string
	body := bodyOf(t, Compat{
		Name:         "ShapedLLM",
		DefaultModel: "m",
		Adapter:      &shapingAdapter{saw: &saw},
	}, LLMConfig{APIKey: "k"}, convo)

	if len(saw) != 2 || saw[0] != RoleSystem || saw[1] != RoleUser {
		t.Errorf("the adapter saw %v, want the converted conversation", saw)
	}
	msgs := messagesOf(t, body)
	if len(msgs) != 3 || msgs[2]["role"] != RoleAssistant || msgs[2]["content"] != "partial" {
		t.Errorf("messages sent = %v, want the adapter's rewrite", msgs)
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
