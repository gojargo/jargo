package responses

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// A schema is sent as a strict json_schema text format, over HTTP and over
// the WebSocket service's one-shot path alike.
func TestRunInferenceSendsTheResponseSchema(t *testing.T) {
	const schema = `{"type":"object","properties":{"answer":{"type":"string"}},` +
		`"required":["answer"],"additionalProperties":false}`
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body = nil
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		_, _ = w.Write([]byte(
			`{"output":[{"content":[{"type":"output_text","text":"{\"answer\": \"no\"}"}]}]}`,
		))
	}))
	t.Cleanup(srv.Close)

	var parsed any
	_ = json.Unmarshal([]byte(schema), &parsed)
	want := map[string]any{"format": map[string]any{
		"type": "json_schema", "name": "response", "schema": parsed, "strict": true,
	}}
	cfg := Config{APIKey: "k", BaseURL: srv.URL, Model: "gpt-4.1"}
	for name, svc := range map[string]llm.Inferencer{"http": NewHTTPLLM(cfg), "websocket": NewLLM(cfg)} {
		convo := frames.NewLLMContext("")
		convo.AddUserMessage("is it?")
		got, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{ResponseSchema: json.RawMessage(schema)})
		if err != nil {
			t.Fatalf("%s: RunInference: %v", name, err)
		}
		if got != `{"answer": "no"}` {
			t.Errorf("%s: answer = %q", name, got)
		}
		if !reflect.DeepEqual(body["text"], want) {
			t.Errorf("%s: text = %v, want %v", name, body["text"], want)
		}
	}
}
