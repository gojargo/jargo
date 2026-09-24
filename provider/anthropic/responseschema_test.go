package anthropic_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/service/llm"
)

const testSchema = `{"type":"object","properties":{"answer":{"type":"string"}},` +
	`"required":["answer"],"additionalProperties":false}`

// inferWithSchema runs one inference on model with testSchema and returns the
// request body the endpoint received.
func inferWithSchema(t *testing.T, model string) map[string]any {
	t.Helper()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
			"content": [{"type": "text", "text": "{\"answer\": \"yes\"}"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 4}
		}`))
	}))
	t.Cleanup(srv.Close)

	svc := anthropic.NewLLM(anthropic.Config{APIKey: "k", BaseURL: srv.URL, Model: model})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("is it?")
	got, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{ResponseSchema: json.RawMessage(testSchema)})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got != `{"answer": "yes"}` {
		t.Errorf("answer = %q", got)
	}
	return body
}

// A schema is sent as a json_schema output format.
func TestRunInferenceSendsTheResponseSchema(t *testing.T) {
	body := inferWithSchema(t, "claude-sonnet-4-5")
	var schema any
	_ = json.Unmarshal([]byte(testSchema), &schema)
	want := map[string]any{"format": map[string]any{"type": "json_schema", "schema": schema}}
	if !reflect.DeepEqual(body["output_config"], want) {
		t.Errorf("output_config = %v, want %v", body["output_config"], want)
	}
}

// A model that cannot enforce a schema runs without one.
func TestRunInferenceLeavesTheSchemaOffALegacyModel(t *testing.T) {
	if body := inferWithSchema(t, "claude-3-5-haiku-20241022"); body["output_config"] != nil {
		t.Errorf("output_config = %v, want none on a model that cannot enforce one", body["output_config"])
	}
}

// Structured outputs arrived with the 4.5 models; a model id without a version
// is assumed to support them.
func TestModelSupportsResponseSchema(t *testing.T) {
	svc := anthropic.NewLLM(anthropic.Config{APIKey: "k"})
	if !svc.SupportsResponseSchema() {
		t.Fatal("Anthropic reports it cannot enforce a response schema")
	}
	for model, want := range map[string]bool{
		"claude-haiku-4-5":           true,
		"claude-sonnet-4-5-20250929": true,
		"claude-opus-4-6":            true,
		"claude-fable-5-1":           true,
		"claude-mythos-preview":      true,
		"claude-opus-4-1-20250805":   false,
		"claude-sonnet-4-20250514":   false,
		"claude-3-7-sonnet-20250219": false,
		"claude-3-5-haiku-20241022":  false,
	} {
		if got := svc.ModelSupportsResponseSchema(model); got != want {
			t.Errorf("%s: supports = %v, want %v", model, got, want)
		}
	}
}
