package chat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// testSchema is the schema the inference tests ask for.
const testSchema = `{"type":"object","properties":{"answer":{"type":"string"}},` +
	`"required":["answer"],"additionalProperties":false}`

// inferWithSchema runs one inference on model with schema and returns the
// request body the endpoint received.
func inferWithSchema(t *testing.T, model, schema string) map[string]any {
	t.Helper()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"answer\": \"yes\"}"}}]}`))
	}))
	t.Cleanup(srv.Close)

	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL, Model: model})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("is it?")
	got, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{ResponseSchema: json.RawMessage(schema)})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got != `{"answer": "yes"}` {
		t.Errorf("answer = %q, want the completion's content", got)
	}
	return body
}

// A schema is sent as a strict json_schema response format.
func TestRunInferenceSendsTheResponseSchema(t *testing.T) {
	body := inferWithSchema(t, "gpt-4.1", testSchema)
	var schema any
	_ = json.Unmarshal([]byte(testSchema), &schema)
	want := map[string]any{
		"type":        "json_schema",
		"json_schema": map[string]any{"name": "response", "schema": schema, "strict": true},
	}
	if !reflect.DeepEqual(body["response_format"], want) {
		t.Errorf("response_format = %v, want %v", body["response_format"], want)
	}
}

// Without a schema no response format is sent.
func TestRunInferenceWithoutASchemaSendsNoFormat(t *testing.T) {
	if body := inferWithSchema(t, "gpt-4.1", ""); body["response_format"] != nil {
		t.Errorf("response_format = %v, want none", body["response_format"])
	}
}

// A model that cannot enforce a schema runs without one.
func TestRunInferenceLeavesTheSchemaOffALegacyModel(t *testing.T) {
	if body := inferWithSchema(t, "gpt-4", testSchema); body["response_format"] != nil {
		t.Errorf("response_format = %v, want none on a model that cannot enforce one", body["response_format"])
	}
}
