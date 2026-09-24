package gemini

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

const testSchema = `{"type":"object","properties":{"answer":{"type":"string"}},` +
	`"required":["answer"],"additionalProperties":false}`

// inferWithSchema runs one inference on model with testSchema and returns the
// generation config the endpoint received.
func inferWithSchema(t *testing.T, model string) map[string]any {
	t.Helper()
	srv := newGenServer(t, `{"candidates":[{"content":{"parts":[{"text":"{\"answer\": \"yes\"}"}]}}]}`)
	svc := serviceFor(srv, &testShaper{base: srv.URL}, Config{APIKey: "k", Model: model})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("is it?")
	got, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{ResponseSchema: json.RawMessage(testSchema)})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got != `{"answer": "yes"}` {
		t.Errorf("answer = %q", got)
	}
	cfg, _ := srv.body["generationConfig"].(map[string]any)
	return cfg
}

// A schema is sent as a JSON reply held to the schema.
func TestRunInferenceSendsTheResponseSchema(t *testing.T) {
	cfg := inferWithSchema(t, "gemini-2.5-flash")
	var schema any
	_ = json.Unmarshal([]byte(testSchema), &schema)
	if cfg["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType = %v, want application/json", cfg["responseMimeType"])
	}
	if !reflect.DeepEqual(cfg["responseJsonSchema"], schema) {
		t.Errorf("responseJsonSchema = %v, want %v", cfg["responseJsonSchema"], schema)
	}
}

// A model that cannot enforce a schema runs without one.
func TestRunInferenceLeavesTheSchemaOffALegacyModel(t *testing.T) {
	cfg := inferWithSchema(t, "gemini-2.0-flash-001")
	if cfg["responseMimeType"] != nil || cfg["responseJsonSchema"] != nil {
		t.Errorf("generationConfig = %v, want no schema on a model that cannot enforce one", cfg)
	}
}

// Gemini takes a JSON schema from the 2.5 models on; a model id without a
// version is assumed to support it.
func TestModelSupportsResponseSchema(t *testing.T) {
	svc := &Service{}
	if !svc.SupportsResponseSchema() {
		t.Fatal("Gemini reports it cannot enforce a response schema")
	}
	for model, want := range map[string]bool{
		"gemini-2.5-flash":       true,
		"gemini-3.6-flash":       true,
		"gemini-3.1-pro-preview": true,
		"gemini-flash-latest":    true,
		"gemini-2.0-flash-001":   false,
		"gemini-1.5-pro":         false,
	} {
		if got := svc.ModelSupportsResponseSchema(model); got != want {
			t.Errorf("%s: supports = %v, want %v", model, got, want)
		}
	}
}
