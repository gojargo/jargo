package interactions

import (
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/service/llm"
)

// interactionReply is a one-shot interaction replying text.
func interactionReply(text string) reply {
	b, _ := json.Marshal(map[string]any{
		"status": "completed",
		"steps": []any{
			map[string]any{"type": "thought"},
			map[string]any{"type": "model_output", "content": []any{map[string]any{"type": "text", "text": text}}},
		},
	})
	return reply{status: http.StatusOK, body: string(b)}
}

// TestRunInferenceWithLLMContext checks RunInference returns the interaction's
// text from a one-shot, unstored request.
func TestRunInferenceWithLLMContext(t *testing.T) {
	api := newFakeAPI(t, interactionReply("Hello! How can I help?"))
	svc := makeService(api, Config{})

	got, err := svc.RunInference(t.Context(), userContext("Hello, world!"), llm.InferenceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got != "Hello! How can I help?" {
		t.Errorf("text = %q", got)
	}
	request := api.request(t, 0)
	if _, ok := request["stream"]; ok {
		t.Errorf("stream = %v, want it left out", request["stream"])
	}
	if request["store"] != false {
		t.Errorf("store = %v", request["store"])
	}
	want := []any{map[string]any{
		"type": "user_input", "content": []any{map[string]any{"type": "text", "text": "Hello, world!"}},
	}}
	if !reflect.DeepEqual(request["input"], want) {
		t.Errorf("input = %v", request["input"])
	}
}

// TestRunInferenceClientError checks a failure from the API is returned.
func TestRunInferenceClientError(t *testing.T) {
	api := newFakeAPI(t, reply{status: http.StatusInternalServerError, body: "Interactions API Error"})
	svc := makeService(api, Config{})

	_, err := svc.RunInference(t.Context(), userContext("Hi"), llm.InferenceOptions{})
	if !errors.Is(err, errStatus) {
		t.Errorf("err = %v, want the API's failure", err)
	}
}

// TestRunInferenceOverrides checks the system instruction and max tokens given
// for the call override the service's.
func TestRunInferenceOverrides(t *testing.T) {
	api := newFakeAPI(t, interactionReply("ok"))
	svc := makeService(api, Config{MaxTokens: 1000})
	svc.SetSystemInstruction("Original.")

	if _, err := svc.RunInference(t.Context(), userContext("Hi"), llm.InferenceOptions{
		MaxTokens: 50, SystemInstruction: "Summarize.",
	}); err != nil {
		t.Fatal(err)
	}
	request := api.request(t, 0)
	if request["system_instruction"] != "Summarize." || genConfig(request)["max_output_tokens"] != float64(50) {
		t.Errorf("request = %v", request)
	}
}

// TestRunInferenceResponseSchema checks a response schema is sent as a JSON
// response format.
func TestRunInferenceResponseSchema(t *testing.T) {
	api := newFakeAPI(t, interactionReply(`{"answer": "yes"}`))
	svc := makeService(api, Config{})
	schema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`)

	got, err := svc.RunInference(t.Context(), userContext("Yes?"), llm.InferenceOptions{ResponseSchema: schema})
	if err != nil {
		t.Fatal(err)
	}
	if got != `{"answer": "yes"}` {
		t.Errorf("text = %q", got)
	}
	var wantSchema any
	_ = json.Unmarshal(schema, &wantSchema)
	want := map[string]any{"type": "text", "mime_type": "application/json", "schema": wantSchema}
	if !reflect.DeepEqual(api.request(t, 0)["response_format"], want) {
		t.Errorf("response_format = %v", api.request(t, 0)["response_format"])
	}
}

// Compile-time checks of the interfaces the base looks for.
var (
	_ llm.Inferencer              = (*LLMService)(nil)
	_ llm.ToolGenerator           = (*LLMService)(nil)
	_ llm.ResponseSchemaSupporter = (*LLMService)(nil)
	_ llm.TraceRenderer           = (*LLMService)(nil)
)

// TestSettingsCarryOver checks the service takes the Gemini service's settings.
func TestSettingsCarryOver(t *testing.T) {
	svc := NewLLM(Config{APIKey: "k", Thinking: &ThinkingConfig{Level: "low"}})
	settings, ok := svc.Settings().(*LLMSettings)
	if !ok || settings.Thinking.Or(ThinkingConfig{}).Level != "low" {
		t.Errorf("settings = %#v", svc.Settings())
	}
}
