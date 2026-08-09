package adapter_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/gemini"
	openaiadapter "github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
)

// TestCustomToolsReachTheirOwnAdapter checks a tool written for one format is
// advertised by that provider and by no other.
func TestCustomToolsReachTheirOwnAdapter(t *testing.T) {
	schema := frames.ToolsSchema{
		Standard: []frames.Tool{{Name: "get_weather"}},
		Custom: map[frames.AdapterType][]any{
			frames.AdapterTypeOpenAI: {openaiadapter.Tool{
				Raw: map[string]any{"type": "web_search_preview"},
			}},
			frames.AdapterTypeGemini: {map[string]any{"google_search": map[string]any{}}},
		},
	}

	openaiTools := (&openaiadapter.Adapter{}).ToProviderToolsFormat(schema)
	if len(openaiTools) != 2 {
		t.Fatalf("openai tools = %+v, want the standard one and openai's custom", openaiTools)
	}
	if openaiTools[1].Raw["type"] != "web_search_preview" {
		t.Errorf("custom tool = %+v, want openai's own", openaiTools[1])
	}

	geminiTools := (&gemini.Adapter{}).ToProviderToolsFormat(schema)
	if len(geminiTools) != 2 {
		t.Fatalf("gemini tools = %+v, want the declarations and gemini's custom", geminiTools)
	}
	if _, ok := geminiTools[1]["google_search"]; !ok {
		t.Errorf("custom tool = %+v, want gemini's own", geminiTools[1])
	}
}

// TestCustomToolGoesOutWhole checks a provider-native tool reaches the wire
// exactly as it was written, rather than being forced into the function shape
// the modeled fields describe.
func TestCustomToolGoesOutWhole(t *testing.T) {
	raw, err := json.Marshal(openaiadapter.Tool{
		Raw: map[string]any{"type": "web_search_preview"},
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(raw) != `{"type":"web_search_preview"}` {
		t.Errorf("encoded = %s, want the tool exactly as it was written", raw)
	}
}

// TestStandardToolStillEncodesAsAFunction checks the escape hatch changes
// nothing for a tool that fits the schema.
func TestStandardToolStillEncodesAsAFunction(t *testing.T) {
	raw, err := json.Marshal(openaiadapter.Tool{
		Type:     "function",
		Function: openaiadapter.Function{Name: "get_weather"},
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got["type"] != "function" {
		t.Errorf("encoded = %s, want a function tool", raw)
	}
}

// TestCustomToolsForRejectsAnotherType checks a custom tool holding something
// its adapter cannot read is reported rather than sent as whatever it is.
func TestCustomToolsForRejectsAnotherType(t *testing.T) {
	schema := frames.ToolsSchema{
		Custom: map[frames.AdapterType][]any{
			frames.AdapterTypeGemini: {"a bare string"},
		},
	}
	_, err := adapter.CustomToolsFor[map[string]any](schema, frames.AdapterTypeGemini)
	if err == nil {
		t.Fatal("CustomToolsFor succeeded, want a conversion error")
	}
	var convErr *adapter.ConversionError
	if !errors.As(err, &convErr) {
		t.Fatalf("err = %v, want an adapter.ConversionError", err)
	}
}

// TestGeminiOmitsDeclarationsWithoutStandardTools checks a conversation
// advertising only a custom tool sends that alone, with no empty declarations
// block beside it.
func TestGeminiOmitsDeclarationsWithoutStandardTools(t *testing.T) {
	schema := frames.ToolsSchema{
		Custom: map[frames.AdapterType][]any{
			frames.AdapterTypeGemini: {map[string]any{"google_search": map[string]any{}}},
		},
	}
	tools := (&gemini.Adapter{}).ToProviderToolsFormat(schema)
	if len(tools) != 1 {
		t.Fatalf("tools = %+v, want only the custom one", tools)
	}
	if _, ok := tools[0]["functionDeclarations"]; ok {
		t.Errorf("tools = %+v, want no empty declarations block", tools)
	}
}
