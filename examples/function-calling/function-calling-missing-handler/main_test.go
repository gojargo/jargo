package main

import (
	"encoding/json"
	"testing"
)

// TestWeatherToolHasNoHandler checks the weather tool is advertised with a
// valid schema and, as the example requires, no handler.
func TestWeatherToolHasNoHandler(t *testing.T) {
	tool := weatherTool()
	var schema struct {
		Properties map[string]any `json:"properties"`
		Required   []string       `json:"required"`
	}
	if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	for _, r := range schema.Required {
		if _, ok := schema.Properties[r]; !ok {
			t.Fatalf("required %q is not a property", r)
		}
	}
	if tool.Handler != nil {
		t.Fatalf("handler is %T, want none", tool.Handler)
	}
}
