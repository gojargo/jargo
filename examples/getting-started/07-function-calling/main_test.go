package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// TestToolsDescribeAndAnswer checks every tool's schema is valid JSON naming
// its required parameters, and its handler answers with a result.
func TestToolsDescribeAndAnswer(t *testing.T) {
	for _, tool := range tools() {
		var schema struct {
			Properties map[string]any `json:"properties"`
			Required   []string       `json:"required"`
		}
		if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
			t.Fatalf("%s: schema is not valid JSON: %v", tool.Name, err)
		}
		for _, r := range schema.Required {
			if _, ok := schema.Properties[r]; !ok {
				t.Fatalf("%s: required %q is not a property", tool.Name, r)
			}
		}
		h, ok := tool.Handler.(llm.FunctionCallHandler)
		if !ok {
			t.Fatalf("%s: handler is %T", tool.Name, tool.Handler)
		}
		var got string
		err := h(context.Background(), llm.FunctionCallParams{
			Result: func(_ context.Context, result string, _ *frames.FunctionCallResultProperties) error {
				got = result
				return nil
			},
		})
		if err != nil || !json.Valid([]byte(got)) {
			t.Fatalf("%s: result %q, err %v", tool.Name, got, err)
		}
	}
}
