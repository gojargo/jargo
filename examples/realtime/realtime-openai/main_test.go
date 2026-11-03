package main

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// TestToolsAreDescribed checks every tool's schema is valid JSON naming its
// required parameters, and carries its handler.
func TestToolsAreDescribed(t *testing.T) {
	for _, tool := range append(tools(), newsTool()) {
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
		if _, ok := tool.Handler.(llm.FunctionCallHandler); !ok {
			t.Fatalf("%s: handler is %T", tool.Name, tool.Handler)
		}
	}
}

// TestNewsToolIsAddedLater checks the conversation starts without the news tool,
// which is added at runtime.
func TestNewsToolIsAddedLater(t *testing.T) {
	names := func(ts []frames.Tool) []string {
		var out []string
		for _, tool := range ts {
			out = append(out, tool.Name)
		}
		return out
	}
	for _, name := range names(tools()) {
		if name == "get_news" {
			t.Fatalf("tools = %v, want get_news added later", names(tools()))
		}
	}
	if newsTool().Name != "get_news" {
		t.Errorf("news tool = %q", newsTool().Name)
	}
}
