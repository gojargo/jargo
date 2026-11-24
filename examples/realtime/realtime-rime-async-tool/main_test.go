package main

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// TestToolsAreDescribed checks every tool's schema is valid JSON naming its
// required parameters, and carries its handler.
func TestToolsAreDescribed(t *testing.T) {
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
		if _, ok := tool.Handler.(llm.FunctionCallHandler); !ok {
			t.Fatalf("%s: handler is %T", tool.Name, tool.Handler)
		}
	}
}

// TestWeatherToolIsAsynchronous checks the weather tool is not canceled when
// the user interrupts.
func TestWeatherToolIsAsynchronous(t *testing.T) {
	for _, tool := range tools() {
		if tool.CancelOnInterruption == nil || *tool.CancelOnInterruption {
			t.Errorf("%s: CancelOnInterruption = %v, want false", tool.Name, tool.CancelOnInterruption)
		}
	}
}

// TestWeatherReportsIntermediateResultFirst checks the weather tool reports an
// intermediate result that starts no reply before it waits on the slow call.
func TestWeatherReportsIntermediateResultFirst(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var results []string
	var props []*frames.FunctionCallResultProperties
	err := getCurrentWeather(ctx, llm.FunctionCallParams{
		Arguments: json.RawMessage(`{"location": "San Francisco, CA", "format": "fahrenheit"}`),
		Result: func(_ context.Context, result string, p *frames.FunctionCallResultProperties) error {
			results = append(results, result)
			props = append(props, p)
			cancel() // Stop the handler before the slow call answers.
			return nil
		},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %v, want one intermediate result", results)
	}
	var got map[string]any
	if err = json.Unmarshal([]byte(results[0]), &got); err != nil {
		t.Fatalf("result is not valid JSON: %v", err)
	}
	if got["status"] != "checking" || got["location"] != "San Francisco, CA" {
		t.Errorf("intermediate result = %v", got)
	}
	p := props[0]
	if p == nil || p.IsFinal == nil || *p.IsFinal || p.RunLLM == nil || *p.RunLLM {
		t.Errorf("intermediate properties = %+v, want IsFinal and RunLLM false", p)
	}
}
