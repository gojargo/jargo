package openai_test

import (
	"testing"

	"github.com/gojargo/jargo/adapter/openai"
)

// The OpenAI models that predate strict JSON schema replies cannot enforce one.
func TestModelSupportsResponseSchema(t *testing.T) {
	for model, want := range map[string]bool{
		"gpt-4o-mini":       true,
		"gpt-4o-2024-08-06": true,
		"gpt-4.1":           true,
		"gpt-5-mini":        true,
		"o3":                true,
		"gpt-4o-2024-05-13": false,
		"gpt-4-turbo":       false,
		"gpt-4":             false,
		"gpt-3.5-turbo":     false,
		"o1-mini":           false,
	} {
		if got := openai.ModelSupportsResponseSchema(model); got != want {
			t.Errorf("%s: supports = %v, want %v", model, got, want)
		}
	}
}
