package grok_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/xai/grok"
)

// TestNewLLM checks the xAI (Grok) shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "XAILLM", "grok-4.20-non-reasoning", grok.NewLLM)
}
