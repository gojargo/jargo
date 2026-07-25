package fireworks_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/fireworks"
)

// TestNewLLM checks the Fireworks AI shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "FireworksLLM", "accounts/fireworks/models/firefunction-v2", fireworks.NewLLM)
}
