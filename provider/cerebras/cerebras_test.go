package cerebras_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/cerebras"
)

// TestNewLLM checks the Cerebras shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "CerebrasLLM", "gpt-oss-120b", cerebras.NewLLM)
}
