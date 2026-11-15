package crusoe_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/crusoe"
)

// TestNewLLM checks the Crusoe shim wires the right service name and default
// model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "CrusoeLLM", "openai/gpt-oss-120b", crusoe.NewLLM)
}
