package baseten_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/baseten"
)

// TestNewLLM checks the Baseten shim wires the right service name and default
// model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "BasetenLLM", "zai-org/GLM-5.3", baseten.NewLLM)
}
