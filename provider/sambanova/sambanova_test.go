package sambanova_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/sambanova"
)

// TestNewLLM checks the SambaNova Cloud shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "SambaNovaLLM", "Meta-Llama-3.3-70B-Instruct", sambanova.NewLLM)
}
