package ollama_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/ollama"
)

// TestNewLLM checks the Ollama shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "OllamaLLM", "llama2", ollama.NewLLM)
}
