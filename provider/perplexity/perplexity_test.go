package perplexity_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/perplexity"
)

// TestNewLLM checks the Perplexity shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "PerplexityLLM", "sonar", perplexity.NewLLM)
}
