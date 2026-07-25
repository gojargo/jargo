package inception_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/inception"
)

// TestNewLLM checks the Inception shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "InceptionLLM", "mercury-2", inception.NewLLM)
}
