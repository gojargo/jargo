package novita_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/novita"
)

// TestNewLLM checks the Novita AI shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "NovitaLLM", "moonshotai/kimi-k2.5", novita.NewLLM)
}
