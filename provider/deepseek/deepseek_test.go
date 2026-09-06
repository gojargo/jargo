package deepseek_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/deepseek"
)

// TestNewLLM checks the DeepSeek shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "DeepSeekLLM", "deepseek-v4-flash", deepseek.NewLLM)
}
