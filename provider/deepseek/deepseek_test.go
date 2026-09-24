package deepseek_test

import (
	"testing"

	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/provider/deepseek"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// TestNewLLM checks the DeepSeek shim wires the right service name and
// default model into the shared OpenAI-compatible client.
func TestNewLLM(t *testing.T) {
	providertest.CompatLLM(t, "DeepSeekLLM", "deepseek-flash", deepseek.NewLLM)
}

// DeepSeek's chat API has no JSON schema response format, so an inference
// given one runs without it.
func TestDeepSeekCannotEnforceAResponseSchema(t *testing.T) {
	if deepseek.NewLLM(chat.LLMConfig{APIKey: "k"}).SupportsResponseSchema() {
		t.Error("DeepSeek reports it can enforce a response schema")
	}
}
