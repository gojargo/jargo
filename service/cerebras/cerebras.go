// Package cerebras provides Cerebras's OpenAI-compatible LLM service.
package cerebras

import "github.com/gojargo/jargo/service/openai"

const (
	baseURL      = "https://api.cerebras.ai/v1"
	apiKeyEnv    = "CEREBRAS_API_KEY" //nolint:gosec // G101: env var name, not a credential
	defaultModel = "llama3.1-8b"
)

// NewLLM builds a Cerebras LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("CerebrasLLM", baseURL, apiKeyEnv, defaultModel, cfg)
}
