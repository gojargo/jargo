// Package together provides Together AI's OpenAI-compatible LLM service.
package together

import "github.com/gojargo/jargo/service/openai"

const (
	baseURL      = "https://api.together.xyz/v1"
	apiKeyEnv    = "TOGETHER_API_KEY" //nolint:gosec // G101: env var name, not a credential
	defaultModel = "zai-org/GLM-5.1"
)

// NewLLM builds a Together AI LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("TogetherLLM", baseURL, apiKeyEnv, defaultModel, cfg)
}
