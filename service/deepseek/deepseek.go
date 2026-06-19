// Package deepseek provides DeepSeek's OpenAI-compatible LLM service.
package deepseek

import "github.com/gojargo/jargo/service/openai"

const (
	baseURL      = "https://api.deepseek.com/v1"
	apiKeyEnv    = "DEEPSEEK_API_KEY" //nolint:gosec // G101: env var name, not a credential
	defaultModel = "deepseek-chat"
)

// NewLLM builds a DeepSeek LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("DeepSeekLLM", baseURL, apiKeyEnv, defaultModel, cfg)
}
