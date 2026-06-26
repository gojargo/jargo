// Package openrouter provides OpenRouter's OpenAI-compatible LLM service, a
// gateway to many models behind one API.
package openrouter

import "github.com/gojargo/jargo/service/openai"

const (
	baseURL      = "https://openrouter.ai/api/v1"
	apiKeyEnv    = "OPENROUTER_API_KEY" //nolint:gosec // G101: env var name, not a credential
	defaultModel = "openai/gpt-4.1"
)

// NewLLM builds an OpenRouter LLM service. Set cfg.Model to any "vendor/model"
// slug OpenRouter exposes.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("OpenRouterLLM", baseURL, apiKeyEnv, defaultModel, cfg)
}
