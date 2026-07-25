package openrouter

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds an OpenRouter LLM service. Set cfg.Model to any "vendor/model"
// slug OpenRouter exposes.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("OpenRouterLLM", baseURL, defaultModel, cfg)
}
