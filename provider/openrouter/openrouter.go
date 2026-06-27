// Package openrouter provides OpenRouter's OpenAI-compatible LLM service, a
// gateway to many models behind one API.
package openrouter

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://openrouter.ai/api/v1"
	defaultModel = "openai/gpt-4.1"
)

// NewLLM builds an OpenRouter LLM service. Set cfg.Model to any "vendor/model"
// slug OpenRouter exposes.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("OpenRouterLLM", baseURL, defaultModel, cfg)
}
