package xai

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds an xAI (Grok) LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("XAILLM", baseURL, defaultModel, cfg)
}
