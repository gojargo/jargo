// Package xai provides xAI's (Grok) OpenAI-compatible LLM service.
package xai

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.x.ai/v1"
	defaultModel = "grok-4.20-non-reasoning"
)

// NewLLM builds an xAI (Grok) LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("XAILLM", baseURL, defaultModel, cfg)
}
