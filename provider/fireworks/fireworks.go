// Package fireworks provides Fireworks AI's OpenAI-compatible LLM service.
package fireworks

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.fireworks.ai/inference/v1"
	defaultModel = "accounts/fireworks/models/firefunction-v2"
)

// NewLLM builds a Fireworks AI LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("FireworksLLM", baseURL, defaultModel, cfg)
}
