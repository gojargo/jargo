// Package mistral provides Mistral AI's OpenAI-compatible LLM service.
package mistral

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.mistral.ai/v1"
	defaultModel = "mistral-small-latest"
)

// NewLLM builds a Mistral AI LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("MistralLLM", baseURL, defaultModel, cfg)
}
