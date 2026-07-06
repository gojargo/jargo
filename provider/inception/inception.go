// Package inception provides Inception's OpenAI-compatible LLM service.
package inception

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.inceptionlabs.ai/v1"
	defaultModel = "mercury-2"
)

// NewLLM builds an Inception LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("InceptionLLM", baseURL, defaultModel, cfg)
}
