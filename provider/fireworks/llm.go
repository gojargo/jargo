package fireworks

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds a Fireworks AI LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("FireworksLLM", baseURL, defaultModel, cfg)
}
