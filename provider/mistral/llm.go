package mistral

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds a Mistral AI LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("MistralLLM", baseURL, defaultModel, cfg)
}
