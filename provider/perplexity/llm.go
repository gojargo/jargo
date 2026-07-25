package perplexity

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds a Perplexity LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("PerplexityLLM", baseURL, defaultModel, cfg)
}
