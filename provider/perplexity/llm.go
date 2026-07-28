package perplexity

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a Perplexity LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("PerplexityLLM", baseURL, defaultModel, cfg)
}
