package cerebras

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a Cerebras LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("CerebrasLLM", baseURL, defaultModel, cfg)
}
