package fireworks

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a Fireworks AI LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("FireworksLLM", baseURL, defaultModel, cfg)
}
