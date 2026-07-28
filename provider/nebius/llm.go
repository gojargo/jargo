package nebius

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a Nebius AI Studio LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("NebiusLLM", baseURL, defaultModel, cfg)
}
