package grok

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds an xAI (Grok) LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("XAILLM", baseURL, defaultModel, cfg)
}
