package qwen

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a Qwen (DashScope) LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("QwenLLM", baseURL, defaultModel, cfg)
}
