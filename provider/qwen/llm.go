package qwen

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a Qwen (DashScope) LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "QwenLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg)
}
