package deepseek

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a DeepSeek LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "DeepSeekLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg)
}
