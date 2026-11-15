package crusoe

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a Crusoe Cloud LLM service.
//
// Crusoe does not support the developer role, so a developer message is sent
// as a user message.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "CrusoeLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg)
}
