package sambanova

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds a SambaNova Cloud LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "SambaNovaLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg)
}
