package nvidia

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds an NVIDIA NIM LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:         "NvidiaLLM",
		BaseURL:      baseURL,
		DefaultModel: defaultModel,
	}, cfg)
}
