package nvidia

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds an NVIDIA NIM LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("NvidiaLLM", baseURL, defaultModel, cfg)
}
