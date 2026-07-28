package inception

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds an Inception LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("InceptionLLM", baseURL, defaultModel, cfg)
}
