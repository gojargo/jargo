package deepseek

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds a DeepSeek LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("DeepSeekLLM", baseURL, defaultModel, cfg)
}
