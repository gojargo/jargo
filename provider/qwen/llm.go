package qwen

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds a Qwen (DashScope) LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("QwenLLM", baseURL, defaultModel, cfg)
}
