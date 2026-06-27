// Package deepseek provides DeepSeek's OpenAI-compatible LLM service.
package deepseek

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.deepseek.com/v1"
	defaultModel = "deepseek-chat"
)

// NewLLM builds a DeepSeek LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("DeepSeekLLM", baseURL, defaultModel, cfg)
}
