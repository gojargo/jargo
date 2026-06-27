// Package cerebras provides Cerebras's OpenAI-compatible LLM service.
package cerebras

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.cerebras.ai/v1"
	defaultModel = "gpt-oss-120b"
)

// NewLLM builds a Cerebras LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("CerebrasLLM", baseURL, defaultModel, cfg)
}
