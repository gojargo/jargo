// Package sambanova provides SambaNova Cloud's OpenAI-compatible LLM service.
package sambanova

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.sambanova.ai/v1"
	defaultModel = "Meta-Llama-3.3-70B-Instruct"
)

// NewLLM builds a SambaNova Cloud LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("SambaNovaLLM", baseURL, defaultModel, cfg)
}
