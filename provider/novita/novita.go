// Package novita provides Novita AI's OpenAI-compatible LLM service.
package novita

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.novita.ai/openai"
	defaultModel = "moonshotai/kimi-k2.5"
)

// NewLLM builds a Novita AI LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("NovitaLLM", baseURL, defaultModel, cfg)
}
