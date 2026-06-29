// Package nebius provides Nebius AI Studio's OpenAI-compatible LLM service.
package nebius

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.studio.nebius.com/v1"
	defaultModel = "meta-llama/Llama-3.3-70B-Instruct"
)

// NewLLM builds a Nebius AI Studio LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("NebiusLLM", baseURL, defaultModel, cfg)
}
