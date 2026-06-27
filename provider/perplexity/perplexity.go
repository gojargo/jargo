// Package perplexity provides Perplexity's OpenAI-compatible LLM service. Its
// base URL has no "/v1" path segment, unlike most compatible providers.
package perplexity

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.perplexity.ai"
	defaultModel = "sonar"
)

// NewLLM builds a Perplexity LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("PerplexityLLM", baseURL, defaultModel, cfg)
}
