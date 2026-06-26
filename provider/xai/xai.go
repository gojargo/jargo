// Package xai provides xAI's (Grok) OpenAI-compatible LLM service.
package xai

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://api.x.ai/v1"
	apiKeyEnv    = "XAI_API_KEY" //nolint:gosec // G101: env var name, not a credential
	defaultModel = "grok-4.20-non-reasoning"
)

// NewLLM builds an xAI (Grok) LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("XAILLM", baseURL, apiKeyEnv, defaultModel, cfg)
}
