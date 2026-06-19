// Package fireworks provides Fireworks AI's OpenAI-compatible LLM service.
package fireworks

import "github.com/gojargo/jargo/service/openai"

const (
	baseURL      = "https://api.fireworks.ai/inference/v1"
	apiKeyEnv    = "FIREWORKS_API_KEY" //nolint:gosec // G101: env var name, not a credential
	defaultModel = "accounts/fireworks/models/llama-v3p1-8b-instruct"
)

// NewLLM builds a Fireworks AI LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("FireworksLLM", baseURL, apiKeyEnv, defaultModel, cfg)
}
