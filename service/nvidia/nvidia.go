// Package nvidia provides NVIDIA NIM's OpenAI-compatible LLM service.
package nvidia

import "github.com/gojargo/jargo/service/openai"

const (
	baseURL      = "https://integrate.api.nvidia.com/v1"
	apiKeyEnv    = "NVIDIA_API_KEY" //nolint:gosec // G101: env var name, not a credential
	defaultModel = "meta/llama-3.1-8b-instruct"
)

// NewLLM builds an NVIDIA NIM LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("NvidiaLLM", baseURL, apiKeyEnv, defaultModel, cfg)
}
