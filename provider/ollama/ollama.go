// Package ollama provides a client for a local Ollama server through its
// OpenAI-compatible LLM endpoint.
package ollama

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "http://localhost:11434/v1"
	apiKeyEnv    = "OLLAMA_API_KEY" //nolint:gosec // G101: env var name, not a credential
	defaultModel = "llama2"
)

// NewLLM builds an LLM service backed by a local Ollama server. Set cfg.BaseURL
// to reach a remote Ollama; no API key is required by default.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("OllamaLLM", baseURL, apiKeyEnv, defaultModel, cfg)
}
