package ollama

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds an LLM service backed by a local Ollama server. Set cfg.BaseURL
// to reach a remote Ollama; no API key is required by default.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("OllamaLLM", baseURL, defaultModel, cfg)
}
