package ollama

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds an LLM service backed by a local Ollama server. Set cfg.BaseURL
// to reach a remote Ollama; no API key is required by default.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("OllamaLLM", baseURL, defaultModel, cfg)
}
