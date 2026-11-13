package ollama

import "github.com/gojargo/jargo/provider/openai/chat"

// NewLLM builds an LLM service backed by a local Ollama server. Set cfg.BaseURL
// to reach a remote Ollama. A local server needs no API key, so with none set
// the placeholder "ollama" is sent; set cfg.APIKey for a server, such as one
// behind a proxy, that requires one.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	if cfg.APIKey == "" {
		cfg.APIKey = placeholderAPIKey
	}
	return chat.NewCompatLLM(chat.Compat{
		Name:            "OllamaLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg)
}
