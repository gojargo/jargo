package ollama

import (
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
)

// LLMService is the Ollama LLM service NewLLM builds.
type LLMService struct {
	*chat.LLMService
}

// NewLLM builds an LLM service backed by a local Ollama server. Set cfg.BaseURL
// to reach a remote Ollama. A local server needs no API key, so with none set
// the placeholder "ollama" is sent; set cfg.APIKey for a server, such as one
// behind a proxy, that requires one.
func NewLLM(cfg chat.LLMConfig) *LLMService {
	s := &LLMService{}
	s.LLMService = buildLLM(cfg, llm.WithSelf(s))
	return s
}

// buildLLM builds the service NewLLM returns, registering the
// service opts names as the processor the pipeline runs.
func buildLLM(cfg chat.LLMConfig, opts ...llm.Option) *chat.LLMService {
	if cfg.APIKey == "" {
		cfg.APIKey = placeholderAPIKey
	}
	return chat.NewCompatLLM(chat.Compat{
		Name:            "OllamaLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg, opts...)
}
