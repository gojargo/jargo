package nebius

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds a Nebius AI Studio LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("NebiusLLM", baseURL, defaultModel, cfg)
}
