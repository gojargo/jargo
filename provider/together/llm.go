package together

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds a Together AI LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("TogetherLLM", baseURL, defaultModel, cfg)
}
