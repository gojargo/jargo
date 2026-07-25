package inception

import "github.com/gojargo/jargo/provider/openai"

// NewLLM builds an Inception LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("InceptionLLM", baseURL, defaultModel, cfg)
}
