package qwen

import (
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
)

// LLMService is the Qwen LLM service NewLLM builds.
type LLMService struct {
	*chat.LLMService
}

// NewLLM builds a Qwen (DashScope) LLM service.
func NewLLM(cfg chat.LLMConfig) *LLMService {
	s := &LLMService{}
	s.LLMService = buildLLM(cfg, llm.WithSelf(s))
	return s
}

// buildLLM builds the service NewLLM returns, registering the
// service opts names as the processor the pipeline runs.
func buildLLM(cfg chat.LLMConfig, opts ...llm.Option) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "QwenLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg, opts...)
}
