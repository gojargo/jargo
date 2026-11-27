package crusoe

import (
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
)

// LLMService is the Crusoe LLM service NewLLM builds.
type LLMService struct {
	*chat.LLMService
}

// NewLLM builds a Crusoe Cloud LLM service.
//
// Crusoe does not support the developer role, so a developer message is sent
// as a user message.
func NewLLM(cfg chat.LLMConfig) *LLMService {
	s := &LLMService{}
	s.LLMService = buildLLM(cfg, llm.WithSelf(s))
	return s
}

// buildLLM builds the service NewLLM returns, registering the
// service opts names as the processor the pipeline runs.
func buildLLM(cfg chat.LLMConfig, opts ...llm.Option) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "CrusoeLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg, opts...)
}
