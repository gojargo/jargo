package grok

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
)

// LLMService is the Grok LLM service NewLLM builds.
type LLMService struct {
	*chat.LLMService
}

// NewLLM builds an xAI (Grok) LLM service.
func NewLLM(cfg chat.LLMConfig) *LLMService {
	s := &LLMService{}
	s.LLMService = buildLLM(cfg, llm.WithSelf(s))
	return s
}

// buildLLM builds the service NewLLM returns, registering the
// service opts names as the processor the pipeline runs.
func buildLLM(cfg chat.LLMConfig, opts ...llm.Option) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:         "XAILLM",
		BaseURL:      baseURL,
		DefaultModel: defaultModel,
		TokenUsage:   tokenUsage,
	}, cfg, opts...)
}

// tokenUsage converts a Grok completion's usage. xAI reports reasoning tokens
// apart from the completion tokens, with the total covering both, so they are
// added to the completion count. A total that already matches the prompt and
// completion counts means reasoning was included, and the counts are kept as
// reported.
func tokenUsage(u frames.LLMTokenUsage) frames.LLMTokenUsage {
	reasoning, _ := frames.TokenCount(u.ReasoningTokens)
	if reasoning != 0 && u.TotalTokens == u.PromptTokens+u.CompletionTokens+reasoning {
		u.CompletionTokens += reasoning
	}
	return u
}
