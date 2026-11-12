package grok

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// NewLLM builds an xAI (Grok) LLM service.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:         "XAILLM",
		BaseURL:      baseURL,
		DefaultModel: defaultModel,
		TokenUsage:   tokenUsage,
	}, cfg)
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
