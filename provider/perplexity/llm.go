package perplexity

import (
	perplexityadapter "github.com/gojargo/jargo/adapter/perplexity"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// NewLLM builds a service for Perplexity's Sonar chat completions API.
// Perplexity speaks the OpenAI chat-completions API there, with two departures
// from it: it has no developer role, and it is stricter than OpenAI about the
// shape of a conversation (which its adapter settles).
//
// Deprecated: Perplexity has ended support for Sonar chat completions in favor
// of its Agent API. Use NewAgentLLM instead.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "PerplexityLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
		// The endpoint has no max_completion_tokens field.
		NoMaxCompletionTokens: true,
		Adapter:               &perplexityadapter.Adapter{},
	}, cfg)
}
