package perplexity

import (
	perplexityadapter "github.com/gojargo/jargo/adapter/perplexity"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// NewLLM builds a Perplexity LLM service. Perplexity speaks the OpenAI
// chat-completions API, with two departures from it: it has no developer role,
// and it is stricter than OpenAI about the shape of a conversation (which its
// adapter settles).
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "PerplexityLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
		Adapter:         &perplexityadapter.Adapter{},
	}, cfg)
}
