package deepseek

import (
	deepseekadapter "github.com/gojargo/jargo/adapter/deepseek"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// ThinkingConfig is DeepSeek's thinking mode configuration.
type ThinkingConfig struct {
	// Type is the thinking mode: "enabled" or "disabled". DeepSeek's V4 models
	// think before answering unless "disabled" turns it off, which for real-time
	// voice cuts the time to the first answer token. It is not restricted to
	// those two so a mode DeepSeek adds later stays usable. Empty sends no
	// thinking field at all, leaving the choice to DeepSeek's own default.
	Type string `json:"type,omitempty"`
}

// LLMConfig configures a DeepSeek LLM service: everything the OpenAI-compatible
// endpoint takes, plus the thinking mode DeepSeek adds to it.
//
// DeepSeek does not support some OpenAI parameters, so Seed,
// MaxCompletionTokens and ServiceTier are not sent. Bound the reply with
// MaxTokens instead.
type LLMConfig struct {
	chat.LLMConfig
	// Thinking is the thinking mode configuration. Nil uses the service
	// default, disabled: DeepSeek's V4 models otherwise run a reasoning pass
	// before every answer, which delays the first spoken token. Set
	// &ThinkingConfig{Type: "enabled"} to turn it on, or &ThinkingConfig{} to
	// leave the choice to DeepSeek's own default.
	Thinking *ThinkingConfig
}

// Validate reports whether the configuration is usable.
func (c LLMConfig) Validate() error { return validate.Struct(c) }

// NewLLM builds a DeepSeek LLM service.
func NewLLM(cfg LLMConfig) *chat.LLMService {
	compat := cfg.LLMConfig
	// DeepSeek doesn't support these OpenAI parameters.
	compat.Seed = nil
	compat.MaxCompletionTokens = nil
	compat.ServiceTier = ""

	thinking := cfg.Thinking
	if thinking == nil {
		thinking = &ThinkingConfig{Type: "disabled"}
	}
	if thinking.Type != "" {
		compat.MergeExtra(map[string]any{"thinking": *thinking})
	}

	return chat.NewCompatLLM(chat.Compat{
		Name:            "DeepSeekLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
		// DeepSeek's chat API has no JSON schema response format.
		NoResponseSchema: true,
		// Supplies the reasoning_content DeepSeek requires on assistant
		// messages in thinking mode.
		Adapter: &deepseekadapter.Adapter{},
	}, compat)
}
