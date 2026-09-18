package groq

import (
	"maps"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/chat"
)

// LLMConfig configures a Groq LLM service: everything the OpenAI-compatible
// endpoint takes, plus the reasoning control Groq adds to it.
type LLMConfig struct {
	chat.LLMConfig
	// ReasoningEffort is how much the model thinks before it answers. Which
	// values a model accepts varies: "low", "medium" and "high" on the GPT-OSS
	// models and qwen/qwen3.8-27b, "none" and "default" on the Qwen models, and
	// the models that do not reason reject the parameter outright. Empty leaves
	// Groq's own per-model default.
	//
	// "none" is what keeps a Qwen model's reasoning out of the spoken reply:
	// Groq's reasoning format is raw by default, so the model streams its chain
	// of thought inline in <think> tags and the whole of it reaches the TTS.
	ReasoningEffort string `validate:"omitempty,oneof=none default low medium high"`
}

// Validate reports whether the configuration is usable.
func (c LLMConfig) Validate() error { return validate.Struct(c) }

// NewLLM builds a Groq LLM service.
func NewLLM(cfg LLMConfig) *chat.LLMService {
	compat := cfg.LLMConfig
	if cfg.ReasoningEffort != "" {
		// The reasoning control is Groq's own, so it travels as an extra body
		// field rather than as one of the modeled OpenAI parameters.
		extra := maps.Clone(compat.Extra)
		if extra == nil {
			extra = map[string]any{}
		}
		extra["reasoning_effort"] = cfg.ReasoningEffort
		compat.Extra = extra
	}
	return chat.NewCompatLLM(chat.Compat{
		Name:         "GroqLLM",
		BaseURL:      baseURL,
		DefaultModel: defaultLLMModel,
	}, compat)
}
