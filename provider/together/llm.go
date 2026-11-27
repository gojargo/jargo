package together

import (
	"maps"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
)

// LLMConfig configures a Together AI LLM service: everything the
// OpenAI-compatible endpoint takes, plus the reasoning controls Together adds to
// it.
type LLMConfig struct {
	chat.LLMConfig
	// Reasoning is Together's reasoning toggle, for the models that support one,
	// such as {"enabled": false}. A reasoning model such as GLM runs a reasoning
	// pass before every answer, which delays the first spoken token. Nil leaves
	// the choice to the model's own default.
	Reasoning map[string]any
	// ReasoningEffort is how much the model reasons before answering. Values
	// vary by model: "low", "medium" and "high" are common, zai-org/GLM-5.3 also
	// takes "max", and some models map a value they do not support onto one they
	// do. Other strings are passed through as given. GLM-5.3 ignores Reasoning,
	// but "low" still shortens its reasoning pass. Empty leaves Together's
	// per-model default.
	ReasoningEffort string
}

// Validate reports whether the configuration is usable.
func (c LLMConfig) Validate() error { return validate.Struct(c) }

// LLMService is the Together AI LLM service NewLLM builds.
type LLMService struct {
	*chat.LLMService
}

// NewLLM builds a Together AI LLM service.
func NewLLM(cfg LLMConfig) *LLMService {
	s := &LLMService{}
	s.LLMService = buildLLM(cfg, llm.WithSelf(s))
	return s
}

// buildLLM builds the service NewLLM returns, registering the
// service opts names as the processor the pipeline runs.
func buildLLM(cfg LLMConfig, opts ...llm.Option) *chat.LLMService {
	compat := cfg.LLMConfig
	// The reasoning controls are Together's own, so they travel as extra body
	// fields. The caller's own Extra wins key by key, as it does over every
	// other request parameter.
	fields := map[string]any{}
	if cfg.Reasoning != nil {
		fields["reasoning"] = cfg.Reasoning
	}
	if cfg.ReasoningEffort != "" {
		fields["reasoning_effort"] = cfg.ReasoningEffort
	}
	if len(fields) > 0 {
		maps.Copy(fields, compat.Extra)
		compat.Extra = fields
	}
	return chat.NewCompatLLM(chat.Compat{
		Name:            "TogetherLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, compat, opts...)
}
