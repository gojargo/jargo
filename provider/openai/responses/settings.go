package responses

import (
	"maps"

	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// LLMSettings is the part of the Responses configuration that can change while
// the pipeline runs, through an LLMUpdateSettingsFrame: the general LLM
// settings (the model, the sampling controls, a system instruction), the bound
// on the output and the reasoning. Every request is built from them, so an
// update reaches the next request.
//
// A setting the service does not model travels in Extra and is merged into the
// request, as Config.Extra is.
type LLMSettings struct {
	settings.LLM
	// MaxCompletionTokens bounds the output, sent as max_output_tokens.
	MaxCompletionTokens settings.Opt[int] `settings:"max_completion_tokens"`
	// Reasoning is how much a reasoning-capable model thinks. Cleared, the
	// request configures none, and the model's thinking default applies.
	Reasoning settings.Opt[ReasoningConfig] `settings:"reasoning"`
}

// newLLMSettings is the starting state, taken from what the service was built
// with.
func newLLMSettings(cfg Config) *LLMSettings {
	s := &LLMSettings{}
	s.Model = settings.Set(cfg.Model)
	if cfg.Temperature != nil {
		s.Temperature = settings.Set(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		s.TopP = settings.Set(*cfg.TopP)
	}
	if cfg.MaxOutputTokens > 0 {
		s.MaxCompletionTokens = settings.Set(cfg.MaxOutputTokens)
	}
	if cfg.Reasoning != nil {
		s.Reasoning = settings.Set(*cfg.Reasoning)
	}
	if len(cfg.Extra) > 0 {
		s.Extra = maps.Clone(cfg.Extra)
	}
	return s
}

// requestConfig is cfg with the live settings in place of what they replace,
// read whole: it is what one request is built and encoded from.
func requestConfig(b *llm.Base, cfg Config, live *LLMSettings) Config {
	b.ReadSettings(func() {
		cfg.Model = live.Model.Or("")
		cfg.Temperature = live.Temperature.Ptr()
		cfg.TopP = live.TopP.Ptr()
		cfg.MaxOutputTokens = live.MaxCompletionTokens.Or(0)
		cfg.Reasoning = live.Reasoning.Ptr()
		cfg.Extra = maps.Clone(live.Extra)
	})
	return cfg
}

// Settings returns the service's runtime-updatable settings, implementing
// llm.SettingsHolder.
func (s *HTTPService) Settings() any { return s.live }

// Settings returns the service's runtime-updatable settings, implementing
// llm.SettingsHolder.
func (s *Service) Settings() any { return s.live }
