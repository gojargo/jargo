package gemini

import (
	"maps"
	"slices"

	"github.com/gojargo/jargo/service/settings"
)

// LLMSettings is the part of the Gemini configuration that can change while
// the pipeline runs, through an LLMUpdateSettingsFrame: the general LLM
// settings (the model, the token bound, the sampling controls, a system
// instruction), thinking and the safety filters. Every request is built from
// them, so an update reaches the next request.
//
// A setting the service does not model travels in Extra and is merged into the
// generation config, as Config.Extra is.
type LLMSettings struct {
	settings.LLM
	// Thinking is the thinking configuration. Cleared, the request configures
	// none, and the model's thinking default applies.
	Thinking settings.Opt[ThinkingConfig] `settings:"thinking"`
	// SafetySettings are the content-safety filters, one per harm category.
	SafetySettings settings.Opt[[]SafetySetting] `settings:"safety_settings"`
}

// newLLMSettings is the starting state, taken from what the service was built
// with.
func newLLMSettings(cfg Config) *LLMSettings {
	s := &LLMSettings{}
	s.Model = settings.Set(cfg.Model)
	s.MaxTokens = settings.Set(cfg.MaxTokens)
	if cfg.Temperature != nil {
		s.Temperature = settings.Set(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		s.TopP = settings.Set(*cfg.TopP)
	}
	if cfg.TopK != nil {
		s.TopK = settings.Set(*cfg.TopK)
	}
	if cfg.Seed != nil {
		s.Seed = settings.Set(*cfg.Seed)
	}
	if cfg.Thinking != nil {
		s.Thinking = settings.Set(*cfg.Thinking)
	}
	if len(cfg.SafetySettings) > 0 {
		s.SafetySettings = settings.Set(slices.Clone(cfg.SafetySettings))
	}
	if len(cfg.Extra) > 0 {
		s.Extra = maps.Clone(cfg.Extra)
	}
	return s
}

// Settings returns the service's runtime-updatable settings, implementing
// llm.SettingsHolder.
func (s *Service) Settings() any { return s.live }

// requestConfig is the service's config with the live settings in place of
// what they replace, read whole: it is what one request is built from.
func (s *Service) requestConfig() Config {
	cfg := s.cfg
	s.ReadSettings(func() {
		l := s.live
		cfg.Model = l.Model.Or("")
		cfg.MaxTokens = l.MaxTokens.Or(0)
		cfg.Temperature = l.Temperature.Ptr()
		cfg.TopP = l.TopP.Ptr()
		cfg.TopK = l.TopK.Ptr()
		cfg.Seed = l.Seed.Ptr()
		cfg.Thinking = l.Thinking.Ptr()
		cfg.SafetySettings = l.SafetySettings.Or(nil)
		cfg.Extra = maps.Clone(l.Extra)
	})
	return cfg
}
