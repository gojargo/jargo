package interactions

import (
	"maps"
	"slices"

	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/service/settings"
)

// LLMSettings is the part of the configuration that can change while the
// pipeline runs, through an LLMUpdateSettingsFrame. They are the Gemini
// service's settings, so a service's settings carry over between the two
// Gemini services. Every request is built from them, so an update reaches the
// next request.
type LLMSettings = gemini.LLMSettings

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

// requestSettings is what one request is built from: the service's config with
// the live settings in place of what they replace, read whole.
type requestSettings struct {
	Config
	frequencyPenalty *float64
	presencePenalty  *float64
}

// requestConfig reads the settings one request is built from.
func (s *Service) requestConfig() requestSettings {
	r := requestSettings{Config: s.cfg}
	s.ReadSettings(func() {
		l := s.live
		r.Model = l.Model.Or("")
		r.MaxTokens = l.MaxTokens.Or(0)
		r.Temperature = l.Temperature.Ptr()
		r.TopP = l.TopP.Ptr()
		r.TopK = l.TopK.Ptr()
		r.Seed = l.Seed.Ptr()
		r.Thinking = l.Thinking.Ptr()
		r.SafetySettings = l.SafetySettings.Or(nil)
		r.Extra = maps.Clone(l.Extra)
		r.frequencyPenalty = l.FrequencyPenalty.Ptr()
		r.presencePenalty = l.PresencePenalty.Ptr()
	})
	return r
}
