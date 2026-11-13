package chat

import (
	"maps"

	"github.com/gojargo/jargo/service/settings"
)

// LLMSettings is the part of the chat-completions configuration that can
// change while the pipeline runs, through an LLMUpdateSettingsFrame: the
// general LLM settings (model, sampling, the token bound, a system
// instruction) and the completion bound OpenAI's reasoning models read. Every
// request is built from them, so an update reaches the next request.
//
// A setting the service does not model travels in Extra and is merged into the
// request body, as LLMConfig.Extra is.
type LLMSettings struct {
	settings.LLM
	// MaxCompletionTokens caps the completion length on models that require it
	// in place of max_tokens.
	MaxCompletionTokens settings.Opt[int] `settings:"max_completion_tokens"`
}

// newLLMSettings is the starting state, taken from what the service was built
// with. A sampling control left nil is left unset, so it is omitted from the
// request.
func newLLMSettings(cfg LLMConfig) *LLMSettings {
	s := &LLMSettings{}
	s.Model = settings.Set(cfg.Model)
	if cfg.MaxTokens > 0 {
		s.MaxTokens = settings.Set(cfg.MaxTokens)
	}
	setPtr(&s.MaxCompletionTokens, cfg.MaxCompletionTokens)
	setPtr(&s.Temperature, cfg.Temperature)
	setPtr(&s.TopP, cfg.TopP)
	setPtr(&s.FrequencyPenalty, cfg.FrequencyPenalty)
	setPtr(&s.PresencePenalty, cfg.PresencePenalty)
	setPtr(&s.Seed, cfg.Seed)
	if len(cfg.Extra) > 0 {
		s.Extra = maps.Clone(cfg.Extra)
	}
	return s
}

// setPtr gives o the value p points to, leaving it unset for a nil p.
func setPtr[T any](o *settings.Opt[T], p *T) {
	if p != nil {
		*o = settings.Set(*p)
	}
}

// SettingsStore is the runtime settings of an endpoint with settings of its
// own, given to Compat.Settings. It is satisfied by a pointer to LLMSettings or
// to a struct embedding it, and by nothing else.
type SettingsStore interface {
	chatSettings() *LLMSettings
}

// chatSettings returns the shared settings, for a struct that embeds them to
// satisfy SettingsStore.
func (s *LLMSettings) chatSettings() *LLMSettings { return s }

// Settings returns the service's runtime-updatable settings, implementing
// llm.SettingsHolder: the endpoint's own when it has some, which embed the
// shared ones.
func (s *LLMService) Settings() any { return s.store }

// liveSettings is a copy of the settings as they stand, read whole, so the
// request being built is not changed under it by an update.
func (s *LLMService) liveSettings() LLMSettings {
	var cur LLMSettings
	s.ReadSettings(func() {
		cur = *s.live
		cur.Extra = maps.Clone(s.live.Extra)
	})
	return cur
}
