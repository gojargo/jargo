package together

import (
	"context"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// TTSSettings is the part of the synthesis configuration that can change while
// the pipeline runs.
//
// The model, the voice, the language and the partial length are read from the
// URL a connection is opened on, and the options are sent once it is open, so
// any change reconnects.
type TTSSettings struct {
	settings.TTS

	// MaxPartialLength caps the partial text length for streaming.
	MaxPartialLength settings.Opt[int] `settings:"max_partial_length"`
	// ExtraParams are model-specific synthesis options sent with the session
	// setup, such as "pronunciation_dict".
	ExtraParams settings.Opt[map[string]any] `settings:"extra_params"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with, once its defaults are applied.
func newTTSSettings(cfg TTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = settings.Set(cfg.Voice)
	// The language is stored as the caller gave it; the service names it the
	// way Together does.
	s.Language = settings.Set(string(cfg.Language))
	s.MaxPartialLength = settings.Cleared[int]()
	if cfg.MaxPartialLength != nil {
		s.MaxPartialLength = settings.Set(*cfg.MaxPartialLength)
	}
	s.ExtraParams = settings.Cleared[map[string]any]()
	if cfg.ExtraParams != nil {
		s.ExtraParams = settings.Set(cfg.ExtraParams)
	}
	return s
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Together does, implementing
// tts.LanguageNamer.
func (s *synthesizer) ServiceLanguage(l language.Language) string { return togetherLanguage(l) }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. Every
// setting is fixed when a connection is opened, so any change reconnects.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.refreshSession()
	s.Disconnect(ctx)
	s.Connect(ctx)
	return nil
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*synthesizer)(nil)
	_ tts.SettingsUpdater = (*synthesizer)(nil)
	_ tts.LanguageNamer   = (*synthesizer)(nil)
)
