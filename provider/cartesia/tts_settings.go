package cartesia

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// TTSSettings is the part of the synthesis configuration that can change while
// the pipeline runs.
//
// Cartesia fixes the voice, the model and the language for the length of a
// context, so a change to any of them takes effect on the next one: the turn
// being spoken is closed out and the rest of it is said with the new settings.
// The generation controls and the pronunciation dictionary go out on every
// request, so a change to those applies to the next sentence.
type TTSSettings struct {
	settings.TTS

	// GenerationConfig guides generation (volume, speed, emotion) on supported
	// models.
	GenerationConfig settings.Opt[*GenerationConfig] `settings:"generation_config"`
	// PronunciationDictID applies a custom pronunciation dictionary.
	PronunciationDictID settings.Opt[string] `settings:"pronunciation_dict_id"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with.
func newTTSSettings(cfg Config) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = settings.Set(cfg.VoiceID)
	if lang := cartesiaLanguage(cfg.Language); lang != "" {
		s.Language = settings.Set(lang)
	}
	if cfg.GenerationConfig != nil {
		s.GenerationConfig = settings.Set(cfg.GenerationConfig)
	}
	if cfg.PronunciationDictID != "" {
		s.PronunciationDictID = settings.Set(cfg.PronunciationDictID)
	}
	return s
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Cartesia does, implementing
// tts.LanguageNamer. Without it a neutral name and the stored code would be in
// different vocabularies, and the same language would read as a change.
func (s *synthesizer) ServiceLanguage(l language.Language) string {
	return cartesiaLanguage(l)
}

// UpdateSettings acts on a change, implementing tts.SettingsUpdater.
//
// The voice, the model and the language are fixed for the length of a context,
// so a change to any of them closes the turn's context out and opens a new one:
// the sentences still to come are spoken with the new settings, and the ones
// already sent finish in the old voice rather than being cut off. Everything
// else goes out on the next request as it stands.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if !changed.Has("voice") && !changed.Has("model") && !changed.Has("language") {
		return nil
	}
	if s.host == nil {
		return nil
	}
	slog.Debug("cartesia tts settings changed, opening a new context for the rest of the turn",
		"fields", changed.String())
	s.host.RotateTurnContext(ctx)
	return nil
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*synthesizer)(nil)
	_ tts.SettingsUpdater = (*synthesizer)(nil)
	_ tts.LanguageNamer   = (*synthesizer)(nil)
)
