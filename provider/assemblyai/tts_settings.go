package assemblyai

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
// Voice is the API's voice_id: any voice by its ID, an AssemblyAI voice by its
// library slug, such as "aai-jane", or a cloned voice by the slug the AssemblyAI
// dashboard shows. It defaults to Audrey; cleared, it leaves the voice to the
// server. Language is the language the session speaks, and Model names the
// model, such as "assemblyai-tts-1". Unset, the session speaks the voice's own
// language and runs on that language's default model.
//
// The session is configured when it connects, so a change to any of them
// reconnects.
type TTSSettings struct {
	settings.TTS
}

// newTTSSettings is the starting state, taken from what the service was built
// with. What the caller left unset is given no value, which is what leaves it
// to the server.
func newTTSSettings(cfg TTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Voice = settings.Set(cfg.VoiceID)
	s.Model = settings.Cleared[string]()
	if cfg.Model != "" {
		s.Model = settings.Set(cfg.Model)
	}
	s.Language = settings.Cleared[string]()
	if lang := assemblyaiTTSLanguage(cfg.Language); lang != "" {
		s.Language = settings.Set(lang)
	}
	return s
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way the Streaming TTS API does,
// implementing tts.LanguageNamer.
func (s *synthesizer) ServiceLanguage(l language.Language) string {
	return assemblyaiTTSLanguage(l)
}

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. The voice,
// the language and the model configure the session, so a change to any of them
// reconnects.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if changed.Has("voice") || changed.Has("language") || changed.Has("model") {
		s.disconnect(ctx)
		s.connect(ctx)
		return nil
	}
	if len(changed) > 0 {
		slog.Warn("assemblyai tts settings changed that the service does not act on",
			"fields", changed.String())
	}
	return nil
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*synthesizer)(nil)
	_ tts.SettingsUpdater = (*synthesizer)(nil)
	_ tts.LanguageNamer   = (*synthesizer)(nil)
)
