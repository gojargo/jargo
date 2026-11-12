package neuphonic

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
// Voice is the voice id, and Language the language spoken, as Neuphonic names
// it. Speed is the speech-speed multiplier, Temperature the randomness
// introduced into the synthesis, from 0 to 1, and IPAKey the regional variant
// the sounds of <phoneme:...> tags are read with. Those left without a value
// are left to Neuphonic.
//
// They all configure the connection, which carries them in its URL, so a change
// to any of them reconnects.
type TTSSettings struct {
	settings.TTS

	// Speed is the speech-speed multiplier.
	Speed settings.Opt[float64] `settings:"speed"`
	// Temperature is the randomness introduced into the synthesis, from 0 to 1.
	Temperature settings.Opt[float64] `settings:"temperature"`
	// IPAKey is the regional variant phoneme tags are read with, either
	// "es-castilian" or "pt-br".
	IPAKey settings.Opt[string] `settings:"ipa_key"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with, once its defaults are applied. What the caller left unset is given no
// value, which leaves it to Neuphonic.
func newTTSSettings(cfg Config) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Cleared[string]()
	s.Voice = settings.Cleared[string]()
	if cfg.VoiceID != "" {
		s.Voice = settings.Set(cfg.VoiceID)
	}
	s.Language = settings.Set(neuphonicLanguage(cfg.Language))
	s.Speed = settings.Set(cfg.Speed)
	s.Temperature = settings.Cleared[float64]()
	if cfg.Temperature != nil {
		s.Temperature = settings.Set(*cfg.Temperature)
	}
	s.IPAKey = settings.Cleared[string]()
	if cfg.IPAKey != "" {
		s.IPAKey = settings.Set(cfg.IPAKey)
	}
	return s
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Neuphonic does, implementing
// tts.LanguageNamer.
func (s *synthesizer) ServiceLanguage(l language.Language) string { return neuphonicLanguage(l) }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. Every
// setting travels in the URL the connection is opened on, so any change
// reconnects.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.refreshEndpoint()
	s.Disconnect(ctx)
	s.Connect(ctx)
	slog.Info("neuphonic tts: switching to new settings", "fields", changed.String())
	return nil
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*synthesizer)(nil)
	_ tts.SettingsUpdater = (*synthesizer)(nil)
	_ tts.LanguageNamer   = (*synthesizer)(nil)
)
