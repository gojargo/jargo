package soniox

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
// Voice is a stock voice name (for example "Adrian") or the UUID of a cloned
// voice in the project owning the API key. Speed multiplies the speaking rate
// from 0.7 to 1.3, ReduceSilence shortens the pauses between words without
// changing how fast the words themselves are spoken (only the models that
// support silence reduction accept it), and ClientReferenceID labels each
// request in Soniox's usage logs, for attributing usage to a customer or a
// session (ignored for a temporary API key). Those left without a value are
// left to Soniox.
//
// They all travel in the configuration that opens a stream, so a change needs
// no new connection: the turn's stream is flushed, and the rest of the turn
// opens a fresh one with the new values.
type TTSSettings struct {
	settings.TTS

	// Speed multiplies the speaking rate, from 0.7 to 1.3.
	Speed settings.Opt[float64] `settings:"speed"`
	// ReduceSilence shortens the pauses between words.
	ReduceSilence settings.Opt[bool] `settings:"reduce_silence"`
	// ClientReferenceID labels each request in Soniox's usage logs.
	ClientReferenceID settings.Opt[string] `settings:"client_reference_id"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with, once its defaults are applied. What the caller left unset is given no
// value, which leaves it to Soniox.
func newTTSSettings(cfg TTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = settings.Set(cfg.Voice)
	s.Language = settings.Set(sonioxTTSLanguage(cfg.Language))
	s.Speed = settings.Cleared[float64]()
	if cfg.Speed != nil {
		s.Speed = settings.Set(*cfg.Speed)
	}
	s.ReduceSilence = settings.Cleared[bool]()
	if cfg.ReduceSilence != nil {
		s.ReduceSilence = settings.Set(*cfg.ReduceSilence)
	}
	s.ClientReferenceID = settings.Cleared[string]()
	if cfg.ClientReferenceID != "" {
		s.ClientReferenceID = settings.Set(cfg.ClientReferenceID)
	}
	return s
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *ttsSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Soniox does, implementing
// tts.LanguageNamer.
func (s *ttsSynthesizer) ServiceLanguage(l language.Language) string { return sonioxTTSLanguage(l) }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater.
//
// Everything Soniox is told about a stream goes in the configuration that opens
// it, so a change takes effect on the next stream. The turn's stream is
// flushed, and the sentences still to come open a new one with the updated
// values; the ones already sent finish as they were.
func (s *ttsSynthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.refreshSpaceless()
	for _, field := range []string{"voice", keyModel, "language", "speed", "reduce_silence", "client_reference_id"} {
		if !changed.Has(field) {
			continue
		}
		slog.Debug("soniox tts settings changed, opening a new stream for the rest of the turn",
			"fields", changed.String())
		if s.host != nil {
			s.host.RotateTurnContext(ctx)
		}
		break
	}
	return nil
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*ttsSynthesizer)(nil)
	_ tts.SettingsUpdater = (*ttsSynthesizer)(nil)
	_ tts.LanguageNamer   = (*ttsSynthesizer)(nil)
)
