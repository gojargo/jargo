package resembleai

import (
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// TTSSettings is the part of the synthesis configuration that can change while
// the pipeline runs.
//
// Voice is the voice UUID, and ApplyCustomPronunciations whether Resemble AI
// applies the team's custom pronunciations to matching words in the text.
// Another way to guide pronunciation is a text transform that finds and
// replaces words before the text reaches any provider, which works the same
// across TTS services.
//
// Every request carries them, so a change takes effect from the next sentence
// on, on the connection already open.
type TTSSettings struct {
	settings.TTS

	// ApplyCustomPronunciations has Resemble AI apply the team's custom
	// pronunciations.
	ApplyCustomPronunciations settings.Opt[bool] `settings:"apply_custom_pronunciations"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with. Resemble AI has no model or language to choose, so neither is given a
// value.
func newTTSSettings(cfg Config) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Cleared[string]()
	s.Language = settings.Cleared[string]()
	s.Voice = settings.Set(cfg.Voice)
	s.ApplyCustomPronunciations = settings.Set(cfg.ApplyCustomPronunciations)
	return s
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder. Nothing has to be done on a change: every
// request is built from the store as it is when the request goes out.
func (s *synthesizer) Settings() any { return s.live }

var _ tts.SettingsHolder = (*synthesizer)(nil)
