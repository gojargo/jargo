package smallest

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
// Every request carries the settings it is spoken with, so a change takes
// effect on the next request, without reconnecting. Those left without a value
// are left to Smallest.
type TTSSettings struct {
	settings.TTS

	// Speed multiplies the speaking rate (0.5 to 2.0).
	Speed settings.Opt[float64] `settings:"speed"`
	// MathNotation reads a math operator between digits as a word.
	MathNotation settings.Opt[bool] `settings:"math_notation"`
	// ContentFilter is the action taken on text that matches Smallest's
	// profanity word list: "reject", "flag" or "redact".
	ContentFilter settings.Opt[string] `settings:"content_filter"`
	// DateFormat is the order all-numeric dates are read in: "EU" or "US".
	DateFormat settings.Opt[string] `settings:"date_format"`
	// NumberPronunciationLanguage is the language code numerals, currency,
	// times and dates are read in.
	NumberPronunciationLanguage settings.Opt[string] `settings:"number_pronunciation_language"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with, once its defaults are applied.
func newTTSSettings(cfg Config) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = settings.Set(cfg.Voice)
	s.Language = settings.Set(cfg.Language)
	s.Speed = settings.Cleared[float64]()
	if cfg.Speed != nil {
		s.Speed = settings.Set(*cfg.Speed)
	}
	s.MathNotation = settings.Cleared[bool]()
	if cfg.MathNotation != nil {
		s.MathNotation = settings.Set(*cfg.MathNotation)
	}
	s.ContentFilter = optString(cfg.ContentFilter)
	s.DateFormat = optString(cfg.DateFormat)
	s.NumberPronunciationLanguage = optString(cfg.NumberPronunciationLanguage)
	return s
}

// optString is a setting holding v, or no value when v is empty.
func optString(v string) settings.Opt[string] {
	if v == "" {
		return settings.Cleared[string]()
	}
	return settings.Set(v)
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Smallest does, implementing
// tts.LanguageNamer.
func (s *synthesizer) ServiceLanguage(l language.Language) string { return smallestLanguage(l) }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. Every
// request carries the settings, so the next one picks the change up; only the
// keepalive, which goes out on a goroutine of its own, has to be rebuilt.
func (s *synthesizer) UpdateSettings(_ context.Context, changed settings.Changed) error {
	if len(changed) > 0 {
		s.refreshKeepalive()
	}
	return nil
}

// smallestLanguage maps a language to the code Smallest takes: its base code.
// A language Smallest is not known to speak still gets its base code, with a
// warning.
func smallestLanguage(l language.Language) string {
	base := l.BaseCode()
	switch base {
	case "ar", "bn", "de", "el", "en", "es", "fi", "fr", "gu", "hi", "id", "it", "ja", "kn", "ko", "ml",
		"mr", "ms", "nl", "no", "or", "pa", "pl", "pt", "ru", "sv", "ta", "te", "tr", "vi", "zh":
	default:
		slog.Warn("smallest tts: language not verified, using its base code", "language", string(l), "code", base)
	}
	return base
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*synthesizer)(nil)
	_ tts.SettingsUpdater = (*synthesizer)(nil)
	_ tts.LanguageNamer   = (*synthesizer)(nil)
)
