package speech

import (
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// TTSSettings is the part of the Azure synthesis configuration that can change
// while the pipeline runs. Every sentence is sent as an SSML document built
// from them, so a change applies from the next sentence.
type TTSSettings struct {
	settings.TTS

	// Effect is the audio effect processor that compensates for playback
	// distortion on a target device, as SSML's effect attribute on <voice>:
	// "eq_car" for enclosed vehicles, "eq_telecomhp8k" for narrowband telephony
	// (pair it with an 8 kHz pipeline). Azure ignores a value it does not know.
	Effect settings.Opt[string] `settings:"effect"`
	// Emphasis is the level of emphasis ("strong", "moderate", "reduced").
	Emphasis settings.Opt[string] `settings:"emphasis"`
	// ForceLocale wraps the text in SSML's <lang xml:lang>, so a multilingual
	// voice speaks in the configured language rather than the one it detects in
	// the text. Single-locale voices ignore it. It also stops per-segment
	// language switching, so mixed text is spoken entirely in the configured
	// locale. A multilingual voice synthesizes no audio at all for a locale it
	// does not speak, so pair it with a language the voice supports.
	ForceLocale settings.Opt[bool] `settings:"force_locale"`
	// Pitch adjusts the pitch ("+10%", "-5Hz", "high").
	Pitch settings.Opt[string] `settings:"pitch"`
	// Rate adjusts the speaking rate ("1.0", "1.25", "slow", "fast").
	Rate settings.Opt[string] `settings:"rate"`
	// Role is the role the voice plays ("YoungAdultFemale").
	Role settings.Opt[string] `settings:"role"`
	// Style is the speaking style ("cheerful", "sad", "excited").
	Style settings.Opt[string] `settings:"style"`
	// StyleDegree is the intensity of the style, from 0.01 to 2.0.
	StyleDegree settings.Opt[string] `settings:"style_degree"`
	// VoiceParameters are the model parameters of an HD voice, as SSML's
	// semicolon-separated parameters attribute on <voice>
	// ("temperature=0.7;top_p=0.8;top_k=22;cfg_scale=1.4", or
	// "enhancePronunciation=true"). Which keys apply depends on the voice's base
	// model; standard neural voices ignore it.
	VoiceParameters settings.Opt[string] `settings:"voice_parameters"`
	// Volume sets the volume ("+20%", "loud", "x-soft").
	Volume settings.Opt[string] `settings:"volume"`
}

// newTTSSettings is the starting state, taken from what the streaming service
// was built with.
func newTTSSettings(cfg RealtimeTTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Voice = settings.Set(cfg.Voice)
	// The language is stored as the caller gave it; the service names it the
	// way Azure does.
	lang := language.Language(defaultTTSLanguage)
	if cfg.Language != "" {
		lang = cfg.Language
	}
	s.Language = settings.Set(string(lang))
	s.ForceLocale = settings.Set(cfg.ForceLocale)
	for _, f := range []struct {
		opt   *settings.Opt[string]
		value string
	}{
		{&s.Effect, cfg.Effect},
		{&s.Emphasis, cfg.Emphasis},
		{&s.Pitch, cfg.Pitch},
		{&s.Rate, cfg.Rate},
		{&s.Role, cfg.Role},
		{&s.Style, cfg.Style},
		{&s.StyleDegree, cfg.StyleDegree},
		{&s.VoiceParameters, cfg.VoiceParameters},
		{&s.Volume, cfg.Volume},
	} {
		if f.value != "" {
			*f.opt = settings.Set(f.value)
		}
	}
	return s
}

// ssmlParams is what the next sentence's SSML is built from.
func (s *TTSSettings) ssmlParams() ssmlParams {
	return ssmlParams{
		Voice:           s.Voice.Or(""),
		Language:        s.Language.Or(""),
		Effect:          s.Effect.Or(""),
		VoiceParameters: s.VoiceParameters.Or(""),
		ForceLocale:     s.ForceLocale.Or(false),
		Style:           s.Style.Or(""),
		StyleDegree:     s.StyleDegree.Or(""),
		Role:            s.Role.Or(""),
		Rate:            s.Rate.Or(""),
		Pitch:           s.Pitch.Or(""),
		Volume:          s.Volume.Or(""),
		Emphasis:        s.Emphasis.Or(""),
	}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *realtimeSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Azure does, implementing
// tts.LanguageNamer.
func (s *realtimeSynthesizer) ServiceLanguage(l language.Language) string {
	return azureLanguage(l)
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*realtimeSynthesizer)(nil)
	_ tts.LanguageNamer  = (*realtimeSynthesizer)(nil)
)
