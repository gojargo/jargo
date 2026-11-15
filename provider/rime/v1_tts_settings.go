package rime

import (
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// V1TTSSettings is the part of the Rime WebSocket v1 synthesis configuration
// that can change while the pipeline runs. Each context uses the settings in
// effect when it starts, so a change applies from the next turn. Rime's
// "speaker" is accepted for the voice.
//
// The model is fixed when the service is built, since it selects the endpoint:
// a change to it is reported and not applied. A context leaves out, with a
// warning, any setting its model does not accept.
type V1TTSSettings struct {
	settings.TTS

	// TimeScaleFactor scales playback speed. Values above 1.0 slow the audio
	// down; values below 1.0 speed it up. mistv2 does not take it.
	TimeScaleFactor settings.Opt[float64] `settings:"time_scale_factor"`
	// TextLookaheadTokens is how many input tokens are collected before audio
	// starts (coda only).
	TextLookaheadTokens settings.Opt[int] `settings:"text_lookahead_tokens"`
	// PauseBetweenBrackets adds pauses between bracketed content (mist models
	// only).
	PauseBetweenBrackets settings.Opt[bool] `settings:"pause_between_brackets"`
	// PhonemizeBetweenBrackets phonemizes bracketed content (mist models only).
	PhonemizeBetweenBrackets settings.Opt[bool] `settings:"phonemize_between_brackets"`
	// SaveOovs saves out-of-vocabulary words (mist models only).
	SaveOovs settings.Opt[bool] `settings:"save_oovs"`
}

// Aliases accepts Rime's own name for the voice.
func (V1TTSSettings) Aliases() map[string]string { return map[string]string{fieldSpeaker: "voice"} }

// newV1TTSSettings is the starting state, taken from what the service was built
// with. What the caller left unset is stored as no value, which leaves it out
// of the start request.
func newV1TTSSettings(cfg V1TTSConfig) *V1TTSSettings {
	s := &V1TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = optString(cfg.Speaker)
	s.Language = optString(rimeV1Language(cfg.Language))
	s.TimeScaleFactor = optPtr(cfg.TimeScaleFactor)
	s.TextLookaheadTokens = optPtr(cfg.TextLookaheadTokens)
	s.PauseBetweenBrackets = optPtr(cfg.PauseBetweenBrackets)
	s.PhonemizeBetweenBrackets = optPtr(cfg.PhonemizeBetweenBrackets)
	s.SaveOovs = optPtr(cfg.SaveOovs)
	return s
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *v1Synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language as the BCP-47 tag Rime's WebSocket v1 API
// takes, implementing tts.LanguageNamer.
func (s *v1Synthesizer) ServiceLanguage(l language.Language) string {
	return rimeV1Language(l)
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*v1Synthesizer)(nil)
	_ tts.LanguageNamer  = (*v1Synthesizer)(nil)
	_ settings.Aliaser   = V1TTSSettings{}
)
