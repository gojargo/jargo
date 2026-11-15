package rime

import (
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// TTSSettings is the part of the Rime synthesis configuration that can change
// while the pipeline runs. The settings carry the names Rime's own API gives
// them, and Rime's "speaker" is accepted for the voice.
//
// The WebSocket service sends every setting as a query parameter of its
// connection, so a change to any of them reconnects. Which of the model
// controls are sent follows the model: the sampling controls and the time
// scale factor go to coda, the latency, bracket and vocabulary controls to the
// mist models.
type TTSSettings struct {
	settings.TTS

	// Segment is the text segmentation mode: "immediate", "bySentence" or
	// "never".
	Segment settings.Opt[string] `settings:"segment"`
	// SpeedAlpha is the speech speed multiplier.
	SpeedAlpha settings.Opt[float64] `settings:"speedAlpha"`
	// ReduceLatency trades quality for latency (mist models only).
	ReduceLatency settings.Opt[bool] `settings:"reduceLatency"`
	// PauseBetweenBrackets adds pauses between bracketed content (mist models
	// only).
	PauseBetweenBrackets settings.Opt[bool] `settings:"pauseBetweenBrackets"`
	// PhonemizeBetweenBrackets phonemizes bracketed content (mist models only).
	PhonemizeBetweenBrackets settings.Opt[bool] `settings:"phonemizeBetweenBrackets"`
	// NoTextNormalization leaves numbers, dates and abbreviations as written
	// (mist models only).
	NoTextNormalization settings.Opt[bool] `settings:"noTextNormalization"`
	// SaveOovs saves out-of-vocabulary words (mist models only).
	SaveOovs settings.Opt[bool] `settings:"saveOovs"`
	// InlineSpeedAlpha is inline speed-control markup.
	InlineSpeedAlpha settings.Opt[string] `settings:"inlineSpeedAlpha"`
	// RepetitionPenalty penalizes token repetition (coda only, 1.0 to 2.0).
	RepetitionPenalty settings.Opt[float64] `settings:"repetition_penalty"`
	// Temperature is the sampling temperature (coda only, 0.0 to 1.0).
	Temperature settings.Opt[float64] `settings:"temperature"`
	// TopP is the cumulative probability threshold (coda only, 0.0 to 1.0).
	TopP settings.Opt[float64] `settings:"top_p"`
	// TimeScaleFactor scales playback speed (coda only). Values above 1.0 slow
	// the audio down; values below 1.0 speed it up.
	TimeScaleFactor settings.Opt[float64] `settings:"timeScaleFactor"`
}

// Aliases accepts Rime's own name for the voice.
func (TTSSettings) Aliases() map[string]string { return map[string]string{fieldSpeaker: "voice"} }

// newRealtimeTTSSettings is the starting state, taken from what the WebSocket
// service was built with once its defaults are applied. What the caller left
// unset is stored as no value, which leaves it out of the connection.
func newRealtimeTTSSettings(cfg RealtimeTTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = optString(cfg.Speaker)
	s.Language = optString(rimeLanguage(cfg.Language))
	s.Segment = optString(cfg.Segment)
	s.SpeedAlpha = optPtr(cfg.SpeedAlpha)
	s.ReduceLatency = optPtr(cfg.ReduceLatency)
	s.PauseBetweenBrackets = optPtr(cfg.PauseBetweenBrackets)
	s.PhonemizeBetweenBrackets = optPtr(cfg.PhonemizeBetweenBrackets)
	s.NoTextNormalization = optPtr(cfg.NoTextNormalization)
	s.SaveOovs = optPtr(cfg.SaveOovs)
	s.InlineSpeedAlpha = settings.Cleared[string]()
	s.RepetitionPenalty = optPtr(cfg.RepetitionPenalty)
	s.Temperature = optPtr(cfg.Temperature)
	s.TopP = optPtr(cfg.TopP)
	s.TimeScaleFactor = optPtr(cfg.TimeScaleFactor)
	return s
}

// optString is the stored form of a string setting: no value when empty.
func optString(v string) settings.Opt[string] {
	if v == "" {
		return settings.Cleared[string]()
	}
	return settings.Set(v)
}

// optPtr is the stored form of an optional setting: no value when nil.
func optPtr[T any](v *T) settings.Opt[T] {
	if v == nil {
		return settings.Cleared[T]()
	}
	return settings.Set(*v)
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *realtimeSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Rime does, implementing
// tts.LanguageNamer.
func (s *realtimeSynthesizer) ServiceLanguage(l language.Language) string {
	return rimeLanguage(l)
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*realtimeSynthesizer)(nil)
	_ tts.SettingsUpdater = (*realtimeSynthesizer)(nil)
	_ tts.LanguageNamer   = (*realtimeSynthesizer)(nil)
	_ settings.Aliaser    = TTSSettings{}
)
