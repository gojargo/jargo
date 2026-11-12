package grok

import (
	"context"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// WebsocketTTSSettings is the part of the streaming synthesis configuration
// that can change while the pipeline runs.
//
// xAI reads every one of them from the URL a connection is opened on, so any
// change reconnects. Which text path the service takes is fixed when it is
// built, by TTSConfig.WordTimestamps: turning the timings off here leaves a
// service built with them pushing no text.
type WebsocketTTSSettings struct {
	settings.TTS

	// Speed multiplies the speaking rate, from 0.7 to 1.5.
	Speed settings.Opt[float64] `settings:"speed"`
	// OptimizeStreamingLatency is the latency optimization level: 0, 1 or 2.
	OptimizeStreamingLatency settings.Opt[int] `settings:"optimize_streaming_latency"`
	// TextNormalization expands numbers, dates and abbreviations before
	// synthesis.
	TextNormalization settings.Opt[bool] `settings:"text_normalization"`
	// WithTimestamps asks for the character timings the words are timed from.
	WithTimestamps settings.Opt[bool] `settings:"with_timestamps"`
}

// newWebsocketTTSSettings is the starting state, taken from what the service
// was built with, once its defaults are applied.
func newWebsocketTTSSettings(cfg TTSConfig) *WebsocketTTSSettings {
	s := &WebsocketTTSSettings{}
	s.Model = settings.Cleared[string]()
	s.Voice = settings.Set(cfg.Voice)
	s.Language = settings.Set(xaiLanguage(cfg.Language))
	s.Speed = optional(cfg.Speed)
	s.OptimizeStreamingLatency = optional(cfg.OptimizeStreamingLatency)
	s.TextNormalization = optional(cfg.TextNormalization)
	s.WithTimestamps = settings.Set(cfg.wordTimestamps())
	return s
}

// optional is a setting holding *v, or no value when v is nil.
func optional[T any](v *T) settings.Opt[T] {
	if v == nil {
		return settings.Cleared[T]()
	}
	return settings.Set(*v)
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *ttsSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way xAI does, implementing
// tts.LanguageNamer.
func (s *ttsSynthesizer) ServiceLanguage(l language.Language) string { return xaiLanguage(l) }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. Every
// setting is fixed when a connection is opened, so any change reconnects.
func (s *ttsSynthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.refreshEndpoint()
	s.Disconnect(ctx)
	s.Connect(ctx)
	return nil
}

// Compile-time checks that the streaming synthesizer is driven the way the
// base drives a websocket service.
var (
	_ tts.Synthesizer             = (*ttsSynthesizer)(nil)
	_ tts.ContextSynthesizer      = (*ttsSynthesizer)(nil)
	_ tts.Describer               = (*ttsSynthesizer)(nil)
	_ tts.WebsocketService        = (*ttsSynthesizer)(nil)
	_ tts.Starter                 = (*ttsSynthesizer)(nil)
	_ tts.AudioFlusher            = (*ttsSynthesizer)(nil)
	_ tts.AudioContextInterrupter = (*ttsSynthesizer)(nil)
	_ tts.TrailingSpaceRequirer   = (*ttsSynthesizer)(nil)
	_ tts.SettingsHolder          = (*ttsSynthesizer)(nil)
	_ tts.SettingsUpdater         = (*ttsSynthesizer)(nil)
	_ tts.LanguageNamer           = (*ttsSynthesizer)(nil)
	_ tts.WordTimestamps          = (*timedTTSSynthesizer)(nil)
	_ tts.ContextSynthesizer      = (*timedTTSSynthesizer)(nil)
)
