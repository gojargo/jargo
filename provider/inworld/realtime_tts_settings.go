package inworld

import (
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// TTSSettings is the part of the Inworld synthesis configuration that can change
// while the pipeline runs.
//
// The WebSocket service sends every setting when it creates a context, so a
// change reconnects and the next context is created with the new values.
type TTSSettings struct {
	settings.TTS

	// SpeakingRate multiplies the speaking rate.
	SpeakingRate settings.Opt[float64] `settings:"speaking_rate"`
	// Temperature guides synthesis variability.
	Temperature settings.Opt[float64] `settings:"temperature"`
	// DeliveryMode trades stability against expressiveness: "STABLE" is
	// reliable and predictable, "BALANCED" the midpoint, "CREATIVE" more
	// expressive and emotionally varied. Only inworld-tts-2 takes it.
	DeliveryMode settings.Opt[string] `settings:"delivery_mode"`
	// Instruction is plain-English steering for a whole request ("speak loudly
	// and urgently"), which the HTTP API honors on inworld-tts-2. The WebSocket
	// API takes steering as inline [tags] in the text instead, so the WebSocket
	// service ignores it.
	Instruction settings.Opt[string] `settings:"instruction"`
	// EnhanceGeneration denoises the synthesized audio on the HTTP API. The
	// WebSocket API has no equivalent, so the WebSocket service ignores it.
	EnhanceGeneration settings.Opt[bool] `settings:"enhance_generation"`
}

// Aliases accepts the names Inworld's own API uses for the voice and the model.
func (TTSSettings) Aliases() map[string]string {
	return map[string]string{
		fieldVoiceID: "voice",
		fieldModelID: "model",
	}
}

// FromMap fills the settings from plain names and values, implementing
// settings.Mapper. The speaking rate is also read from a nested audioConfig, the
// shape Inworld's own requests carry it in.
func (s *TTSSettings) FromMap(values map[string]any) error {
	flat := make(map[string]any, len(values))
	for k, v := range values {
		if k == fieldAudioConfig {
			continue
		}
		flat[k] = v
	}
	if nested, ok := values[fieldAudioConfig].(map[string]any); ok {
		if _, given := flat["speaking_rate"]; !given {
			flat["speaking_rate"] = nested["speakingRate"]
		}
	}
	return settings.FromMapFields(s, flat)
}

// newTTSSettings is the starting state, taken from what the WebSocket service
// was built with.
func newTTSSettings(cfg RealtimeTTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = settings.Set(cfg.VoiceID)
	// The language is stored as the caller gave it; the service names it the
	// way Inworld does.
	if cfg.Language != "" {
		s.Language = settings.Set(string(cfg.Language))
	}
	if cfg.SpeakingRate != nil {
		s.SpeakingRate = settings.Set(*cfg.SpeakingRate)
	}
	if cfg.Temperature != nil {
		s.Temperature = settings.Set(*cfg.Temperature)
	}
	if cfg.DeliveryMode != "" {
		s.DeliveryMode = settings.Set(cfg.DeliveryMode)
	}
	return s
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *realtimeSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Inworld does, implementing
// tts.LanguageNamer.
func (s *realtimeSynthesizer) ServiceLanguage(l language.Language) string {
	return inworldLanguage(l)
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*realtimeSynthesizer)(nil)
	_ tts.SettingsUpdater = (*realtimeSynthesizer)(nil)
	_ tts.LanguageNamer   = (*realtimeSynthesizer)(nil)
	_ settings.Aliaser    = TTSSettings{}
	_ settings.Mapper     = (*TTSSettings)(nil)
)
