package sarvam

import (
	"cmp"
	"context"
	"log/slog"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// TTSSettings is the part of the synthesis configuration that can change while
// the pipeline runs.
//
// Voice is Sarvam's speaker and Language its target language code, which a
// plain update may also name "speaker" and "target_language_code", as the API
// does. The rest are the voice controls and the buffering of the WebSocket
// protocol; a control the model does not support is kept but not sent.
//
// The model is pinned on the URL a connection is opened on, so a change of
// model reconnects, and moves the voice to the new model's default speaker
// unless the same update names one. Every other change is sent to the live
// session as a new config message.
type TTSSettings struct {
	settings.TTS

	// EnablePreprocessing toggles text preprocessing.
	EnablePreprocessing settings.Opt[bool] `settings:"enable_preprocessing"`
	// Pace multiplies the speaking rate.
	Pace settings.Opt[float64] `settings:"pace"`
	// Pitch adjusts the voice pitch. Only bulbul:v4-flash and bulbul:v2.
	Pitch settings.Opt[float64] `settings:"pitch"`
	// Loudness multiplies the volume. Only bulbul:v4-flash and bulbul:v2.
	Loudness settings.Opt[float64] `settings:"loudness"`
	// Temperature controls output randomness. Only the v3 variants.
	Temperature settings.Opt[float64] `settings:"temperature"`
	// MinBufferSize is the minimum characters buffered before audio generation.
	MinBufferSize settings.Opt[int] `settings:"min_buffer_size"`
	// MaxChunkLength is the maximum characters processed in one chunk.
	MaxChunkLength settings.Opt[int] `settings:"max_chunk_length"`
}

// Aliases accepts the names Sarvam's own API uses for the voice and the
// language. Without them those names would land in Extra, which never reaches
// the API.
func (TTSSettings) Aliases() map[string]string {
	return map[string]string{
		"voice_id":             "voice",
		"speaker":              "voice",
		"target_language_code": "language",
	}
}

// ApplyUpdate merges an update, implementing settings.Updater. A model Sarvam
// does not offer is refused with a warning: the model stays as it was and is
// not reported as changed, while the rest of the update still applies.
func (s *TTSSettings) ApplyUpdate(delta any) (settings.Changed, error) {
	changed, err := settings.ApplyFields(s, delta)
	if err != nil {
		return nil, err
	}
	if !changed.Has("model") {
		return changed, nil
	}
	model, _ := s.Model.Value()
	if _, ok := ttsModelConfigs[model]; ok {
		return changed, nil
	}
	slog.Warn("sarvam tts: unsupported model, ignoring", "model", model)
	s.Model = settings.Cleared[string]()
	if old, ok := changed["model"].(string); ok {
		s.Model = settings.Set(old)
	}
	delete(changed, "model")
	return changed, nil
}

// newTTSSettings is the starting state, taken from what the service was built
// with and settled against the model's capabilities: preprocessing is forced
// on where the model always does it, the pace is clamped to its range, and a
// control it does not support is dropped with a warning unless left at its
// neutral value.
func newTTSSettings(cfg TTSConfig, mc ttsModelConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = settings.Set(firstNonEmpty(cfg.Voice, mc.defaultSpeaker))
	// The language is stored as the caller gave it, Sarvam's default when none
	// was; the service names it the way Sarvam does.
	s.Language = settings.Set(cmp.Or(string(cfg.Language), defaultTTSLanguage))
	s.EnablePreprocessing = settings.Set(mc.preprocessingAlwaysEnabled ||
		(cfg.EnablePreprocessing != nil && *cfg.EnablePreprocessing))
	s.MinBufferSize = settings.Set(intOr(cfg.MinBufferSize, defaultMinBufferSize))
	s.MaxChunkLength = settings.Set(intOr(cfg.MaxChunkLength, defaultMaxChunkLength))
	s.Pace = settings.Set(clampPace(cfg.Pace, mc))
	s.Pitch = modelControl(cfg.Model, "pitch", cfg.Pitch, mc.supportsPitch, neutralPitch)
	s.Loudness = modelControl(cfg.Model, "loudness", cfg.Loudness, mc.supportsLoudness, neutralLoudness)
	s.Temperature = modelControl(cfg.Model, "temperature", cfg.Temperature, mc.supportsTemperature,
		neutralTemperature)
	return s
}

// modelControl is the starting value of a control only some models support.
// One the model does not support is dropped, with a warning when it was set to
// anything but its neutral value.
func modelControl(model, name string, v *float64, supported bool, neutral float64) settings.Opt[float64] {
	if v == nil {
		return settings.Cleared[float64]()
	}
	if !supported && *v != neutral {
		slog.Warn("sarvam tts: parameter is ignored for this model", "parameter", name, "model", model)
		return settings.Cleared[float64]()
	}
	return settings.Set(*v)
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Sarvam does, implementing
// tts.LanguageNamer.
func (s *synthesizer) ServiceLanguage(l language.Language) string { return resolveTTSLanguage(l) }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater.
//
// The model is pinned on the URL, so a new model needs a new connection, which
// takes the new model's capabilities and, unless the update named a voice too,
// its default speaker. Any other change is sent to the live session as a new
// config message.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if changed.Has("model") {
		model, _ := s.live.Model.Value()
		mc := ttsModelConfigs[model]
		s.mc = mc
		if !changed.Has("voice") {
			s.live.Voice = settings.Set(mc.defaultSpeaker)
		}
		s.refreshConfig()
		s.Disconnect(ctx)
		s.Connect(ctx)
		return nil
	}
	if len(changed) == 0 {
		return nil
	}
	s.refreshConfig()
	return s.sendConfig(ctx)
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*synthesizer)(nil)
	_ tts.SettingsUpdater = (*synthesizer)(nil)
	_ tts.LanguageNamer   = (*synthesizer)(nil)
	_ settings.Updater    = (*TTSSettings)(nil)
	_ settings.Aliaser    = TTSSettings{}
)
