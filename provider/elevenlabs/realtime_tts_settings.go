package elevenlabs

import (
	"context"
	"log/slog"
	"slices"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// RealtimeTTSSettings is the part of the WebSocket synthesis configuration that
// can change while the pipeline runs.
//
// The voice, the model, the language and the text normalization are query
// parameters of the connection, so a change to any of them reconnects. The
// voice controls (stability, similarity boost, style, speaker boost, speed)
// are sent when a context opens, so a change to those closes the open
// contexts and the next one opens with the new values.
type RealtimeTTSSettings struct {
	settings.TTS

	// Stability controls how stable the voice is, from 0 to 1.
	Stability settings.Opt[float64] `settings:"stability"`
	// SimilarityBoost controls how closely the voice keeps to the original,
	// from 0 to 1.
	SimilarityBoost settings.Opt[float64] `settings:"similarity_boost"`
	// Style controls how much style the voice is given, from 0 to 1.
	Style settings.Opt[float64] `settings:"style"`
	// UseSpeakerBoost turns on the speaker boost.
	UseSpeakerBoost settings.Opt[bool] `settings:"use_speaker_boost"`
	// Speed controls how fast the voice speaks, from 0.7 to 1.2.
	Speed settings.Opt[float64] `settings:"speed"`
	// ApplyTextNormalization is how numbers and abbreviations are read: "auto",
	// "on" or "off".
	ApplyTextNormalization settings.Opt[string] `settings:"apply_text_normalization"`
}

// realtimeURLFields are the settings carried in the connection URL. A change to
// any of them only takes effect on a new connection.
//
//nolint:gochecknoglobals // the set of fields, fixed
var realtimeURLFields = []string{"voice", "model", "language", "apply_text_normalization"}

// realtimeVoiceSettingsFields are the settings carried in a context's opening
// message. A change to any of them takes effect on the next context.
//
//nolint:gochecknoglobals // the set of fields, fixed
var realtimeVoiceSettingsFields = []string{"stability", "similarity_boost", "style", "use_speaker_boost", "speed"}

// newRealtimeTTSSettings is the starting state, taken from what the service was
// built with.
func newRealtimeTTSSettings(cfg RealtimeTTSConfig) *RealtimeTTSSettings {
	s := &RealtimeTTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = settings.Set(cfg.VoiceID)
	// A language ElevenLabs takes is stored as the caller gave it; the service
	// names it the way ElevenLabs does.
	if elevenlabsLanguage(cfg.Language) != "" {
		s.Language = settings.Set(string(cfg.Language))
	}
	if v := cfg.VoiceSettings; v != nil {
		setOpt(&s.Stability, v.Stability)
		setOpt(&s.SimilarityBoost, v.SimilarityBoost)
		setOpt(&s.Style, v.Style)
		setOpt(&s.UseSpeakerBoost, v.UseSpeakerBoost)
		setOpt(&s.Speed, v.Speed)
	}
	if cfg.ApplyTextNormalization != "" {
		s.ApplyTextNormalization = settings.Set(cfg.ApplyTextNormalization)
	}
	return s
}

// setOpt gives o the value v points to, leaving it unset when v is nil.
func setOpt[T any](o *settings.Opt[T], v *T) {
	if v != nil {
		*o = settings.Set(*v)
	}
}

// voiceSettings builds the voice settings a context opens with, or nil when none
// is set and the voice's own defaults apply.
func (s *RealtimeTTSSettings) voiceSettings() *VoiceSettings {
	v := &VoiceSettings{
		Stability:       s.Stability.Ptr(),
		SimilarityBoost: s.SimilarityBoost.Ptr(),
		Style:           s.Style.Ptr(),
		UseSpeakerBoost: s.UseSpeakerBoost.Ptr(),
		Speed:           s.Speed.Ptr(),
	}
	if v.Stability == nil && v.SimilarityBoost == nil && v.Style == nil &&
		v.UseSpeakerBoost == nil && v.Speed == nil {
		return nil
	}
	return v
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *realtimeSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way ElevenLabs does, implementing
// tts.LanguageNamer. Without it a neutral name and the stored code would be in
// different vocabularies, and the same language would read as a change.
func (s *realtimeSynthesizer) ServiceLanguage(l language.Language) string {
	return elevenlabsLanguage(l)
}

// UpdateSettings acts on a change, implementing tts.SettingsUpdater.
//
// A setting carried in the connection URL only takes effect on a new
// connection, so the service disconnects and connects again. A voice control
// is sent when a context opens, so every open context is closed and the next
// one opens with the new values. A reconnect applies every setting; otherwise
// the fields neither path applies are reported as not taking effect.
func (s *realtimeSynthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.refreshConnectionSettings()

	urlChanged := changedAny(changed, realtimeURLFields)
	voiceSettingsChanged := changedAny(changed, realtimeVoiceSettingsFields)

	switch {
	case urlChanged:
		slog.Debug("elevenlabs realtime tts: a URL setting changed, reconnecting", "fields", changed.String())
		s.Disconnect(ctx)
		s.Connect(ctx)
	case voiceSettingsChanged:
		slog.Debug("elevenlabs realtime tts: voice settings changed, closing the open contexts to apply them",
			"fields", changed.String())
		for _, id := range s.svc.AudioContextIDs() {
			s.closeContext(ctx, id)
			s.resetAlignmentState(id)
		}
	}

	if !urlChanged {
		handled := append(append([]string(nil), realtimeURLFields...), realtimeVoiceSettingsFields...)
		s.svc.WarnUnhandledUpdatedSettings(ctx, changed.Except(handled...))
	}
	return nil
}

// changedAny reports whether any of fields changed.
func changedAny(changed settings.Changed, fields []string) bool {
	return slices.ContainsFunc(fields, changed.Has)
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder  = (*realtimeSynthesizer)(nil)
	_ tts.SettingsUpdater = (*realtimeSynthesizer)(nil)
	_ tts.LanguageNamer   = (*realtimeSynthesizer)(nil)
)
