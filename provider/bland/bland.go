// Package bland provides Bland text-to-speech: a realtime service streaming
// text into one session held open for the call over Bland's /v2/tts/ws
// WebSocket, and an HTTP service making one /v2/tts request per sentence.
//
// See https://docs.bland.ai/api-v2/post/tts-ws for the realtime WebSocket API
// and https://docs.bland.ai/api-v2/post/tts for the HTTP API.
package bland

import (
	"log/slog"
	"slices"

	"github.com/gojargo/jargo/service/settings"
)

const (
	// defaultSampleRate is what Bland synthesizes at when the configured rate is
	// not one it renders. 48 kHz is what BTTS_V3 generates natively, so it is
	// the shortest path to audio.
	defaultSampleRate = 48000
	// defaultVoiceID is the voice spoken in when none is configured.
	defaultVoiceID = "29158307-9893-4149-8a75-bc9ce313d64e"
	// encoding is the audio encoding requested: 16-bit little-endian PCM.
	encoding = "pcm_s16le"
	// keyVoice names the voice in a request.
	keyVoice = "voice"
)

// sampleRates are the rates Bland renders directly; asking for one of these
// avoids a resample.
//
//nolint:gochecknoglobals // read-only lookup table
var sampleRates = []int{8000, 16000, 24000, 44100, 48000}

// resolveSampleRate picks the rate Bland will synthesize at for the service's
// output rate.
//
// Bland renders a fixed set of rates. A service running at any other rate is
// served at 48 kHz and its audio is resampled to the service's rate, which
// costs a resample rather than fidelity.
func resolveSampleRate(sampleRate int) int {
	if slices.Contains(sampleRates, sampleRate) {
		return sampleRate
	}
	slog.Warn("bland cannot render the sample rate, synthesizing at the default and resampling on output",
		"rate", sampleRate, "supported", sampleRates, "default", defaultSampleRate)
	return defaultSampleRate
}

// TTSSettings is the Bland configuration that can change while the pipeline
// runs, through a TTSUpdateSettingsFrame.
//
// Bland reads only the voice, the controls and the formatting; no other field
// of the shared settings reaches it. The voice sets the model, and
// Expressiveness and Stability are calibrated for BTTS_V3.
type TTSSettings struct {
	settings.TTS

	// Expressiveness is from 0.0 to 1.0. Higher values produce more varied
	// intonation.
	Expressiveness settings.Opt[float64] `settings:"expressiveness"`
	// Stability is from 0.0 to 1.0. Higher values produce more consistent
	// delivery.
	Stability settings.Opt[float64] `settings:"stability"`
	// AutoFormatting rewrites numbers into the form the voice reads best before
	// synthesis. Phone numbers and SSNs, spelled out ("seven three two...") or
	// written as bare digits ("7327412065"), are read digit by digit in groups.
	AutoFormatting settings.Opt[bool] `settings:"auto_formatting"`
}

// newTTSSettings is the starting state: the voice, defaulting to Bland's, and
// the controls and formatting the caller set. No model and no language are
// sent to Bland.
func newTTSSettings(voice string, expressiveness, stability *float64, autoFormatting *bool) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Cleared[string]()
	if voice == "" {
		voice = defaultVoiceID
	}
	s.Voice = settings.Set(voice)
	s.Language = settings.Cleared[string]()
	s.Expressiveness = optPtr(expressiveness)
	s.Stability = optPtr(stability)
	s.AutoFormatting = optPtr(autoFormatting)
	return s
}

// optPtr is a pointer's value as a setting, cleared when nil.
func optPtr[T any](v *T) settings.Opt[T] {
	if v == nil {
		return settings.Cleared[T]()
	}
	return settings.Set(*v)
}

// controls are the voice controls the caller set. One left unset keeps Bland's
// default.
func controls(s *TTSSettings) map[string]float64 {
	out := map[string]float64{}
	if v, ok := s.Expressiveness.Value(); ok {
		out["expressiveness"] = v
	}
	if v, ok := s.Stability.Value(); ok {
		out["stability"] = v
	}
	return out
}

// applyRequestOptions adds the settings Bland reads from a session's init
// message or a /v2/tts body.
func applyRequestOptions(request map[string]any, s *TTSSettings) {
	if c := controls(s); len(c) > 0 {
		request["controls"] = c
	}
	if v, ok := s.AutoFormatting.Value(); ok {
		request["auto_formatting"] = v
	}
}
