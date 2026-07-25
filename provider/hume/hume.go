// Package hume is a streaming text-to-speech service backed by Hume AI's Octave
// TTS. It uses the /stream/file endpoint with PCM output so raw audio streams
// straight downstream. Hume's PCM is fixed at 48 kHz; the output transport
// resamples to the pipeline rate.
package hume

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultBaseURL = "https://api.hume.ai"
	// humeSampleRate is Hume's fixed PCM output rate.
	humeSampleRate       = 48000
	defaultVoiceProvider = "HUME_AI"
)

// Config configures the Hume TTS service.
type Config struct {
	// APIKey is the Hume API key, sent as X-Hume-Api-Key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base; empty uses the hosted API.
	BaseURL string
	// VoiceID selects a Hume voice by id. Setting a voice (id or name) is
	// recommended; with none, Hume invents a new voice each call.
	VoiceID string
	// VoiceName selects a Hume voice by name, when VoiceID is empty.
	VoiceName string
	// VoiceProvider is the voice's provider, "HUME_AI" (default) or
	// "CUSTOM_VOICE".
	VoiceProvider string
	// Description is an optional acting/voice prompt steering delivery.
	Description string
	// Version pins the Octave model version (e.g. "2"); empty leaves it unset.
	Version string
	// Speed scales the speaking rate; nil leaves it at the voice default.
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
