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
	humeSampleRate = 48000
)

// Config configures the Hume TTS service.
type Config struct {
	// APIKey is the Hume API key, sent as X-Hume-Api-Key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base; empty uses the hosted API.
	BaseURL string
	// VoiceID selects the Hume voice to speak with. Required: a voice is named
	// by id and by nothing else, and without one Hume invents a new voice on
	// every call, so the bot would not sound like itself twice running.
	VoiceID string `validate:"required"`
	// Description is an optional acting/voice prompt steering delivery.
	Description string
	// Version pins the Octave model version (e.g. "2"); empty leaves it unset.
	Version string
	// Speed scales the speaking rate; nil leaves it at the voice default.
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
