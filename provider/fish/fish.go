// Package fish is a streaming text-to-speech service backed by Fish Audio. It
// requests PCM output so raw audio streams straight downstream.
package fish

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultBaseURL    = "https://api.fish.audio"
	defaultModel      = "s2.1-pro"
	defaultSampleRate = 24000
)

// Config configures the Fish Audio TTS service.
type Config struct {
	// APIKey is the Fish Audio API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base; empty uses the hosted API.
	BaseURL string
	// Model is the backend model, sent as the "model" request header; empty uses
	// a current default.
	Model string
	// ReferenceID selects a voice model (a library or cloned voice). Recommended;
	// empty uses the model's default voice.
	ReferenceID string
	// SampleRate is the PCM output rate; empty uses 24 kHz. Supported: 8000,
	// 16000, 24000, 32000, 44100.
	SampleRate int
	// Speed scales the speaking rate; nil leaves it at the default.
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
