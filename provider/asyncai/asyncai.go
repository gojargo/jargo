// Package asyncai is a streaming text-to-speech service backed by Async's TTS
// WebSocket. For each sentence it opens a connection, sends an init message and
// the transcript, and streams the raw PCM audio chunks downstream.
package asyncai

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultURL        = "wss://api.async.com/text_to_speech/websocket/ws"
	defaultVersion    = "v1"
	defaultModel      = "async_flash_v1.5"
	defaultLanguage   = "en"
	defaultEncoding   = "pcm_s16le"
	defaultContainer  = "raw"
	defaultSampleRate = 24000
)

// Config configures the Async TTS service.
type Config struct {
	// APIKey is the Async API key. Required.
	APIKey string `validate:"required"`
	// Voice is the voice id (UUID) to synthesize with. Required.
	Voice string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Version sets the API version; empty uses a default.
	Version string
	// Model is the Async model id; empty uses a default.
	Model string
	// Language is the synthesis language code; empty uses "en".
	Language string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
