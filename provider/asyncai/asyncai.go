// Package asyncai is a streaming text-to-speech service backed by Async's TTS
// WebSocket. It holds one connection open for the call: the session is opened
// with an init message when the pipeline starts, each sentence is sent on the
// context it belongs to, and the raw PCM audio streams back on that context.
package asyncai

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/wsservice"
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
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
