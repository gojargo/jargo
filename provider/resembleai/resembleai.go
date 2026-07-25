// Package resembleai is a streaming text-to-speech service backed by Resemble
// AI's WebSocket. For each sentence it opens a connection, sends the transcript,
// and streams the raw PCM audio chunks downstream. The request sets
// no_audio_header so the audio arrives as headerless 16-bit PCM.
package resembleai

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

// errProtocol is returned when Resemble reports an error message.
//
//nolint:gochecknoglobals // sentinel error
var errProtocol = errors.New("resembleai: protocol error")

const (
	defaultURL        = "wss://websocket.cluster.resemble.ai/stream"
	defaultSampleRate = 22050
	defaultPrecision  = "PCM_16"
)

// Config configures the Resemble AI TTS service.
type Config struct {
	// APIKey is the Resemble AI API key. Required.
	APIKey string `validate:"required"`
	// Voice is the voice UUID to synthesize with. Required.
	Voice string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 22.05 kHz.
	SampleRate int
	// Precision is the audio precision; empty uses "PCM_16".
	Precision string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
