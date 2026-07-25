// Package neuphonic is a streaming text-to-speech service backed by Neuphonic's
// TTS WebSocket. For each sentence it opens a connection, sends the transcript
// terminated by a <STOP> marker, and streams the base64 PCM chunks downstream
// until Neuphonic marks the response complete.
package neuphonic

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

// errProtocol is returned when Neuphonic reports an error message.
//
//nolint:gochecknoglobals // sentinel error
var errProtocol = errors.New("neuphonic: protocol error")

const (
	defaultURL        = "wss://api.neuphonic.com"
	defaultSampleRate = 22050
	defaultEncoding   = "pcm_linear"
	defaultSpeed      = 1.0
)

// Config configures the Neuphonic TTS service.
type Config struct {
	// APIKey is the Neuphonic API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket base; empty uses the hosted endpoint.
	URL string
	// VoiceID is the voice id; empty uses Neuphonic's default voice.
	VoiceID string
	// Language for synthesis; the zero value uses English. Mapped to Neuphonic's
	// language code.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 22.05 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_linear".
	Encoding string
	// Speed is the speech-speed multiplier; 0 uses 1.0.
	Speed float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
