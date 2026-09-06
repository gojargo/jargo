// Package minimax is a streaming text-to-speech service backed by MiniMax's T2A
// API. Unlike most providers, MiniMax streams audio as hex-encoded PCM inside
// Server-Sent Events rather than as a raw byte stream, so this service parses
// the SSE frames and hex-decodes each chunk before emitting it downstream.
package minimax

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultBaseURL    = "https://api.minimax.io"
	defaultModel      = "speech-2.8-turbo"
	defaultVoice      = "Calm_Woman"
	defaultSampleRate = 24000
	// sseMaxLine bounds a single SSE line; a hex-encoded audio chunk can be large.
	sseMaxLine = 8 << 20
	// statusFinal marks the summary event, whose audio (if any) repeats earlier
	// chunks and must not be emitted again.
	statusFinal = 2
)

// errStatus is returned when the API responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("minimax: unexpected status")

// Config configures the MiniMax TTS service.
type Config struct {
	// APIKey is the MiniMax API key. Required.
	APIKey string `validate:"required"`
	// GroupID is the MiniMax group id, sent as the GroupId query parameter.
	// Required.
	GroupID string `validate:"required"`
	// BaseURL overrides the API base; empty uses the hosted API.
	BaseURL string
	// Model is the T2A model; empty uses a current default.
	Model string
	// VoiceID is the voice; empty uses a default.
	VoiceID string
	// SampleRate is the PCM output rate; empty uses 24 kHz. Supported: 8000,
	// 16000, 22050, 24000, 32000, 44100.
	SampleRate int
	// Speed scales the speaking rate (0.5 to 2.0); nil leaves it at 1.0.
	Speed *float64
	// Volume scales the loudness (>0 to 10); nil leaves it at the default.
	Volume *float64
	// Pitch shifts the pitch (-12 to 12); nil leaves it at 0.
	Pitch *int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
