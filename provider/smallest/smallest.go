// Package smallest is a streaming text-to-speech service backed by Smallest AI's
// Waves WebSocket. For each sentence it opens a connection, sends the transcript,
// and streams the raw PCM audio chunks downstream.
package smallest

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

// errProtocol is returned when Smallest reports an error message.
//
//nolint:gochecknoglobals // sentinel error
var errProtocol = errors.New("smallest: protocol error")

const (
	defaultURL      = "wss://api.smallest.ai/waves/v1/tts/live"
	modelLightning  = "lightning_v3.1"
	modelLightningP = "lightning_v3.1_pro"
	defaultModel    = modelLightningP
	// defaultLanguage and defaultSampleRate are the same for every model.
	defaultLanguage   = "en"
	defaultSampleRate = 24000
)

// modelDefaultVoices is the voice each model speaks with when the caller names
// none. The two models have different voice catalogs, so one default for both
// would name a voice the other does not serve.
//
//nolint:gochecknoglobals // a fixed table
var modelDefaultVoices = map[string]string{
	modelLightning:  "sophia",
	modelLightningP: "meher",
}

// Config configures the Smallest AI TTS service.
type Config struct {
	// APIKey is the Smallest AI API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the Smallest model id; empty uses a default.
	Model string
	// Voice is the voice id; empty uses a default voice.
	Voice string
	// Language is the synthesis language code; empty uses "en".
	Language string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int
	// Speed multiplies the speaking rate; nil uses the service default.
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
