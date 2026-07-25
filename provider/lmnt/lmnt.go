// Package lmnt is a streaming text-to-speech service backed by LMNT's HTTP
// "speech bytes" endpoint, which streams raw PCM.
package lmnt

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	apiURL       = "https://api.lmnt.com/v1/ai/speech/bytes"
	defaultModel = "blizzard"
	defaultVoice = "leah"
	defaultLang  = "en"
	// defaultSampleRate is the PCM rate jargo requests from LMNT.
	defaultSampleRate = 24000
)

// Config configures the LMNT TTS service.
type Config struct {
	// APIKey is the LMNT API key. Required.
	APIKey string `validate:"required"`
	// Model is the LMNT model; empty uses a default.
	Model string
	// Voice is the voice id; empty uses a default.
	Voice string
	// Lang is the spoken language; empty uses "en".
	Lang string
	// SampleRate is the PCM rate requested from LMNT and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
