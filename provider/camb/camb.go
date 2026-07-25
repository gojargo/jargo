// Package camb is a streaming text-to-speech service backed by Camb.ai's MARS
// models. For each sentence it posts the text to Camb.ai's HTTP streaming
// endpoint and streams the raw 16-bit PCM response downstream. The model sets
// the native output rate: mars-flash and mars-instruct produce 22.05 kHz,
// mars-pro produces 48 kHz.
package camb

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

const (
	defaultURL     = "https://client.camb.ai/apis/tts-stream"
	defaultModel   = "mars-flash"
	defaultVoiceID = 147320
	// maxTextLen bounds a single request; Camb.ai rejects longer input.
	maxTextLen = 3000
	// instructModel is the only model that honors user instructions.
	instructModel = "mars-instruct"
	// proSampleRate and flashSampleRate are the MARS models' native PCM rates.
	proSampleRate   = 48000
	flashSampleRate = 22050
)

// Config configures the Camb.ai TTS service.
type Config struct {
	// APIKey is the Camb.ai API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the streaming endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the MARS model id; empty uses "mars-flash".
	Model string
	// VoiceID is the numeric voice id; 0 uses a default public voice.
	VoiceID int
	// Language for synthesis; the zero value uses "en-us". Mapped to Camb.ai's
	// lower-case, region-qualified BCP-47 code.
	Language language.Language
	// SampleRate is the PCM rate requested from Camb.ai and emitted downstream;
	// 0 uses the model's native rate (22.05 kHz for mars-flash/mars-instruct,
	// 48 kHz for mars-pro).
	SampleRate int
	// UserInstructions guide delivery for the mars-instruct model only; empty
	// omits them and other models ignore them.
	UserInstructions string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
