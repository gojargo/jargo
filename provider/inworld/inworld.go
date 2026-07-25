// Package inworld is a streaming text-to-speech service backed by Inworld AI's
// HTTP streaming endpoint. For each sentence it posts the text and reads the
// newline-delimited JSON response, decoding each chunk's base64 audio (stripping
// any WAV header) and streaming the 16-bit PCM downstream.
package inworld

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

// errStatus is returned when Inworld responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("inworld: unexpected status")

const (
	defaultURL        = "https://api.inworld.ai/tts/v1/voice:stream"
	defaultVoiceID    = "Ashley"
	defaultModel      = "inworld-tts-2"
	defaultSampleRate = 48000
	defaultEncoding   = "PCM"
	// wavHeaderSize is the byte length of the RIFF/WAV header Inworld prefixes to
	// PCM chunks; it is stripped so only raw samples reach downstream.
	wavHeaderSize = 44
)

// Config configures the Inworld AI TTS service.
type Config struct {
	// APIKey is the Inworld API key (a base64 Basic credential). Required.
	APIKey string `validate:"required"`
	// URL overrides the streaming endpoint; empty uses the hosted endpoint.
	URL string
	// VoiceID is the voice id; empty uses a default voice.
	VoiceID string
	// Model is the model id; empty uses "inworld-tts-2".
	Model string
	// Language for synthesis; the zero value leaves it unset. Mapped to Inworld's
	// BCP-47 language tag.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 48 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "PCM".
	Encoding string
	// SpeakingRate multiplies the speaking rate (0.5-1.5); nil omits it.
	SpeakingRate *float64
	// Temperature guides synthesis variability; nil omits it.
	Temperature *float64
	// DeliveryMode trades stability against expressiveness ("STABLE", "BALANCED",
	// "CREATIVE"; inworld-tts-2 only); empty omits it.
	DeliveryMode string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
