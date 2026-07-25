// Package fal is a batch speech-to-text service backed by Fal's Wizper API. A
// turn detector upstream delimits each utterance; the whole segment is uploaded
// as a WAV data URI and transcribed in one request.
package fal

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

// errStatus is returned when the API responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("fal: unexpected status")

const defaultEndpoint = "https://fal.run/fal-ai/wizper"

// Config configures the Fal Wizper transcription service.
type Config struct {
	// APIKey is the Fal API key. Required.
	APIKey string `validate:"required"`
	// Endpoint overrides the Wizper run endpoint; empty uses the hosted endpoint.
	Endpoint string
	// Language of the audio; the zero value omits it (auto-detect). Mapped to the
	// base code.
	Language language.Language
	// Task selects the operation ("transcribe" or "translate"); empty omits it.
	Task string
	// Version pins the Wizper model version; empty omits it.
	Version string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
