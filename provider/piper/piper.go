// Package piper is a text-to-speech provider for a local Piper HTTP server
// (python -m piper.http_server). The server accepts the text to speak as the
// request body and returns a WAV file; this provider strips the WAV header and
// streams the raw PCM samples downstream.
//
// Piper's sample rate depends on the voice the server was launched with (16 kHz
// for "low" voices, 22.05 kHz for "medium"). Set SampleRate to match the voice;
// the pipeline resamples to the output rate from there.
package piper

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

// defaultSampleRate is the rate of Piper's "medium" voices.
const defaultSampleRate = 22050

// emitChunk is the size of each PCM chunk pushed downstream.
const emitChunk = 4096

// errStatus is returned when the server responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("piper: unexpected status")

// errBadWAV is returned when the response is not a parseable WAV file.
//
//nolint:gochecknoglobals // sentinel error
var errBadWAV = errors.New("piper: response is not a WAV file")

// Config configures the Piper TTS provider.
type Config struct {
	// BaseURL is the Piper HTTP server's synthesis endpoint (e.g.
	// http://localhost:5000). The text to speak is POSTed as the body. Required.
	BaseURL string `validate:"required,url"`
	// SampleRate is the PCM rate of the configured Piper voice; 0 uses 22050.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
