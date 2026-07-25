// Package xtts provides text-to-speech via a self-hosted Coqui XTTS streaming
// server (https://github.com/coqui-ai/xtts-streaming-server). The server holds
// the voices; this client fetches a studio speaker's embeddings once and streams
// raw 24 kHz PCM for each sentence.
package xtts

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

// Sentinel errors for the studio-speaker lookup.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errStatus       = errors.New("xtts: unexpected studio_speakers status")
	errUnknownVoice = errors.New("xtts: unknown studio speaker")
)

const (
	defaultLanguage   = "en"
	defaultSampleRate = 24000 // XTTS streams 24 kHz PCM.
)

// Config configures a Coqui XTTS streaming-server TTS service.
type Config struct {
	// BaseURL is the XTTS streaming server base (e.g. http://localhost:8000). Required.
	BaseURL string `validate:"required"`
	// Voice is the studio speaker name to synthesize with. Required.
	Voice string `validate:"required"`
	// Language is the synthesis language code; empty uses "en".
	Language string
	// SampleRate is the PCM output rate reported downstream; 0 uses 24 kHz.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
