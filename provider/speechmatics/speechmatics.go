// Package speechmatics provides Speechmatics' real-time streaming speech-to-text
// over its WebSocket API. It pushes InterimTranscriptionFrames as partials
// arrive and a finalized TranscriptionFrame at each end-of-utterance.
package speechmatics

import (
	"errors"
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

const (
	defaultURL                   = "wss://eu.rt.speechmatics.com/v2"
	defaultOperatingPoint        = "enhanced"
	defaultMaxDelay              = 1.0
	defaultEndOfUtteranceSilence = 0.5
	// readLimit bounds a single inbound message; transcript messages are small.
	readLimit = 1 << 20
)

// errServer wraps an error reported by Speechmatics.
//
//nolint:gochecknoglobals // sentinel error
var errServer = errors.New("speechmatics: server error")

// Config configures the Speechmatics STT service.
type Config struct {
	// APIKey is the Speechmatics API key, sent as a Bearer token. Required.
	APIKey string `validate:"required"`
	// URL overrides the real-time endpoint; empty uses the EU endpoint. Use the
	// US endpoint (wss://us.rt.speechmatics.com/v2) for US data residency.
	URL string
	// Language is the transcription language; empty uses English.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// OperatingPoint trades accuracy for latency ("standard" or "enhanced");
	// empty uses "enhanced".
	OperatingPoint string
	// EnablePartials emits interim results; nil defaults to true.
	EnablePartials *bool
	// MaxDelay is the maximum finalization delay in seconds; 0 uses 1.0.
	MaxDelay float64
	// EndOfUtteranceSilence is the silence in seconds that ends an utterance; nil
	// uses 0.5. Set 0 to disable end-of-utterance detection (must be < MaxDelay).
	EndOfUtteranceSilence *float64

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.SpeechmaticsTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
