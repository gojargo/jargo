// Package soniox provides Soniox's real-time speech services over its WebSocket
// APIs.
//
// Speech-to-text (NewSTT) streams tokens, each marked final or provisional, and
// signals end-of-turn with a special "<end>" token; the service surfaces
// provisional text as interims and the finalized utterance with EndOfTurn set.
//
// Text-to-speech (NewTTS) opens a synthesis stream per sentence, sends the text,
// and streams the audio chunks downstream as Soniox generates them, reporting
// per-word timing from the character timestamps it returns.
package soniox

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

const (
	defaultURL   = "wss://stt-rt.soniox.com/transcribe-websocket"
	defaultModel = "stt-rt-v5"
	// readLimit bounds a single inbound message.
	readLimit = 1 << 20
	// endToken and finToken are Soniox's in-band turn-boundary markers.
	endToken = "<end>"
	finToken = "<fin>"
)

// errServer wraps an error reported by Soniox.
//
//nolint:gochecknoglobals // sentinel error
var errServer = errors.New("soniox: server error")

// Config configures the Soniox STT service.
type Config struct {
	// APIKey is the Soniox API key, sent in the config handshake. Required.
	APIKey string `validate:"required"`
	// URL overrides the real-time endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the transcription model; empty uses a current default.
	Model string
	// Language hints the spoken language; the zero value lets Soniox auto-detect.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// EnableEndpointDetection emits an end-of-turn marker on detected silence;
	// nil defaults to true.
	EnableEndpointDetection *bool

	// The remaining options may also be changed while the pipeline runs, through
	// an STTUpdateSettingsFrame. A change reopens the session, since Soniox is
	// told all of this in the handshake that opens one.

	// LanguageHintsStrict confines recognition to the hinted language; nil omits
	// it and lets Soniox decide.
	LanguageHintsStrict *bool
	// Context primes the model with what the conversation is about, for the
	// models that take it; nil omits it.
	Context *Context
	// EnableSpeakerDiarization labels speakers in the transcript; nil omits it.
	EnableSpeakerDiarization *bool
	// EnableLanguageIdentification reports the language of each token; nil omits
	// it.
	EnableLanguageIdentification *bool
	// MaxEndpointDelayMs bounds how long Soniox waits before ending a turn; nil
	// omits it.
	MaxEndpointDelayMs *int
	// EndpointSensitivity tunes how readily Soniox calls a turn ended; nil omits
	// it.
	EndpointSensitivity *float64
	// EndpointLatencyAdjustmentLevel trades endpoint latency against accuracy;
	// nil omits it.
	EndpointLatencyAdjustmentLevel *int
	// ClientReferenceID labels the session in Soniox's own records; empty omits
	// it.
	ClientReferenceID string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
