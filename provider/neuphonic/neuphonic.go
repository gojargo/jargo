// Package neuphonic is a streaming text-to-speech service backed by Neuphonic's
// TTS WebSocket. It holds one connection open for the call, configured by its
// URL, and sends each sentence terminated by a <STOP> marker. Neuphonic's
// messages name no request, so the audio that comes back is labeled with the
// context playing.
package neuphonic

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/wsservice"
)

const (
	defaultURL        = "wss://api.neuphonic.com"
	defaultSampleRate = 22050
	defaultEncoding   = "pcm_linear"
	defaultSpeed      = 1.0
)

// Config configures the Neuphonic TTS service.
type Config struct {
	// APIKey is the Neuphonic API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket base; empty uses the hosted endpoint.
	URL string
	// VoiceID is the voice id; empty uses Neuphonic's default voice.
	VoiceID string
	// Language for synthesis; the zero value uses English. Mapped to Neuphonic's
	// language code.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 22.05 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_linear".
	Encoding string
	// Speed is the speech-speed multiplier; 0 uses 1.0.
	Speed float64
	// Temperature is the randomness introduced into the synthesis, from 0 to 1.
	// Nil leaves it to Neuphonic.
	Temperature *float64
	// IPAKey is the regional variant the sounds of <phoneme:...> tags in the
	// text are read with, either "es-castilian" or "pt-br". Empty leaves it to
	// Neuphonic.
	IPAKey string
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
