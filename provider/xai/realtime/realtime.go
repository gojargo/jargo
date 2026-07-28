// Package realtime is a speech-to-speech service built on xAI's Realtime
// API. Unlike the cascaded STT -> LLM -> TTS pipeline, a single bidirectional
// WebSocket carries the conversation: input audio streams up, and the model
// streams its spoken reply, its transcript, and server-side voice-activity
// events back down.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output. The session exchanges 16-bit mono PCM at the configured
// rate (24 kHz by default), so run the pipeline at that rate by setting the
// transport's input and output sample rates to match; audio at other rates is
// sent through unchanged and will sound wrong.
//
// The model's server VAD drives turn-taking: on detected user speech the
// service emits an InterruptionFrame (barge-in) so the output transport drops
// buffered bot audio. Tools are advertised to the session, including xAI's
// built-in web, X and file search, but executing a function call is not yet
// wired up.
package realtime

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultBaseURL = "wss://api.x.ai/v1/realtime"
	defaultModel   = "grok-voice-think-fast-1.0"
	// defaultVoice is xAI's default voice. The others are Rex, Sal, Eve and Leo.
	defaultVoice = "Ara"
	// defaultSampleRate is the PCM rate the session runs at when unset.
	defaultSampleRate = 24000
	// pcmFormat is the session's audio format: raw 16-bit little-endian mono PCM.
	pcmFormat = "audio/pcm"
	// keyType is the discriminator key on every wire object.
	keyType = "type"
	// readLimit bounds a single inbound WebSocket message; audio deltas are far
	// larger than the library's 32 KiB default.
	readLimit = 1 << 24
)

// errNotConnected is returned when audio is sent before the socket is open.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("xairealtime: not connected")

// errServer wraps an error event reported by the Realtime API.
//
//nolint:gochecknoglobals // sentinel error
var errServer = errors.New("xairealtime: server error")

// FileSearch configures xAI's built-in document-collection search tool.
type FileSearch struct {
	// VectorStoreIDs are the collections to search. Required.
	VectorStoreIDs []string `validate:"required,min=1"`
	// MaxResults caps how many results come back; 0 uses the service default.
	MaxResults int `validate:"omitempty,min=1"`
}

// Config configures the xAI Realtime service.
type Config struct {
	// APIKey is the xAI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the Realtime WebSocket endpoint.
	BaseURL string
	// Model is the realtime model id; empty uses a current default. xAI selects
	// the model on the handshake, so it cannot change while a session is open.
	Model string
	// Voice is the model voice (Ara, Rex, Sal, Eve or Leo); empty uses a default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// SampleRate is the PCM rate the session exchanges audio at; 0 uses 24 kHz.
	SampleRate int `validate:"omitempty,oneof=8000 16000 21050 22050 24000 32000 44100 48000"`
	// ServerVAD lets xAI detect turn boundaries and drive barge-in; nil defaults
	// to true. Set it to false for manual turn detection, in which case nothing
	// in this service commits the input buffer or asks for a response.
	ServerVAD *bool
	// WebSearch advertises xAI's built-in web-search tool to the model.
	WebSearch bool
	// XSearch advertises xAI's built-in X search tool to the model.
	XSearch bool
	// XSearchHandles restricts XSearch to these X handles; empty searches all.
	// It has no effect unless XSearch is set.
	XSearchHandles []string
	// FileSearch advertises xAI's built-in collection-search tool; nil omits it.
	FileSearch *FileSearch `validate:"omitempty"`
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// serverVAD reports whether xAI drives turn detection.
func (c Config) serverVAD() bool { return c.ServerVAD == nil || *c.ServerVAD }

// sampleRate is the configured PCM rate, or the default.
func (c Config) sampleRate() int {
	if c.SampleRate == 0 {
		return defaultSampleRate
	}
	return c.SampleRate
}
