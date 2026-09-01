// Package realtime is a speech-to-speech service built on OpenAI's
// Realtime API. Unlike the cascaded STT -> LLM -> TTS pipeline, a single
// bidirectional WebSocket carries the conversation: input audio streams up, and
// the model streams its spoken reply, its transcript, and server-side
// voice-activity events back down.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output. The Realtime API exchanges 16-bit mono PCM at 24 kHz, so run
// the pipeline at that rate (set the transport's input and output sample rates
// to 24000); audio at other rates is sent through unchanged and will sound
// wrong.
//
// The model's server VAD drives turn-taking: on detected user speech the service
// emits an InterruptionFrame (barge-in) so the output transport drops buffered
// bot audio.
//
// Tools reach the model from the conversation, and the ones carrying a handler
// are run when it calls them: the result goes back to the session and the model
// is asked to speak the answer.
package realtime

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultBaseURL            = "wss://api.openai.com/v1/realtime"
	defaultModel              = "gpt-realtime"
	defaultVoice              = "alloy"
	defaultTranscriptionModel = "whisper-1"
	// sampleRate is the fixed rate of the Realtime API's pcm16 audio format.
	sampleRate = 24000
	// readLimit bounds a single inbound WebSocket message; audio deltas are far
	// larger than the library's 32 KiB default.
	readLimit = 1 << 24
	// Wire keys and values shared by the session and transcription messages.
	keyType          = "type"
	keyAudio         = "audio"
	msgSessionUpdate = "session.update"
)

// errNotConnected is returned when audio is sent before the socket is open.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("openairealtime: not connected")

// errNotGenerator is returned by the generation entry point this service does
// not use: it generates continuously rather than answering a conversation.
//
//nolint:gochecknoglobals // sentinel error
var errNotGenerator = errors.New("openairealtime: the model generates continuously")

// errServer wraps an error event reported by the Realtime API.
//
//nolint:gochecknoglobals // sentinel error
var errServer = errors.New("openairealtime: server error")

// Config configures the OpenAI Realtime service.
type Config struct {
	// APIKey is the OpenAI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the Realtime WebSocket endpoint.
	BaseURL string
	// Model is the realtime model id; empty uses a current default.
	Model string
	// Voice is the model voice (e.g. alloy, echo, shimmer); empty uses a default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// TranscriptionModel transcribes the user's audio; empty uses whisper-1. Set
	// to "-" to disable input transcription.
	TranscriptionModel string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
