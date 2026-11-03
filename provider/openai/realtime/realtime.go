// Package realtime is a speech-to-speech service built on OpenAI's
// Realtime API. Unlike the cascaded STT -> LLM -> TTS pipeline, a single
// bidirectional WebSocket carries the conversation: input audio streams up, and
// the model streams its spoken reply, its transcript, and server-side
// voice-activity events back down.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output. The Realtime API exchanges 16-bit mono PCM at 24 kHz: input
// audio at any other rate is resampled to it before it is sent, and the model's
// audio comes out at 24 kHz.
//
// The session's properties (events.SessionProperties) configure it: its audio,
// transcription, turn detection, voice and tools. By default the server detects
// the user's turns; the service proposes each boundary it reports, and the
// pipeline's turn strategies turn the proposals into turns and interruptions.
//
// Tools reach the model from the conversation, or from the session properties
// while the conversation advertises none, and the ones carrying a handler are
// run when it calls them: the result goes back to the session and the model is
// asked to speak the answer.
package realtime

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
)

const (
	defaultBaseURL = "wss://api.openai.com/v1/realtime"
	defaultModel   = "gpt-realtime-2.1"
	// sampleRate is the fixed rate of the Realtime API's pcm16 audio format.
	sampleRate = 24000
	// readLimit bounds a single inbound WebSocket message; audio deltas are far
	// larger than the library's 32 KiB default.
	readLimit = 1 << 24
	// Wire keys and values shared by the session and transcription messages.
	keyType          = "type"
	keyAudio         = "audio"
	msgSessionUpdate = "session.update"
	// roleAssistant is the role of the model's messages.
	roleAssistant = "assistant"
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
	// Model is the realtime model id; empty uses a current default. It is
	// chosen when the session opens and cannot change while it runs.
	Model string
	// Instructions is the system instruction for the session.
	Instructions string
	// SessionProperties configure the session: its audio (format,
	// transcription, turn detection, noise reduction, voice), modalities,
	// tools, reasoning and the rest. A property left unset leaves the server's
	// own, so the user's audio is not transcribed unless a transcription is
	// configured. The model and instructions are kept in step with the fields
	// above, which win.
	SessionProperties events.SessionProperties
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
