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
// The model's server VAD drives turn-taking: the service proposes each boundary
// it detects and the external turn strategies it recommends resolve those
// proposals into turn frames and the barge-in that drops buffered bot audio.
// Tools are advertised to the session, including xAI's built-in web, X and file
// search, and a call the model makes is run and answered back into the session.
package realtime

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultBaseURL = "wss://api.x.ai/v1/realtime"
	// defaultModel is xAI's recommended Voice Agent alias, which tracks their
	// current model. Pin a versioned model for stability.
	defaultModel = "grok-voice-latest"
	// defaultVoice is the voice xAI documents as its own default.
	defaultVoice = "eve"
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

// errNotGenerator is returned by the generation entry point this service does
// not use: it generates continuously rather than answering a conversation.
//
//nolint:gochecknoglobals // sentinel error
var errNotGenerator = errors.New("xairealtime: the model generates continuously")

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

// Transcription configures the transcript the session produces of the user's
// audio. It is what the pipeline reads the user's words from, so a session
// without it hears the user but reports nothing they said.
type Transcription struct {
	// Model is the transcription model, "grok-transcribe" for the streaming one
	// that reports a transcript as the user speaks; empty leaves xAI's default.
	Model string
	// LanguageHint is a BCP-47 code biasing recognition toward one language.
	LanguageHint string
	// Keyterms bias recognition toward domain terms: at most 100 of them, each
	// at most 50 characters.
	Keyterms []string `validate:"omitempty,max=100,dive,max=50"`
}

// VADParams tunes xAI's server-side voice-activity detection. It applies only
// while ServerVAD is on; a zero field leaves xAI's own default.
type VADParams struct {
	// Threshold is the activation threshold (0.1 to 0.9). A higher one needs
	// louder speech before the server calls it a turn.
	Threshold float64 `validate:"omitempty,min=0.1,max=0.9"`
	// SilenceMS is how long the user has to stop for before the server ends
	// their turn.
	SilenceMS int `validate:"omitempty,min=1"`
	// PrefixPaddingMS is how much of the audio before the detected start of
	// speech the server keeps.
	PrefixPaddingMS int `validate:"omitempty,min=1"`
	// IdleTimeoutMS asks the server to speak again on its own after this much
	// silence following one of its responses; 0 leaves it quiet.
	IdleTimeoutMS int `validate:"omitempty,min=1"`
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
	// Voice is the voice the model speaks in: a built-in voice id from xAI's
	// catalog or a custom one from their Custom Voices API. Ids are
	// case-insensitive; empty uses xAI's own default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// SampleRate is the PCM rate the session exchanges audio at; 0 uses 24 kHz.
	SampleRate int `validate:"omitempty,oneof=8000 16000 21050 22050 24000 32000 44100 48000"`
	// ServerVAD lets xAI detect turn boundaries and drive barge-in; nil defaults
	// to true. Set it to false for manual turn detection, where the pipeline's
	// own turn frames commit the input buffer and ask for a response.
	ServerVAD *bool
	// VAD tunes the server-side voice-activity detection; nil leaves xAI's
	// defaults. It has no effect while ServerVAD is false.
	VAD *VADParams `validate:"omitempty"`
	// Transcription configures the transcript of the user's audio; nil leaves
	// the session's default.
	Transcription *Transcription `validate:"omitempty"`
	// Reasoning sets how much the model thinks before it answers: "high" turns
	// reasoning on, "none" turns it off. Empty leaves the field off the session,
	// so xAI's own default applies.
	Reasoning string `validate:"omitempty,oneof=high none"`
	// Resumption asks the server to cache the conversation's turns so a session
	// that drops can be resumed against its conversation id.
	Resumption bool
	// Replace maps a written form onto the pronunciation the model should speak,
	// applied before synthesis.
	Replace map[string]string
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
