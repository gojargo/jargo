// Package cartesia is a streaming text-to-speech service backed by Cartesia's
// TTS WebSocket. For each sentence it opens a connection, sends the transcript,
// and streams the raw PCM audio chunks downstream.
package cartesia

import (
	"errors"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

// errProtocol is returned when Cartesia reports an error message.
//
//nolint:gochecknoglobals // sentinel error
var errProtocol = errors.New("cartesia: protocol error")

const (
	defaultURL = "wss://api.cartesia.ai/tts/websocket"
	// defaultVersion pins the Cartesia API version; it is sent on every request.
	defaultVersion = "2026-03-01"
	defaultModel   = "sonic-3.6"
	// defaultVoiceID is a public Cartesia voice.
	defaultVoiceID = "694f9389-aac1-45b6-b726-9d9369183238"
	// defaultSampleRate is the PCM rate jargo requests from Cartesia.
	defaultSampleRate = 24000
	defaultEncoding   = "pcm_s16le"
	defaultContainer  = "raw"
	// readLimit bounds a single WebSocket message; audio chunks arrive base64.
	readLimit = 1 << 20
)

// Config configures the Cartesia TTS service.
type Config struct {
	// APIKey is the Cartesia API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Version sets the Cartesia-Version header; empty uses a pinned default.
	Version string
	// Model is the Cartesia model id; empty uses a default.
	Model string
	// VoiceID is the voice id; empty uses a default public voice.
	VoiceID string
	// Language for synthesis; the zero value leaves it unset (Cartesia defaults
	// to English). Mapped to Cartesia's base code.
	Language language.Language
	// SampleRate is the PCM rate requested from Cartesia and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_s16le".
	Encoding string
	// Container is the audio container; empty uses "raw".
	Container string
	// GenerationConfig guides generation (volume, speed, emotion) on supported
	// models; nil omits it.
	GenerationConfig *GenerationConfig
	// PronunciationDictID applies a custom pronunciation dictionary; empty omits it.
	PronunciationDictID string
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.AggregationSentence (the zero value) waits for a sentence boundary,
	// frames.AggregationToken passes each token through as it arrives, which
	// lowers latency at the cost of a less tested streaming path. It also decides
	// the default for MaxBufferDelayMs.
	TextAggregation frames.AggregationType
	// MaxBufferDelayMs is the server-side buffering window before generation
	// starts. 0 disables server buffering (the client is buffering instead); any
	// value in (0, 5000] enables managed buffering. Left nil it is derived from
	// TextAggregation: 0 for sentences, which avoids stacking client and server
	// buffering, and unset for tokens, which leaves Cartesia on its own 3000ms
	// default.
	MaxBufferDelayMs *int
	// WordTimestamps requests per-word timestamps and drives the word-aligned
	// text path: the TTS base emits a TTSTextFrame for each spoken word as its
	// audio plays, mapped back to its original written form, so the assistant
	// context records only what was actually spoken before an interruption. It is
	// off by default; leaving it off keeps the service behaving exactly as before.
	WordTimestamps bool
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
