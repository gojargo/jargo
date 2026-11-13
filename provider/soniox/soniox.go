// Package soniox provides Soniox's real-time speech services over its WebSocket
// APIs.
//
// Speech-to-text (NewSTT) streams tokens, each marked final or provisional, and
// signals end-of-turn with a special "<end>" token; the service surfaces
// provisional text as interims and the finalized utterance with EndOfTurn set.
//
// Text-to-speech (NewTTS) holds one connection open for the call and speaks
// each turn on a stream of its own on it, sending the turn's sentences to the
// stream as they are ready and streaming the audio chunks downstream as Soniox
// generates them, reporting per-word timing from the character timestamps it
// returns.
package soniox

import (
	"net/http"
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/wsservice"
)

const (
	defaultURL   = "wss://stt-rt.soniox.com/transcribe-websocket"
	defaultModel = "stt-rt-v6"
	// audioFormat is the format of the audio the service sends: 16-bit
	// little-endian PCM.
	audioFormat = "pcm_s16le"
	// readLimit bounds a single inbound message.
	readLimit = 1 << 20
	// endToken and finToken are Soniox's in-band turn-boundary markers.
	endToken = "<end>"
	finToken = "<fin>"
	// keepaliveMessage holds an idle session open, and finalizeMessage asks for
	// the final tokens of the audio sent so far.
	keepaliveMessage = `{"type": "keepalive"}`
	finalizeMessage  = `{"type": "finalize"}`
	// keepaliveTimeout is how long a session may go without audio before the
	// keepalive is sent, and keepaliveInterval how often that is checked.
	keepaliveTimeout  = time.Second
	keepaliveInterval = 5 * time.Second
)

// Config configures the Soniox STT service.
type Config struct {
	// APIKey is the Soniox API key, sent on the WebSocket handshake. Required.
	APIKey string `validate:"required"`
	// URL overrides the real-time endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the transcription model; empty uses a current default.
	Model string
	// Language hints the spoken language; the zero value lets Soniox auto-detect.
	// It becomes the one language hint, unless LanguageHints are given.
	Language language.Language
	// LanguageHints are the languages to expect, for a conversation in more
	// than one. When given, they are the hints and Language is not added.
	LanguageHints []language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// EnableEndpointDetection has no effect: endpoint detection is always on.
	// A VAD stop finalizes the transcript sooner, and the endpoint still ends
	// the transcript of speech the VAD missed.
	//
	// Deprecated: endpoint detection is always on.
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

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.SonioxTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// authHeader is the WebSocket handshake header that authenticates a session.
// Soniox stops accepting the key in the configuration message on 2027-01-15.
func authHeader(apiKey string) http.Header {
	return http.Header{"Authorization": []string{"Bearer " + apiKey}}
}
