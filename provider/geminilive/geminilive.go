// Package geminilive is a speech-to-speech service built on Google's Gemini
// Live API (BidiGenerateContent). A single bidirectional WebSocket carries the
// conversation: input audio streams up and the model streams its spoken reply,
// its transcript, and the user's transcript back down.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output. The Live API takes 16 kHz mono PCM in and returns 24 kHz
// mono PCM out, so run the transport input at 16000 and output at 24000.
//
// The model's server VAD drives turn-taking: when it reports an interruption the
// service emits an InterruptionFrame (barge-in) so the output transport drops
// buffered bot audio.
package geminilive

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultBaseURL = "wss://generativelanguage.googleapis.com/ws/" +
		"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
	defaultModel = "gemini-live-2.5-flash-native-audio"
	defaultVoice = "Puck"
	// outputSampleRate is the fixed rate of the Live API's output audio.
	outputSampleRate = 24000
	// readLimit bounds a single inbound message; audio parts are large.
	readLimit = 1 << 24
)

// Config configures the Gemini Live service.
type Config struct {
	// APIKey is the Google AI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the Live API WebSocket endpoint.
	BaseURL string
	// Model is the Live model id (without the "models/" prefix); empty uses a
	// current default.
	Model string
	// Voice is the prebuilt voice name (e.g. "Puck", "Kore"); empty uses a
	// default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
