// Package live is a speech-to-speech service built on Google's Gemini
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
package live

import (
	"errors"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/google/gemini"
)

// errNotGenerator is returned by the generation entry point this service does
// not use: it generates continuously rather than answering a conversation.
//
//nolint:gochecknoglobals // sentinel error
var errNotGenerator = errors.New("geminilive: the model generates continuously")

const (
	defaultBaseURL = "wss://generativelanguage.googleapis.com/ws/" +
		"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
	defaultModel = "gemini-3.8-live"
	defaultVoice = "Charon"
	// outputSampleRate is the fixed rate of the Live API's output audio.
	outputSampleRate = 24000
	// readLimit bounds a single inbound message; audio parts are large.
	readLimit = 1 << 24
	// modalityAudio and modalityText are the Live API's modality names.
	modalityAudio = "AUDIO"
	modalityText  = "TEXT"
	// keyText and keyTurnComplete are the keys of a text part and of the flag
	// that completes a turn of client content.
	keyText         = "text"
	keyTurnComplete = "turnComplete"
	// The keys of the client messages that carry content and realtime input,
	// of a content's parts, and the role of the user's turns.
	keyClientContent = "clientContent"
	keyRealtimeInput = "realtimeInput"
	keyParts         = "parts"
	keyTurns         = "turns"
	roleUser         = "user"
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
	// Voice is the prebuilt voice name (e.g. "Puck", "Kore"); empty uses
	// Charon.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// MaxTokens bounds each response; nil uses 4096.
	MaxTokens *int `validate:"omitempty,min=1"`
	// Temperature is the sampling temperature (0 to 2); nil leaves the model's.
	Temperature *float64 `validate:"omitempty,min=0,max=2"`
	// TopP is the nucleus sampling probability (0 to 1); nil leaves the model's.
	TopP *float64 `validate:"omitempty,min=0,max=1"`
	// TopK is the top-k sampling cutoff; nil leaves the model's.
	TopK *int `validate:"omitempty,min=0"`
	// FrequencyPenalty discourages repeating the same tokens (0 to 2); nil
	// leaves the model's.
	FrequencyPenalty *float64 `validate:"omitempty,min=0,max=2"`
	// PresencePenalty discourages repeating the same subjects (0 to 2); nil
	// leaves the model's.
	PresencePenalty *float64 `validate:"omitempty,min=0,max=2"`
	// Seed makes generation reproducible; nil leaves it random.
	Seed *int
	// Modalities is what the model responds with; empty uses audio.
	Modalities Modality `validate:"omitempty,oneof=AUDIO TEXT"`
	// Language is the language the model speaks; empty uses en-US. A language
	// Gemini Live lists is sent as its locale code ("es" as "es-ES"); any other
	// is sent as it stands.
	Language language.Language
	// MediaResolution is how finely the model sees images and video; empty
	// leaves it to the model.
	MediaResolution MediaResolution
	// VAD tunes the server's voice activity detection; nil leaves its defaults.
	VAD *VADParams
	// TurnCoverage is which realtime input a user turn covers; empty uses the
	// model's own default.
	TurnCoverage TurnCoverage
	// ContextWindowCompression bounds the context window by sliding it; nil
	// leaves it off.
	ContextWindowCompression *ContextWindowCompression
	// EnableAffectiveDialog lets the model adapt to expression and tone, on a
	// model that supports it.
	EnableAffectiveDialog bool
	// Proactivity lets the model decide how to behave on its own, on a model
	// that supports it; nil leaves it off.
	Proactivity *Proactivity
	// Tools are tools configured on the service itself, as an alternative to
	// advertising them through the conversation. They are what the session
	// carries while the conversation advertises none, and the handlers of the
	// standard ones register without a separate RegisterFunction call. A
	// conversation that brings tools of its own replaces them. Tools in Gemini's
	// own format go under Custom, keyed frames.AdapterTypeGemini.
	Tools frames.ToolsSchema
	// InferenceOnContextInitialization says whether the model answers the
	// conversation the session is seeded with as soon as it is: the bot speaks
	// first. Nil means true. False seeds the session and waits for the user.
	InferenceOnContextInitialization *bool
	// StartAudioPaused starts the service with user audio paused; see
	// Service.SetAudioInputPaused.
	StartAudioPaused bool
	// UserAudioPrerollSecs pins how much user audio is kept, while server-side
	// voice activity detection is disabled, to replay once the user's turn
	// starts so its onset is not clipped. Nil sizes it from the start delay of
	// the pipeline's own voice activity detection plus a margin, and uses half
	// a second until that is known. Pin it for a turn-start strategy that is
	// not driven by voice activity.
	UserAudioPrerollSecs *float64 `validate:"omitempty,min=0"`
	// Thinking configures what the model may spend on reasoning, for a model
	// that reasons. The Live thinking models require a level and refuse a setup
	// without one, so an unset level defaults to the lowest they accept, which
	// keeps the reply latency down. Nil leaves the field off the setup entirely.
	Thinking *gemini.ThinkingConfig `validate:"omitempty"`
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
