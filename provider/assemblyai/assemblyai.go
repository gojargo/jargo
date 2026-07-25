// Package assemblyai is a streaming speech-to-text service backed by
// AssemblyAI's Universal-Streaming (v3) WebSocket. It streams audio and pushes
// InterimTranscriptionFrames and finalized TranscriptionFrames downstream.
//
// A turn is finalized once AssemblyAI reports a formatted end-of-turn, which
// marks the end of the user's speech.
package assemblyai

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

const (
	wsBase = "wss://streaming.assemblyai.com/v3/ws"
	// defaultEncoding is the audio encoding AssemblyAI expects from jargo.
	defaultEncoding = "pcm_s16le"
	// readLimit bounds a single WebSocket message; long turns carry many words.
	readLimit = 1 << 20
)

// Config configures the AssemblyAI STT service. Optional fields modeled as
// pointers or slices are omitted from the request when unset.
type Config struct {
	// APIKey is the AssemblyAI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the streaming WebSocket endpoint; empty uses the hosted
	// endpoint.
	BaseURL string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_s16le".
	Encoding string
	// Model selects the speech model (sent as speech_model); empty uses the
	// account default.
	Model string
	// Language declares the audio language (sent as language_code); the zero value
	// leaves it unset. Mapped to AssemblyAI's base code.
	Language language.Language
	// LanguageDetection enables automatic language detection; nil omits it.
	// Mutually exclusive with Language.
	LanguageDetection *bool
	// FormatTurns formats finalized turns (punctuation/casing); nil defaults to true.
	FormatTurns *bool
	// FormattedFinals formats partial finals; nil omits it.
	FormattedFinals *bool
	// WordFinalizationMaxWaitTime caps the wait in ms for word finalization;
	// nil omits it.
	WordFinalizationMaxWaitTime *int
	// EndOfTurnConfidenceThreshold sets the confidence needed to end a turn
	// (0.0-1.0); nil omits it.
	EndOfTurnConfidenceThreshold *float64
	// MinTurnSilence is the minimum silence in ms to end a turn when confident;
	// nil omits it.
	MinTurnSilence *int
	// MaxTurnSilence is the maximum silence in ms before forcing a turn end;
	// nil omits it.
	MaxTurnSilence *int
	// KeytermsPrompt boosts recognition of the given terms; empty omits it.
	KeytermsPrompt []string
	// Prompt steers transcription with free-text guidance; empty omits it.
	Prompt string
	// SpeakerLabels enables speaker diarization; nil omits it.
	SpeakerLabels *bool
	// VADThreshold sets the VAD confidence threshold (0.0-1.0); nil omits it.
	VADThreshold *float64
	// Domain hints the transcription domain; empty omits it.
	Domain string
	// ExtraQuery sets arbitrary additional query parameters not modeled above
	// (e.g. U3 Pro-only options); values override any param of the same name.
	ExtraQuery map[string]string
}

// Validate reports whether the configuration is usable.
func (cfg Config) Validate() error { return validate.Struct(cfg) }
