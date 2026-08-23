// Package gladia is a streaming speech-to-text service backed by Gladia's Live
// STT v2 API. A session is opened with a REST call that returns a tokenized
// WebSocket URL; audio then streams over the socket and transcripts come back as
// interim and final utterances.
package gladia

import (
	"errors"
	"time"

	"github.com/gojargo/jargo/internal/validate"
)

// errStatus is returned when the session REST call responds with a non-2xx
// status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("gladia: unexpected status")

const (
	liveURL = "https://api.gladia.io/v2/live"
	// defaults for the audio config sent at session init.
	defaultModel    = "solaria-1"
	defaultEncoding = "wav/pcm"
	defaultBitDepth = 16
	defaultChannels = 1
	// readLimit bounds a single WebSocket message.
	readLimit = 1 << 20

	// Message types the session can carry.
	msgTranscript  = "transcript"
	msgSpeechStart = "speech_start"
	msgSpeechEnd   = "speech_end"
)

// LanguageConfig configures language detection and handling.
type LanguageConfig struct {
	// Languages restricts transcription to the given language codes.
	Languages []string `json:"languages,omitempty"`
	// CodeSwitching auto-detects language changes mid-stream.
	CodeSwitching *bool `json:"code_switching,omitempty"`
}

// PreProcessingConfig configures audio pre-processing.
type PreProcessingConfig struct {
	// AudioEnhancer enhances the input audio before transcription.
	AudioEnhancer *bool `json:"audio_enhancer,omitempty"`
	// SpeechThreshold sets speech-detection sensitivity (0.0-1.0).
	SpeechThreshold *float64 `json:"speech_threshold,omitempty"`
}

// MessagesConfig filters which WebSocket messages Gladia sends. Fields left nil
// are omitted.
type MessagesConfig struct {
	ReceivePartialTranscripts       *bool `json:"receive_partial_transcripts,omitempty"`
	ReceiveFinalTranscripts         *bool `json:"receive_final_transcripts,omitempty"`
	ReceiveSpeechEvents             *bool `json:"receive_speech_events,omitempty"`
	ReceivePreProcessingEvents      *bool `json:"receive_pre_processing_events,omitempty"`
	ReceiveRealtimeProcessingEvents *bool `json:"receive_realtime_processing_events,omitempty"`
	ReceivePostProcessingEvents     *bool `json:"receive_post_processing_events,omitempty"`
	ReceiveAcknowledgments          *bool `json:"receive_acknowledgments,omitempty"`
	ReceiveErrors                   *bool `json:"receive_errors,omitempty"`
}

// Config configures the Gladia STT service. Optional fields modeled as pointers,
// slices or maps are omitted from the session init when unset.
type Config struct {
	// APIKey is the Gladia API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the session-init endpoint; empty uses the hosted endpoint.
	URL string
	// Region pins the processing region ("us-west" or "eu-west"); empty omits it.
	Region string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// Encoding is the audio encoding; empty uses "wav/pcm".
	Encoding string
	// BitDepth is the audio bit depth; 0 uses 16.
	BitDepth int
	// Channels is the channel count; 0 uses 1.
	Channels int
	// Model selects the transcription model; empty uses "solaria-1".
	Model string
	// Endpointing is the silence in seconds that marks end of speech; nil omits it.
	Endpointing *float64
	// MaximumDurationWithoutEndpointing caps utterance duration in seconds without
	// silence; nil omits it.
	MaximumDurationWithoutEndpointing *int
	// LanguageConfig configures language detection; nil omits it.
	LanguageConfig *LanguageConfig
	// PreProcessing configures audio pre-processing; nil omits it.
	PreProcessing *PreProcessingConfig
	// RealtimeProcessing passes Gladia's realtime_processing block (custom
	// vocabulary, translation, NER, sentiment, etc.) through verbatim; nil omits it.
	RealtimeProcessing map[string]any
	// MessagesConfig filters received messages; nil defaults to partial+final
	// transcripts, matching jargo's needs.
	MessagesConfig *MessagesConfig
	// EnableVAD lets Gladia's own voice-activity detection drive the turn. With
	// it set the service reports when the user starts and stops speaking, the
	// speech events are requested, and the turn strategy recommended downstream
	// becomes the external one so the aggregator defers to those reports rather
	// than running its own detection. Left unset, speech detection stays with
	// the pipeline.
	EnableVAD bool
	// InterruptOnSpeech barges in when Gladia reports the user starting to
	// speak; nil enables it. It only applies with EnableVAD set. It is passed
	// along to the user turn strategies this service recommends, which own the
	// interruption; strategies the application configures itself override the
	// recommendation and this setting with it.
	InterruptOnSpeech *bool
	// CustomMetadata attaches metadata to the session; nil omits it.
	CustomMetadata map[string]any
	// ExtraSettings sets arbitrary additional session-init fields not modeled
	// above; values override any field of the same name.
	ExtraSettings map[string]any

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.GladiaTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (cfg Config) Validate() error { return validate.Struct(cfg) }
