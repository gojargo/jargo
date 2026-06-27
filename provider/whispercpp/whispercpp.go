// Package whispercpp is a speech-to-text provider for a local whisper.cpp
// server. whisper.cpp's HTTP server exposes an OpenAI-compatible
// /v1/audio/transcriptions endpoint, so this provider is a thin wrapper over the
// OpenAI-compatible transcription service pointed at a local URL.
//
// It is segmented: a turn detector upstream delimits each utterance, which is
// uploaded and transcribed in one request.
package whispercpp

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/openai"
	"github.com/gojargo/jargo/service/stt"
)

// defaultModel is sent as the transcription model. whisper.cpp uses the model it
// was launched with and ignores this field, but the API expects it present.
const defaultModel = "whisper-1"

// Config configures the whisper.cpp STT provider.
type Config struct {
	// BaseURL is the OpenAI-compatible API base of the local server, including
	// any version segment (e.g. http://localhost:8080/v1). Required.
	BaseURL string `validate:"required,url"`
	// APIKey is an optional bearer token; most local servers need none.
	APIKey string
	// Model overrides the model field sent with each request.
	Model string
	// Language of the audio, sent as an ISO code; the zero value omits it
	// (auto-detect).
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewSTT builds a whisper.cpp transcription service.
func NewSTT(cfg Config) *stt.SegmentService {
	return openai.NewCompatSTT("WhisperCppSTT", cfg.BaseURL, defaultModel, openai.STTConfig{
		APIKey:     cfg.APIKey,
		BaseURL:    cfg.BaseURL,
		Model:      cfg.Model,
		Language:   cfg.Language,
		SampleRate: cfg.SampleRate,
	})
}
