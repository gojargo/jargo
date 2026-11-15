// Kokoro text-to-speech through a Kokoro-FastAPI server. Kokoro-FastAPI
// exposes an OpenAI-compatible /v1/audio/speech endpoint that can return raw
// PCM, so this service is a thin wrapper over the OpenAI-compatible TTS
// service pointed at the server.

package kokoro

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/tts"
)

const (
	// defaultHTTPModel is the model id Kokoro-FastAPI serves Kokoro under.
	defaultHTTPModel = "kokoro"
	// defaultHTTPVoice is a commonly available Kokoro voice.
	defaultHTTPVoice = "af_sky"
)

// HTTPConfig configures the Kokoro-FastAPI TTS service.
type HTTPConfig struct {
	// BaseURL is the OpenAI-compatible API base of the server, including any
	// version segment (e.g. http://localhost:8880/v1). Required.
	BaseURL string `validate:"required,url"`
	// APIKey is an optional bearer token; most local servers need none.
	APIKey string
	// Model overrides the model field; empty uses the Kokoro default.
	Model string
	// Voice is the voice name; empty uses a default voice.
	Voice string
	// Speed controls the speaking rate (0.25 to 4.0); nil omits it.
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c HTTPConfig) Validate() error { return validate.Struct(c) }

// NewHTTPTTS builds a TTS service speaking through a Kokoro-FastAPI server.
// Kokoro-FastAPI returns 24 kHz PCM, which matches the OpenAI "pcm" response
// format this wrapper requests. NewTTS runs the model in process instead.
func NewHTTPTTS(cfg HTTPConfig) *tts.Base {
	return chat.NewCompatTTS("KokoroHTTPTTS", cfg.BaseURL, defaultHTTPModel, defaultHTTPVoice, chat.TTSConfig{
		APIKey:  cfg.APIKey,
		BaseURL: cfg.BaseURL,
		Model:   cfg.Model,
		Voice:   cfg.Voice,
		Speed:   cfg.Speed,
	})
}
