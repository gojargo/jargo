// Package kokoro is a text-to-speech provider for a local Kokoro-FastAPI server.
// Kokoro-FastAPI exposes an OpenAI-compatible /v1/audio/speech endpoint that can
// return raw PCM, so this provider is a thin wrapper over the OpenAI-compatible
// TTS service pointed at a local URL.
package kokoro

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	// defaultModel is the Kokoro model id.
	defaultModel = "kokoro"
	// defaultVoice is a commonly available Kokoro voice.
	defaultVoice = "af_sky"
)

// Config configures the Kokoro TTS provider.
type Config struct {
	// BaseURL is the OpenAI-compatible API base of the local server, including
	// any version segment (e.g. http://localhost:8880/v1). Required.
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
func (c Config) Validate() error { return validate.Struct(c) }
