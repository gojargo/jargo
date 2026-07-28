// Package gemini is a streaming LLM service backed by Google's Gemini API
// (generateContent with SSE). It consumes an LLMContextFrame and emits the
// response as LLM response frames, like every other jargo LLM service.
package gemini

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

// errStatus is returned when the API responds with a non-200 status. It is shared
// by the LLM, STT and TTS services.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("google: unexpected status")

// defaultLangCode is the default BCP-47 language code for the STT and TTS
// services.
const defaultLangCode = "en-US"

const (
	apiBase          = "https://generativelanguage.googleapis.com/v1beta/models"
	defaultModel     = "gemini-2.5-flash"
	defaultMaxTokens = 1024
	// Gemini content/part map keys, hoisted to avoid repeated string literals.
	keyRole  = "role"
	keyParts = "parts"
	keyName  = "name"
	keyText  = "text"
)

// Config configures the Gemini LLM service. The sampling controls are pointers
// so a deliberate zero is distinguishable from "unset"; a nil value is omitted
// from the request, leaving the API default.
type Config struct {
	// APIKey is the Gemini API key. Required.
	APIKey string `validate:"required"`
	// Model is the model id; empty uses a low-latency flash default.
	Model string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
	// Temperature is the sampling temperature (0.0 to 2.0); nil omits it.
	Temperature *float64
	// TopP is the nucleus-sampling parameter (0.0 to 1.0); nil omits it.
	TopP *float64
	// TopK is the top-k sampling parameter; nil omits it.
	TopK *int
	// Extra sets arbitrary additional generationConfig fields not modeled above,
	// applied to every request.
	Extra map[string]any
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
