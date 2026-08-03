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

// SafetySetting is one content-safety filter: a category of harm and the
// threshold at which the model blocks content for it. A category left
// unspecified keeps the API's own default.
type SafetySetting struct {
	// Category is the harm category the threshold applies to. Required.
	Category string `json:"category" validate:"required,oneof=HARM_CATEGORY_HARASSMENT HARM_CATEGORY_HATE_SPEECH HARM_CATEGORY_SEXUALLY_EXPLICIT HARM_CATEGORY_DANGEROUS_CONTENT HARM_CATEGORY_CIVIC_INTEGRITY"` //nolint:lll // one line per accepted value would not read better
	// Threshold is how much of the category to block. Required.
	Threshold string `json:"threshold" validate:"required,oneof=BLOCK_LOW_AND_ABOVE BLOCK_MEDIUM_AND_ABOVE BLOCK_ONLY_HIGH BLOCK_NONE OFF"` //nolint:lll // one line per accepted value would not read better
	// Method selects whether the threshold is read against the probability that
	// the content is harmful or the severity of the harm; empty leaves the API
	// default.
	Method string `json:"method,omitempty" validate:"omitempty,oneof=SEVERITY PROBABILITY"`
}

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
	// SafetySettings are the content-safety filters, one per harm category.
	// Empty sends none, leaving every category at the API's default.
	SafetySettings []SafetySetting `validate:"omitempty,dive"`
	// Extra sets arbitrary additional generationConfig fields not modeled above,
	// applied to every request.
	Extra map[string]any
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
