// Package gemini is a streaming LLM service backed by Google's Gemini API
// (generateContent with SSE). It consumes an LLMContextFrame and emits the
// response as LLM response frames, like every other jargo LLM service.
package gemini

import (
	"errors"
	"time"

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
	defaultModel     = "gemini-3.6-flash"
	defaultMaxTokens = 1024
	// defaultStreamIdleTimeout bounds the gap between two chunks of a streamed
	// response. The API client applies no timeout of its own, so a stream that
	// stops producing without closing would hold the turn open for good.
	defaultStreamIdleTimeout = 20 * time.Second
	// defaultRetryTimeout is how long the first chunk is waited for before a
	// stalled request is re-issued, when that is asked for.
	defaultRetryTimeout = 5 * time.Second
	// Gemini content/part map keys, hoisted to avoid repeated string literals.
	keyRole  = "role"
	keyParts = "parts"
	keyName  = "name"
	keyText  = "text"
	keyID    = "id"
	// keyThinkingConfig is the generationConfig key the thinking controls live
	// under.
	keyThinkingConfig = "thinkingConfig"
)

// lowestThinkingLevels is the least each model will think, keyed by model
// prefix. A model that is not listed accepts "minimal", the fastest setting.
//
//nolint:gochecknoglobals // a lookup table
var lowestThinkingLevels = map[string]string{
	"gemini-3.7-flash": "low",
}

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

// ThinkingConfig controls the model's internal reasoning before it answers. Set
// one of Level or Budget, never both: the Gemini 3 models read the level and the
// 2.5 series reads the budget.
type ThinkingConfig struct {
	// Level is how much a Gemini 3 model thinks: "minimal", "low", "medium" or
	// "high". Which of them a model accepts varies (Gemini 3 Pro takes "low" and
	// "high" only), and Google may add more, so it is not checked here.
	Level string
	// Budget is the token budget a Gemini 2.5 model may spend on thinking: -1
	// lets the model decide, 0 turns thinking off, and a positive value caps it.
	// Nil leaves the model's own default.
	Budget *int
	// IncludeThoughts asks for a summary of what the model thought. Today's
	// models leave it out unless asked.
	IncludeThoughts bool
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
	// Thinking controls the model's internal reasoning; nil applies the
	// low-latency default for the model, which turns thinking off on the flash
	// models. A voice pipeline wants an answer sooner rather than a considered
	// one, and a model that thinks emits nothing while it does.
	Thinking *ThinkingConfig `validate:"omitempty"`
	// Seed makes sampling repeatable: the same seed and the same request usually
	// produce the same answer, though Gemini treats it as best effort. Nil omits
	// it.
	Seed *int
	// StreamIdleTimeout bounds the gap between two chunks of a streamed
	// response; nil uses 20 seconds and a zero value waits indefinitely.
	// Reaching it ends the response with whatever text arrived and reports a
	// completion timeout. It bounds the gap rather than the whole response, so a
	// slow but healthy stream is never cut short. Raise it for a model
	// configured to think at length, since thinking emits no chunks.
	StreamIdleTimeout *time.Duration
	// RetryOnTimeout re-issues a request once when its first chunk does not
	// arrive within RetryTimeout, so a request the API accepts and then never
	// answers costs a few seconds rather than the whole idle timeout. Only the
	// first chunk is retried: re-issuing after that would duplicate the answer.
	// The window covers the whole round trip, including any thinking the model
	// does before it emits anything, so leave it off for a model that thinks at
	// length.
	RetryOnTimeout bool
	// RetryTimeout is how long the first chunk is waited for before the request
	// is re-issued; 0 uses 5 seconds. It applies only with RetryOnTimeout.
	RetryTimeout time.Duration `validate:"omitempty,min=0"`
	// Extra sets arbitrary additional generationConfig fields not modeled above,
	// applied to every request.
	Extra map[string]any
}

// streamIdleTimeout is how long a gap between chunks is tolerated, or zero for
// no bound at all.
func (c Config) streamIdleTimeout() time.Duration {
	if c.StreamIdleTimeout == nil {
		return defaultStreamIdleTimeout
	}
	return *c.StreamIdleTimeout
}

// retryTimeout is how long the first chunk is waited for before the request is
// re-issued.
func (c Config) retryTimeout() time.Duration {
	if c.RetryTimeout == 0 {
		return defaultRetryTimeout
	}
	return c.RetryTimeout
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
