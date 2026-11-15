// Package interactions is a streaming LLM service backed by Google's Gemini
// Interactions API. It consumes an LLMContextFrame and emits the response as
// LLM response frames, like every other jargo LLM service.
//
// It takes the same settings as the Gemini service (package gemini), and the two
// read each other's thought signatures from the conversation, so a conversation
// can move between them.
//
// Each request carries the full conversation with store set to false, so Google
// keeps no copy of it. Sending only the new turns, with previous_interaction_id,
// would need store set to true, which retains every interaction on Google's
// servers for days; the Interactions API has no connection-local alternative
// like the OpenAI Responses WebSocket mode.
//
// The Vertex AI variant is vertex.NewInteractionsLLM.
package interactions

import (
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/google/gemini"
)

const (
	// apiURL is the Gemini API's Interactions endpoint.
	apiURL = "https://generativelanguage.googleapis.com/v1beta/interactions"
	// defaultModel is the model used when none is configured.
	defaultModel = "gemini-3.6-flash"
	// defaultMaxTokens caps a response when no cap is configured.
	defaultMaxTokens = 4096
	// defaultStreamIdleTimeout bounds the gap between two events of a streamed
	// response.
	defaultStreamIdleTimeout = 20 * time.Second
	// defaultRetryTimeout is how long the first event is waited for before a
	// stalled request is re-issued, when that is asked for.
	defaultRetryTimeout = 5 * time.Second
)

// ThinkingConfig controls the model's internal reasoning. The Interactions API
// controls thinking by level only: Level and IncludeThoughts apply, and a Budget
// is ignored with a warning.
type ThinkingConfig = gemini.ThinkingConfig

// SafetySetting is one content-safety filter, in the Gemini service's form. It
// is sent in the Interactions API's: HARM_CATEGORY_HARASSMENT becomes
// harassment, BLOCK_ONLY_HIGH becomes block_only_high, and so on.
type SafetySetting = gemini.SafetySetting

// Config configures the Interactions LLM service. It holds the same settings as
// the Gemini service's, so a service's settings carry over between the two. The
// Interactions API has no top-k, frequency penalty or presence penalty, and
// controls thinking by level only; a setting it has no field for is ignored
// with a warning.
type Config struct {
	// APIKey is the Gemini API key. Required.
	APIKey string `validate:"required"`
	// Model is the model id; empty uses a low-latency flash default.
	Model string
	// MaxTokens caps the response length; 0 uses 4096.
	MaxTokens int
	// Temperature is the sampling temperature (0.0 to 2.0); nil omits it.
	Temperature *float64
	// TopP is the nucleus-sampling parameter (0.0 to 1.0); nil omits it.
	TopP *float64
	// TopK is accepted for parity with the Gemini service, and ignored with a
	// warning: the Interactions API has no top-k.
	TopK *int
	// SafetySettings are the content-safety filters, one per harm category.
	// Empty sends none, leaving every category at the API's default.
	SafetySettings []SafetySetting `validate:"omitempty,dive"`
	// Thinking controls the model's internal reasoning: Level and
	// IncludeThoughts apply, and a Budget is ignored. If it is nil, the service
	// sends the lowest thinking level Google documents for the model, for the
	// fastest responses, and logs what it sent. Models it does not list, and
	// every model when the use_provider_thinking_default setting is set, run at
	// Google's default.
	Thinking *ThinkingConfig `validate:"omitempty"`
	// Seed makes sampling repeatable; nil omits it.
	Seed *int
	// Tools are offered to the model when the conversation advertises none, in
	// Gemini's generateContent format ({"function_declarations": [...]},
	// {"google_search": {}}) or the Interactions API's ({"type": ...}).
	Tools []map[string]any
	// ToolConfig configures tool use, in Gemini's generateContent format
	// ({"function_calling_config": {"mode": "ANY", "allowed_function_names":
	// [...]}}). It is sent as the Interactions API's tool_choice.
	ToolConfig map[string]any
	// StreamIdleTimeout bounds how long to wait for the next event of a
	// streamed response before giving up on it. It bounds the wait when the API
	// accepts a request and then stops producing without closing the stream.
	// Nil uses 20 seconds and a zero value waits indefinitely.
	StreamIdleTimeout *time.Duration
	// RetryOnTimeout re-issues the request once when its first event does not
	// arrive within RetryTimeout.
	RetryOnTimeout bool
	// RetryTimeout is how long the first event is waited for before the request
	// is re-issued; 0 uses 5 seconds. It applies only with RetryOnTimeout.
	RetryTimeout time.Duration `validate:"omitempty,min=0"`
	// Extra sets additional request fields not modeled above, applied to every
	// request. A generation_config entry holding an object is merged into the
	// generation configuration field by field; any other entry replaces the
	// request field of its name.
	Extra map[string]any
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// streamIdleTimeout is how long a gap between events is tolerated, or zero for
// no bound at all.
func (c Config) streamIdleTimeout() time.Duration {
	if c.StreamIdleTimeout == nil {
		return defaultStreamIdleTimeout
	}
	return *c.StreamIdleTimeout
}

// retryTimeout is how long the first event is waited for before the request is
// re-issued.
func (c Config) retryTimeout() time.Duration {
	if c.RetryTimeout == 0 {
		return defaultRetryTimeout
	}
	return c.RetryTimeout
}
