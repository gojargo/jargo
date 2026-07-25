// Package anthropic is a streaming LLM service backed by the Anthropic API. The
// shared LLM base brackets the response with start/end frames; this service
// streams the text deltas. It defaults to Claude Haiku for low latency and
// caches the system prompt.
package anthropic

import (
	"time"

	"github.com/gojargo/jargo/internal/validate"
)

// defaultMaxTokens keeps spoken responses short and snappy.
const defaultMaxTokens = 1024

// ThinkingConfig configures extended thinking on each request. Type is the mode:
// "enabled" (a fixed BudgetTokens, older models), "adaptive" (the model decides
// how much to think, 4.6+ models), or "disabled" (no thinking). BudgetTokens
// applies only to "enabled". Leaving Config.Thinking nil omits the parameter, so
// the model's own default applies — note adaptive thinking is on by default on
// Sonnet 5 / Opus 4.8, so set "disabled" for the lowest latency.
type ThinkingConfig struct {
	Type         string `validate:"required,oneof=enabled disabled adaptive"`
	BudgetTokens int
}

// Config configures the LLM service.
type Config struct {
	// APIKey is the Anthropic API key; empty uses the ANTHROPIC_API_KEY env var.
	APIKey string
	// BaseURL overrides the API base (e.g. a proxy or compatible gateway); empty
	// uses the SDK default.
	BaseURL string
	// Model is the model id; empty uses Claude Haiku 4.5.
	Model string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
	// Temperature, TopP and TopK are optional sampling controls. A nil value
	// leaves the API default in place; they are pointers so a deliberate zero is
	// distinguishable from "unset".
	Temperature *float64
	TopP        *float64
	TopK        *int64
	// Thinking configures extended thinking. Nil omits the parameter (model
	// default). For voice, "disabled" avoids the latency of adaptive thinking on
	// models where it is on by default.
	Thinking *ThinkingConfig `validate:"omitempty"`
	// RequestTimeout bounds a single request attempt, including the full stream;
	// 0 leaves the SDK default. Keep it generously above the expected response
	// time, since for a streaming request it caps the whole response.
	RequestTimeout time.Duration
	// MaxRetries overrides how many times a failed request is retried before the
	// stream begins (transient connection errors, 429s, 5xx); 0 leaves the SDK
	// default of two retries. Mid-stream failures are not retried.
	MaxRetries int
	// EnablePromptCaching caches the system prompt with an ephemeral cache
	// breakpoint so repeated turns reuse it; nil defaults to true (jargo caches
	// for latency). Set to false to disable caching.
	EnablePromptCaching *bool
	// Extra sets arbitrary additional top-level request-body fields not modeled
	// above (e.g. beta parameters), applied to every request.
	Extra map[string]any
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
