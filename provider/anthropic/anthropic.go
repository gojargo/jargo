// Package anthropic is a streaming LLM service backed by the Anthropic API. The
// shared LLM base brackets the response with start/end frames; this service
// streams the text deltas. It defaults to Claude Haiku for low latency and
// caches the system prompt.
package anthropic

import (
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gojargo/jargo/internal/validate"
)

// defaultMaxTokens keeps spoken responses short and snappy.
const defaultMaxTokens = 1024

// ThinkingConfig configures thinking on each request. Type is the mode:
// "adaptive" lets the model decide when and how deeply to think and is the one
// to prefer; "enabled" is the legacy manual mode sized by BudgetTokens, which
// Claude 4.7 and later reject and Claude 4.5 and earlier accept only; "disabled"
// turns thinking off. BudgetTokens applies only to "enabled".
type ThinkingConfig struct {
	Type         string `validate:"required,oneof=enabled disabled adaptive"`
	BudgetTokens int
	// Display is how thinking text comes back: "summarized" for readable
	// thinking, which is what a thought frame carries, or "omitted" for thinking
	// blocks whose text is empty. Claude 4.7 and later default to "omitted", so
	// ask for "summarized" there to keep those frames carrying text. Empty leaves
	// it unset, and it is not allowed when Type is "disabled".
	Display string `validate:"omitempty,oneof=summarized omitted"`
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
	// Thinking configures thinking. Nil turns thinking off on Sonnet 5 and later,
	// which otherwise decide per request whether to think and can spend seconds
	// on it before the first answer token; Opus and Fable are left at Anthropic's
	// own default, since choosing one of those is a decision to reason.
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

// sonnetThinksByDefaultFrom is the first Sonnet generation with adaptive
// thinking on when a request omits the thinking parameter. Earlier Sonnets, and
// every Haiku, have thinking off unless it is asked for.
const sonnetThinksByDefaultFrom = 5

// sonnetGeneration is the generation of a Sonnet model id, and -1 for any other
// model.
//
// Searched rather than anchored, because the service also takes Bedrock and
// Vertex ids, which prefix the name ("anthropic.claude-sonnet-5"). Pre-4 ids
// such as "claude-3-5-sonnet-20241022" put the generation before the name and do
// not match; they do not think either.
// A generation of more than two digits is not one: it is part of some other
// identifier that happens to follow the name, and is left unmatched.
func sonnetGeneration(model string) int {
	m := sonnetGenerationRE.FindStringSubmatch(strings.ToLower(model))
	if m == nil || len(m[1]) > 2 {
		return -1
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		return -1
	}
	return n
}

//nolint:gochecknoglobals // compiled once, read-only
var sonnetGenerationRE = regexp.MustCompile(`sonnet-(\d+)`)

// sonnetThinksByDefault reports whether the model thinks unless told not to.
func sonnetThinksByDefault(model string) bool {
	g := sonnetGeneration(model)
	return g >= sonnetThinksByDefaultFrom
}
