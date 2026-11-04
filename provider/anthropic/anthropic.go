// Package anthropic is a streaming LLM service backed by the Anthropic API. The
// shared LLM base brackets the response with start/end frames; this service
// streams the text deltas. It defaults to Claude Haiku for low latency and
// caches the system prompt.
package anthropic

import (
	"regexp"
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
// turns thinking off on the models that allow it; "between_tools" is how Claude
// Sonnet 5.5, which rejects "disabled", runs without up-front thinking, and no
// other model accepts it. BudgetTokens applies only to "enabled".
type ThinkingConfig struct {
	Type         string `validate:"required,oneof=enabled disabled adaptive between_tools"`
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
	// Thinking configures thinking. If neither this nor Effort is set, the
	// service sends a low-latency thinking default for the model (its lowest
	// documented setting, unless that makes the model speak its reasoning) and
	// logs what it sent. Models it does
	// not list run at Anthropic's default.
	Thinking *ThinkingConfig `validate:"omitempty"`
	// Effort is how much the model spends on its response, thinking included:
	// "low", "medium", "high", "xhigh" or "max". It is sent as
	// output_config.effort. Empty uses the model's default. Lower effort is
	// Anthropic's recommended way to reduce thinking, and the only one on models
	// whose thinking cannot be turned off. It is a plain string rather than a
	// closed set, so a level the API gains can be asked for without a release.
	Effort string
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
	// SystemPromptCacheTTL is how long the system prompt's cache entry lives
	// when prompt caching is enabled: "5m" or "1h". Empty uses Anthropic's
	// default of 5 minutes. "1h" keeps a system prompt shared by many
	// conversations cached across the gaps between them, at twice the base
	// input price per cache write instead of 1.25 times. Anthropic caches
	// nothing when the tools and system prompt together fall below the model's
	// minimum cacheable prompt length.
	SystemPromptCacheTTL string `validate:"omitempty,oneof=5m 1h"`
	// Extra sets arbitrary additional top-level request-body fields not modeled
	// above (e.g. beta parameters), applied to every request.
	Extra map[string]any
	// OmitInterleavedThinkingBeta leaves the interleaved-thinking beta off every
	// request. The Anthropic API takes it on any request; a backend that serves
	// the same models through an API of its own (Amazon Bedrock) may reject a
	// beta a model does not support, so a service built for one sets this.
	OmitInterleavedThinkingBeta bool
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// The thinking types and the effort level the thinking defaults are made of.
const (
	thinkingDisabled     = "disabled"
	thinkingBetweenTools = "between_tools"
	effortLow            = "low"
)

// thinkingDefault is the request fields a model's thinking default is sent as:
// a thinking type, or an effort, on a model whose thinking cannot be turned off
// or that speaks its reasoning with thinking off.
type thinkingDefault struct {
	thinking string
	effort   string
}

// thinkingDefaults is the request fields sent when the caller sets neither
// Thinking nor Effort, keyed by model name. Each is the model's lowest
// documented setting, for the fastest responses, unless that setting makes the
// model speak its reasoning: Haiku 5.5 with thinking disabled narrates its
// decisions before tool calls, so it gets low effort. "between_tools" is Sonnet
// 5.5's lowest setting, and "disabled" is already the default on the Claude 4
// models.
//
//nolint:gochecknoglobals // a lookup table
var thinkingDefaults = map[string]thinkingDefault{
	"claude-fable-5-1":  {effort: effortLow},
	"claude-fable-5":    {effort: effortLow},
	"claude-opus-5-5":   {effort: effortLow},
	"claude-opus-5":     {thinking: thinkingDisabled},
	"claude-sonnet-5-5": {thinking: thinkingBetweenTools},
	"claude-sonnet-5":   {thinking: thinkingDisabled},
	"claude-haiku-5-5":  {effort: effortLow},
	"claude-opus-4-8":   {thinking: thinkingDisabled},
	"claude-opus-4-7":   {thinking: thinkingDisabled},
	"claude-opus-4-6":   {thinking: thinkingDisabled},
	"claude-sonnet-4-6": {thinking: thinkingDisabled},
	"claude-opus-4-5":   {thinking: thinkingDisabled},
	"claude-sonnet-4-5": {thinking: thinkingDisabled},
	"claude-haiku-4-5":  {thinking: thinkingDisabled},
}

// Suffixes a model id carries beyond the model's name.
//
//nolint:gochecknoglobals // compiled once, read-only
var (
	bedrockVersionSuffix = regexp.MustCompile(`-v\d+:\d+$`)
	snapshotDateSuffix   = regexp.MustCompile(`-\d{8}$`)
)

// modelName is the name a model id refers to, the key it is looked up by in
// thinkingDefaults.
//
// Bedrock and Vertex ids wrap the name, and a dated snapshot gets its model's
// entry: "us.anthropic.claude-sonnet-5-20260630-v1:0", "claude-sonnet-5@20260630"
// and "claude-sonnet-5-20260630" all name "claude-sonnet-5".
func modelName(model string) string {
	name := strings.ToLower(model)
	if i := strings.LastIndex(name, "."); i >= 0 {
		name = name[i+1:]
	}
	name, _, _ = strings.Cut(name, "@")
	name = bedrockVersionSuffix.ReplaceAllString(name, "")
	return snapshotDateSuffix.ReplaceAllString(name, "")
}
