// Package gemini is a streaming LLM service backed by Google's Gemini API
// (generateContent with SSE). It consumes an LLMContextFrame and emits the
// response as LLM response frames, like every other jargo LLM service.
package gemini

import (
	"errors"
	"regexp"
	"strings"
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

// The thinkingConfig keys and the levels the thinking defaults are made of.
const (
	keyThinkingBudget = "thinkingBudget"
	keyThinkingLevel  = "thinkingLevel"
	levelMinimal      = "minimal"
	levelLow          = "low"
)

// thinkingDefaults is the lowest thinking setting Google documents for each
// model, sent when the caller does not configure Thinking, keyed by model name.
// The 2.5 models take a budget, where 0 turns thinking off and 2.5 Pro's minimum
// is 128; later models take a level. 3.7 Flash, 3.8 Flash, 3.1 Pro and 4 Argon
// reject "minimal", and the Flash-Lite models already default to it. Aliases
// such as "gemini-flash-latest" are left out, since the model they name changes.
//
//nolint:gochecknoglobals // a lookup table
var thinkingDefaults = map[string]map[string]any{
	"gemini-2.5-flash":                   {keyThinkingBudget: 0},
	"gemini-2.5-flash-lite":              {keyThinkingBudget: 0},
	"gemini-2.5-pro":                     {keyThinkingBudget: 128},
	"gemini-3-flash-preview":             {keyThinkingLevel: levelMinimal},
	"gemini-3.1-flash-lite":              {keyThinkingLevel: levelMinimal},
	"gemini-3.1-pro-preview":             {keyThinkingLevel: levelLow},
	"gemini-3.1-pro-preview-customtools": {keyThinkingLevel: levelLow},
	"gemini-3.5-flash":                   {keyThinkingLevel: levelMinimal},
	"gemini-3.5-flash-lite":              {keyThinkingLevel: levelMinimal},
	"gemini-3.6-flash":                   {keyThinkingLevel: levelMinimal},
	"gemini-3.7-flash":                   {keyThinkingLevel: levelLow},
	"gemini-3.8-flash":                   {keyThinkingLevel: levelLow},
	"gemini-4-argon":                     {keyThinkingLevel: levelLow},
	"gemma-4-26b-a4b-it":                 {keyThinkingLevel: levelMinimal},
	"gemma-4-31b-it":                     {keyThinkingLevel: levelMinimal},
}

// modelsSupportingThinkingLevel are the models that take their thinking
// configuration from a level, keyed by model name prefix. Whether one of these
// honors a budget set alongside it varies by model and by backend, so such a
// budget cannot be relied on.
//
//nolint:gochecknoglobals // a lookup table
var modelsSupportingThinkingLevel = []string{"gemini-3", "gemini-4", "gemma-4"}

// versionSuffix matches the version a model id can carry, as in
// "gemini-2.5-flash-001".
//
//nolint:gochecknoglobals // compiled once, read-only
var versionSuffix = regexp.MustCompile(`-\d{3}$`)

// modelName is the name a model id refers to, for looking the model up in
// thinkingDefaults and modelsSupportingThinkingLevel. Ids can name a model by
// resource path ("models/gemini-3.6-flash",
// "publishers/google/models/gemini-3.6-flash") or by version
// ("gemini-2.5-flash-001").
func modelName(model string) string {
	name := strings.ToLower(model)
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	return versionSuffix.ReplaceAllString(name, "")
}

// thinkingModel matches the names of the Gemini and Gemma text models that can
// think: Gemini 2.5 and later, Gemma 4, and aliases such as
// "gemini-flash-latest".
//
//nolint:gochecknoglobals // compiled once, read-only
var thinkingModel = regexp.MustCompile(`^(?:gemini-(?:2\.5|[3-9])|gemma-4|gemini-[a-z-]+-latest$)`)

// canThink reports whether a model name belongs to a Gemini or Gemma text model
// that can think. Image, speech and embedding models are excluded.
func canThink(name string) bool {
	for _, kind := range []string{"image", "tts", "audio", "live", "embedding"} {
		if strings.Contains(name, kind) {
			return false
		}
	}
	return thinkingModel.MatchString(name)
}

// ThinkingDefault is the lowest thinking setting Google documents for a model,
// or nil when it lists none, and whether the model can think, which decides
// whether a model with no default is warned about. model may be in any form
// Google names models by: a resource path or a versioned id.
//
// levelsOnly is for an API that takes thinking as a level only. The 2.5 models
// take a budget, so such an API has no default to send them and no setting to
// suggest.
func ThinkingDefault(model string, levelsOnly bool) (*ThinkingConfig, bool) {
	name := modelName(model)
	if levelsOnly && strings.HasPrefix(name, "gemini-2.5") {
		return nil, false
	}
	d, ok := thinkingDefaults[name]
	if !ok {
		return nil, canThink(name)
	}
	thinking := &ThinkingConfig{}
	if level, ok := d[keyThinkingLevel].(string); ok {
		thinking.Level = level
	}
	if budget, ok := d[keyThinkingBudget].(int); ok {
		thinking.Budget = &budget
	}
	return thinking, canThink(name)
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
	// "high" only, and so do Gemini 3.7 Flash and Gemini 3.8 Flash), and Google
	// may add more, so it is not checked here.
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
	// Thinking controls the model's internal reasoning. If it is nil, the
	// service sends the lowest thinking setting Google documents for the model,
	// for the fastest responses, and logs what it sent. Models it does not list
	// run at Google's default.
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
