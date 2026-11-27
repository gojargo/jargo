package smallest

import (
	"errors"
	"maps"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

const (
	// defaultLLMBaseURL is the Waves API, which serves Smallest AI's LLM on an
	// OpenAI-compatible chat completions endpoint.
	defaultLLMBaseURL = "https://api.smallest.ai/waves/v1"
	// defaultLLMModel is Electron, Smallest AI's LLM.
	defaultLLMModel = "electron"
)

// errMissingLLMAPIKey is returned for an LLM configuration with no API key.
var errMissingLLMAPIKey = errors.New("smallest: the LLM requires an api_key")

// LLMConfig configures a Smallest AI LLM service: everything the
// OpenAI-compatible endpoint takes, plus the prompt cache key Electron adds to
// it. The model defaults to Electron.
type LLMConfig struct {
	chat.LLMConfig
	// PromptCacheKey groups the requests of one conversation, so each turn is
	// served where that conversation is already cached. Use a fresh value per
	// conversation and never share one between conversations. Empty sends none.
	PromptCacheKey string
}

// Validate reports whether the configuration is usable: it needs an API key.
func (c LLMConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.APIKey == "" {
		return errMissingLLMAPIKey
	}
	return nil
}

// LLMSettings is the Smallest AI LLM configuration that can change while the
// pipeline runs, through an LLMUpdateSettingsFrame: the OpenAI-compatible
// settings, plus the prompt cache key.
type LLMSettings struct {
	chat.LLMSettings
	// PromptCacheKey groups the requests of one conversation; unset sends none.
	PromptCacheKey settings.Opt[string] `settings:"prompt_cache_key"`
}

// LLMService is the Smallest AI LLM service NewLLM builds.
type LLMService struct {
	*chat.LLMService
}

// NewLLM builds a Smallest AI LLM service over the Waves API's
// OpenAI-compatible endpoint, defaulting to Electron.
//
// Electron does not support the developer role, so a developer message is sent
// as a user message. Smallest's source identification header is always sent,
// over the caller's own headers.
func NewLLM(cfg LLMConfig) *LLMService {
	s := &LLMService{}
	s.LLMService = buildLLM(cfg, llm.WithSelf(s))
	return s
}

// buildLLM builds the service NewLLM returns, registering the
// service opts names as the processor the pipeline runs.
func buildLLM(cfg LLMConfig, opts ...llm.Option) *chat.LLMService {
	compat := cfg.LLMConfig
	headers := maps.Clone(compat.DefaultHeaders)
	if headers == nil {
		headers = map[string]string{}
	}
	headers["X-Source"] = "jargo"
	compat.DefaultHeaders = headers

	live := &LLMSettings{PromptCacheKey: settings.Cleared[string]()}
	if cfg.PromptCacheKey != "" {
		live.PromptCacheKey = settings.Set(cfg.PromptCacheKey)
	}
	return chat.NewCompatLLM(chat.Compat{
		Name:            "SmallestLLM",
		BaseURL:         defaultLLMBaseURL,
		DefaultModel:    defaultLLMModel,
		NoDeveloperRole: true,
		Settings:        live,
		Request:         live.editRequest,
	}, compat, opts...)
}

// editRequest adds Electron's prompt cache key to the request when one is set.
// The caller's Extra fields are merged over it, and win.
func (s *LLMSettings) editRequest(body map[string]any) {
	if key, ok := s.PromptCacheKey.Value(); ok {
		body["prompt_cache_key"] = key
	}
}
