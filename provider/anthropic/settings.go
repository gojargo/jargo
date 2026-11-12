package anthropic

import (
	"maps"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/gojargo/jargo/service/settings"
)

// LLMSettings is the part of the Anthropic configuration that can change while
// the pipeline runs, through an LLMUpdateSettingsFrame: the general LLM
// settings (the model, the token bound, the sampling controls, a system
// instruction), prompt caching, thinking and the effort. Every request is
// built from them, so an update reaches the next request.
//
// A setting the service does not model travels in Extra and is set on the
// request body, as Config.Extra is.
type LLMSettings struct {
	settings.LLM
	// EnablePromptCaching places the cache breakpoints on the prompt.
	EnablePromptCaching settings.Opt[bool] `settings:"enable_prompt_caching"`
	// SystemPromptCacheTTL is how long the system prompt's cache entry lives:
	// "5m" or "1h". Unset leaves Anthropic's default.
	SystemPromptCacheTTL settings.Opt[string] `settings:"system_prompt_cache_ttl"`
	// Thinking is the extended-thinking configuration.
	Thinking settings.Opt[ThinkingConfig] `settings:"thinking"`
	// Effort is how much the model spends on its response, sent as
	// output_config.effort.
	Effort settings.Opt[string] `settings:"effort"`
}

// newLLMSettings is the starting state, taken from what the service was built
// with, the service's defaults filled in.
func newLLMSettings(cfg Config) *LLMSettings {
	s := &LLMSettings{}
	s.Model = settings.Set(sdk.ModelClaudeHaiku4_5)
	if cfg.Model != "" {
		s.Model = settings.Set(cfg.Model)
	}
	s.MaxTokens = settings.Set(defaultMaxTokens)
	if cfg.MaxTokens > 0 {
		s.MaxTokens = settings.Set(cfg.MaxTokens)
	}
	if cfg.Temperature != nil {
		s.Temperature = settings.Set(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		s.TopP = settings.Set(*cfg.TopP)
	}
	if cfg.TopK != nil {
		s.TopK = settings.Set(int(*cfg.TopK))
	}
	s.EnablePromptCaching = settings.Set(true)
	if cfg.EnablePromptCaching != nil {
		s.EnablePromptCaching = settings.Set(*cfg.EnablePromptCaching)
	}
	if cfg.SystemPromptCacheTTL != "" {
		s.SystemPromptCacheTTL = settings.Set(cfg.SystemPromptCacheTTL)
	}
	if cfg.Thinking != nil {
		s.Thinking = settings.Set(*cfg.Thinking)
	}
	if cfg.Effort != "" {
		s.Effort = settings.Set(cfg.Effort)
	}
	if len(cfg.Extra) > 0 {
		s.Extra = maps.Clone(cfg.Extra)
	}
	return s
}

// Settings returns the service's runtime-updatable settings, implementing
// llm.SettingsHolder.
func (s *Service) Settings() any { return s.live }

// requestSettings is what one request is built from: the settings as they
// stand, read whole and put in the SDK's terms.
type requestSettings struct {
	model       sdk.Model
	maxTokens   int64
	temperature param.Opt[float64]
	topP        param.Opt[float64]
	topK        param.Opt[int64]
	// thinking is the extended-thinking config sent; thinkingSet reports
	// whether one is configured (an unset union means "omit").
	thinking             sdk.ThinkingConfigParamUnion
	thinkingSet          bool
	effort               string
	cachePrompt          bool
	systemPromptCacheTTL string
	extra                map[string]any
}

// requestSettings reads the settings as they stand.
func (s *Service) requestSettings() requestSettings {
	var rs requestSettings
	s.ReadSettings(func() {
		l := s.live
		rs.model = l.Model.Or("")
		rs.maxTokens = int64(l.MaxTokens.Or(defaultMaxTokens))
		if v, ok := l.Temperature.Value(); ok {
			rs.temperature = param.NewOpt(v)
		}
		if v, ok := l.TopP.Value(); ok {
			rs.topP = param.NewOpt(v)
		}
		if v, ok := l.TopK.Value(); ok {
			rs.topK = param.NewOpt(int64(v))
		}
		if t, ok := l.Thinking.Value(); ok {
			rs.thinking, rs.thinkingSet = thinkingParam(t)
		}
		rs.effort = l.Effort.Or("")
		rs.cachePrompt = l.EnablePromptCaching.Or(true)
		rs.systemPromptCacheTTL = l.SystemPromptCacheTTL.Or("")
		rs.extra = maps.Clone(l.Extra)
	})
	return rs
}

// thinkingParam puts a thinking configuration in the SDK's terms, reporting
// false for a type it does not know, which leaves thinking unset.
func thinkingParam(t ThinkingConfig) (sdk.ThinkingConfigParamUnion, bool) {
	switch t.Type {
	case thinkingDisabled:
		return sdk.ThinkingConfigParamUnion{OfDisabled: &sdk.ThinkingConfigDisabledParam{}}, true
	case thinkingBetweenTools:
		return sdk.ThinkingConfigParamUnion{OfBetweenTools: &sdk.ThinkingConfigBetweenToolsParam{}}, true
	case "adaptive":
		return sdk.ThinkingConfigParamUnion{
			OfAdaptive: &sdk.ThinkingConfigAdaptiveParam{Display: sdk.ThinkingConfigAdaptiveDisplay(t.Display)},
		}, true
	case "enabled":
		enabled := sdk.ThinkingConfigParamOfEnabled(int64(t.BudgetTokens))
		enabled.OfEnabled.Display = sdk.ThinkingConfigEnabledDisplay(t.Display)
		return enabled, true
	default:
		return sdk.ThinkingConfigParamUnion{}, false
	}
}
