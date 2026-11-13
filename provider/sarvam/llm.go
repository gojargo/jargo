package sarvam

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/settings"
)

const (
	// llmAPIBase is Sarvam's API host. The version the LLM is served under
	// depends on the model, so it is added per model.
	llmAPIBase      = "https://api.sarvam.ai"
	defaultLLMModel = modelSarvam105b
	// partTypeImageURL is the content part an image travels in.
	partTypeImageURL = "image_url"
)

// The LLM models Sarvam serves.
const (
	modelDeepSeekV41Flash        = "deepseekv4.1-flash"
	modelGemma4                  = "gemma4"
	modelGLM53                   = "glm5.3"
	modelSarvam105b              = "sarvam-105b"
	modelSarvam105bConversations = "sarvam-105b-conversations"
)

// The models Sarvam serves, and what each can do.
//
//nolint:gochecknoglobals // fixed lookup tables
var (
	supportedLLMModels = []string{
		modelDeepSeekV41Flash, modelGemma4, modelGLM53, modelSarvam105b, modelSarvam105bConversations,
	}
	visionModels        = []string{modelGemma4}
	reasoningModels     = []string{modelDeepSeekV41Flash, modelGemma4, modelGLM53, modelSarvam105b}
	wikiGroundingModels = []string{modelGemma4, modelSarvam105b}
	// v1Models are served under /v1; every other model is served under /v2.
	v1Models = []string{modelSarvam105bConversations}
)

var (
	errMissingAPIKey          = errors.New("sarvam: the LLM needs an APIKey")
	errUnsupportedLLMModel    = errors.New("sarvam: unsupported LLM model")
	errToolChoiceWithoutTools = errors.New("sarvam requires non-empty `tools` when `tool_choice` is provided")
	errNoImageInput           = errors.New("does not support image input; use a vision-capable model instead")
	errNoReasoningEffort      = errors.New("does not support reasoning_effort; use a reasoning-capable model instead")
	errNoWikiGrounding        = errors.New("does not support wiki_grounding; use a model that supports it instead")
)

// LLMConfig configures a Sarvam LLM service: everything the OpenAI-compatible
// endpoint takes, plus the controls Sarvam adds to it.
//
// Sarvam does not take some OpenAI parameters, so the usage request,
// MaxCompletionTokens and ServiceTier are not sent. Bound the reply with
// MaxTokens instead.
type LLMConfig struct {
	chat.LLMConfig
	// WikiGrounding grounds the answer in Wikipedia. Only gemma4 and
	// sarvam-105b take it; on another model a turn is refused with an error.
	// Nil sends nothing, leaving Sarvam's default.
	WikiGrounding *bool
	// ReasoningEffort is how much the model reasons before it answers. Every
	// model but sarvam-105b-conversations takes it; on that one a turn is
	// refused with an error. Empty sends nothing, leaving Sarvam's default.
	ReasoningEffort string `validate:"omitempty,oneof=low medium high"`
}

// Validate reports whether the configuration is usable: it needs an API key,
// and a model Sarvam serves.
func (c LLMConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.APIKey == "" {
		return errMissingAPIKey
	}
	if c.Model != "" && !slices.Contains(supportedLLMModels, c.Model) {
		allowed := slices.Sorted(slices.Values(supportedLLMModels))
		return fmt.Errorf("%w %q. Allowed values: %s", errUnsupportedLLMModel, c.Model, strings.Join(allowed, ", "))
	}
	return nil
}

// LLMSettings is the Sarvam LLM configuration that can change while the
// pipeline runs, through an LLMUpdateSettingsFrame: the OpenAI-compatible
// settings, plus Sarvam's own.
type LLMSettings struct {
	chat.LLMSettings
	// WikiGrounding grounds the answer in Wikipedia, on the models that take it.
	WikiGrounding settings.Opt[bool] `settings:"wiki_grounding"`
	// ReasoningEffort is how much the model reasons: "low", "medium" or "high".
	ReasoningEffort settings.Opt[string] `settings:"reasoning_effort"`
}

// NewLLM builds a Sarvam LLM service over its OpenAI-compatible endpoint.
//
// The endpoint is chosen by model when BaseURL is empty: /v1 for
// sarvam-105b-conversations and /v2 for every other model. The API key is sent
// as Sarvam's api-subscription-key header as well as a Bearer token.
//
// Before each request the service checks the turn against what the model can
// do: tool_choice without tools, an image on a model that cannot see, or a
// setting the model does not take. A turn that fails a check is not sent, and
// is reported as a non-fatal error, so a switcher can fall back to another
// service.
func NewLLM(cfg LLMConfig) *chat.LLMService {
	compat := cfg.LLMConfig
	if compat.Model == "" {
		compat.Model = defaultLLMModel
	}
	if compat.BaseURL == "" {
		compat.BaseURL = llmBaseURL(compat.Model)
	}
	if compat.APIKey != "" {
		// Applied over the caller's own headers: Sarvam authorizes by it.
		headers := maps.Clone(compat.DefaultHeaders)
		if headers == nil {
			headers = map[string]string{}
		}
		headers["api-subscription-key"] = compat.APIKey
		compat.DefaultHeaders = headers
	}

	live := &LLMSettings{}
	if cfg.WikiGrounding != nil {
		live.WikiGrounding = settings.Set(*cfg.WikiGrounding)
	}
	if cfg.ReasoningEffort != "" {
		live.ReasoningEffort = settings.Set(cfg.ReasoningEffort)
	}
	return chat.NewCompatLLM(chat.Compat{
		Name:            "SarvamLLM",
		DefaultModel:    defaultLLMModel,
		NoDeveloperRole: true,
		// The endpoint has no max_completion_tokens field.
		NoMaxCompletionTokens: true,
		Settings:              live,
		CheckRequest:          live.checkRequest,
		Request:               live.editRequest,
	}, compat)
}

// llmBaseURL is the API base model is served under: /v1 for the models still
// served there, /v2 for every other.
func llmBaseURL(model string) string {
	if slices.Contains(v1Models, model) {
		return llmAPIBase + "/v1"
	}
	return llmAPIBase + "/v2"
}

// editRequest shapes a request the way Sarvam takes it: without the fields it
// does not take, and with its own controls on the models that take them.
func (s *LLMSettings) editRequest(body map[string]any) {
	delete(body, "stream_options")
	delete(body, "max_completion_tokens")
	delete(body, "service_tier")

	model := s.Model.Or("")
	if slices.Contains(wikiGroundingModels, model) {
		if grounding, ok := s.WikiGrounding.Value(); ok {
			body["wiki_grounding"] = grounding
		}
	}
	if slices.Contains(reasoningModels, model) {
		if effort, ok := s.ReasoningEffort.Value(); ok {
			body["reasoning_effort"] = effort
		}
	}
}

// checkRequest refuses a turn the model cannot serve, naming why, so the error
// says what to change rather than the request failing on Sarvam's side or the
// setting being dropped without a word.
func (s *LLMSettings) checkRequest(p openai.Params) error {
	if err := checkToolParameters(p); err != nil {
		return err
	}
	if err := s.checkVisionSupport(p); err != nil {
		return err
	}
	return s.checkCapabilitySupport()
}

// checkToolParameters refuses a tool_choice sent without tools to choose from.
func checkToolParameters(p openai.Params) error {
	if p.ToolChoice != "" && len(p.Tools) == 0 {
		return errToolChoiceWithoutTools
	}
	return nil
}

// checkVisionSupport refuses an image sent to a model that cannot see.
func (s *LLMSettings) checkVisionSupport(p openai.Params) error {
	model := s.Model.Or("")
	if slices.Contains(visionModels, model) {
		return nil
	}
	for _, m := range p.Messages {
		for _, part := range m.ContentParts {
			if part.Type == partTypeImageURL {
				return fmt.Errorf("sarvam: model %q %w", model, errNoImageInput)
			}
		}
	}
	return nil
}

// checkCapabilitySupport refuses a reasoning effort or wiki grounding set on a
// model that does not take it.
func (s *LLMSettings) checkCapabilitySupport() error {
	model := s.Model.Or("")
	if _, ok := s.ReasoningEffort.Value(); ok && !slices.Contains(reasoningModels, model) {
		return fmt.Errorf("sarvam: model %q %w", model, errNoReasoningEffort)
	}
	if _, ok := s.WikiGrounding.Value(); ok && !slices.Contains(wikiGroundingModels, model) {
		return fmt.Errorf("sarvam: model %q %w", model, errNoWikiGrounding)
	}
	return nil
}
