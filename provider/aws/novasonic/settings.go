package novasonic

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// Settings are the Nova Sonic service's settings: the general model settings,
// the voice, and the endpointing sensitivity. Nova Sonic takes them all when a
// session opens, so an update reaches the session only when the next one opens.
type Settings struct {
	settings.LLM

	// Voice is the voice the model speaks with.
	Voice settings.Opt[string] `settings:"voice"`
	// EndpointingSensitivity is how quickly Nova Sonic decides the user has
	// stopped speaking: LOW, MEDIUM or HIGH. Empty leaves the model's own
	// behavior.
	EndpointingSensitivity settings.Opt[string] `settings:"endpointing_sensitivity"`
}

// newSettings is the starting state, taken from what the service was built
// with over the defaults.
func newSettings(cfg Config) *Settings {
	s := &Settings{}
	s.Model = settings.Set(cfg.Model)
	s.SystemInstruction = settings.Cleared[string]()
	if cfg.Instructions != "" {
		s.SystemInstruction = settings.Set(cfg.Instructions)
	}
	s.Voice = settings.Set(cfg.Voice)
	s.Temperature = settings.Set(defaultTemperature)
	if cfg.Temperature != nil {
		s.Temperature = settings.Set(*cfg.Temperature)
	}
	s.MaxTokens = settings.Set(defaultMaxTokens)
	if cfg.MaxTokens != nil {
		s.MaxTokens = settings.Set(*cfg.MaxTokens)
	}
	s.TopP = settings.Set(defaultTopP)
	if cfg.TopP != nil {
		s.TopP = settings.Set(*cfg.TopP)
	}
	s.TopK = settings.Cleared[int]()
	s.FrequencyPenalty = settings.Cleared[float64]()
	s.PresencePenalty = settings.Cleared[float64]()
	s.Seed = settings.Cleared[int]()
	s.FilterIncompleteUserTurns = settings.Set(false)
	s.UserTurnCompletionConfig = settings.Cleared[any]()
	s.EndpointingSensitivity = settings.Set(cfg.EndpointingSensitivity)

	// The first-generation model takes no endpointing sensitivity.
	if sensitivity := s.EndpointingSensitivity.Or(""); sensitivity != "" &&
		!endpointingSensitivitySupported(s.Model.Or("")) {
		slog.Warn("the endpointing sensitivity is not supported by this model and is ignored; "+
			"Nova 2 Sonic (amazon.nova-2-sonic-v1:0) and later models such as Nova 2.5 Sonic take it",
			"model", s.Model.Or(""), "endpointing_sensitivity", sensitivity)
		s.EndpointingSensitivity = settings.Set("")
	}
	return s
}

// isFirstGenerationModel reports whether model is the original Nova Sonic.
func isFirstGenerationModel(model string) bool { return model == firstGenerationModel }

// endpointingSensitivitySupported reports whether model takes an endpointing
// sensitivity, which Nova 2 Sonic and later models do.
func endpointingSensitivitySupported(model string) bool { return !isFirstGenerationModel(model) }

// assistantResponseTriggerNeeded reports whether model needs the assistant
// response trigger to start speaking, which only the original Nova Sonic does.
func assistantResponseTriggerNeeded(model string) bool { return isFirstGenerationModel(model) }

// Settings returns the store a settings update is merged into.
func (s *Service) Settings() any { return s.live }

// UpdateSettings reports every setting an update changed as not applied: Nova
// Sonic takes them all when a session opens, so they reach the session only
// when the next one opens.
func (s *Service) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	s.WarnUnhandledUpdatedSettings(ctx, changed.Names())
	return nil
}

// ServiceTools returns the tools the service was built with, which the base
// registers the handlers of while the conversation advertises none.
func (s *Service) ServiceTools() frames.ToolsSchema { return s.tools }

// model is the model the session runs.
func (s *Service) model() string { return s.live.Model.Or("") }
