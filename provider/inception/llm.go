package inception

import (
	"log/slog"

	"github.com/gojargo/jargo/provider/openai/chat"
)

// NewLLM builds an Inception LLM service.
//
// Deprecated: No replacement. Inception stops serving the Mercury API after
// December 28, 2026, so NewLLM will be removed in the first release after that
// date.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	slog.Warn("inception: `NewLLM` is deprecated and will be removed. No replacement. " +
		"Inception stops serving the Mercury API after December 28, 2026, so the service " +
		"will be removed in the first release after that date.")
	return chat.NewCompatLLM(chat.Compat{
		Name:            "InceptionLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg)
}
