package baseten

import (
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
)

// LLMService is the Baseten LLM service NewLLM builds.
type LLMService struct {
	*chat.LLMService
}

// NewLLM builds a Baseten LLM service.
//
// It defaults to Baseten's Model APIs, a serverless endpoint hosting
// open-weights models. To use a dedicated deployment running on your own GPUs,
// set BaseURL to the deployment's /sync/v1 URL and Model to its served model
// name:
//
//	baseten.NewLLM(chat.LLMConfig{
//		APIKey:  apiKey,
//		BaseURL: "https://model-{model_id}.api.baseten.co/environments/production/sync/v1",
//		Model:   "Qwen/Qwen2.5-3B-Instruct",
//	})
//
// Baseten's message roles are system, user, assistant and tool, so a developer
// message is sent as a user message.
func NewLLM(cfg chat.LLMConfig) *LLMService {
	s := &LLMService{}
	s.LLMService = buildLLM(cfg, llm.WithSelf(s))
	return s
}

// buildLLM builds the service NewLLM returns, registering the
// service opts names as the processor the pipeline runs.
func buildLLM(cfg chat.LLMConfig, opts ...llm.Option) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "BasetenLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg, opts...)
}
