package baseten

import "github.com/gojargo/jargo/provider/openai/chat"

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
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM(chat.Compat{
		Name:            "BasetenLLM",
		BaseURL:         baseURL,
		DefaultModel:    defaultModel,
		NoDeveloperRole: true,
	}, cfg)
}
