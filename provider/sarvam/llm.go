package sarvam

import "github.com/gojargo/jargo/provider/openai/chat"

const (
	llmBaseURL      = "https://api.sarvam.ai/v1"
	defaultLLMModel = "sarvam-30b"
)

// NewLLM builds a Sarvam LLM service over its OpenAI-compatible endpoint.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	return chat.NewCompatLLM("SarvamLLM", llmBaseURL, defaultLLMModel, cfg)
}
