// Package groq provides Groq's OpenAI-compatible LLM service and its Whisper
// speech-to-text and text-to-speech services.
package groq

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL         = "https://api.groq.com/openai/v1"
	defaultLLMModel = "llama-3.3-70b-versatile"
	defaultSTTModel = "whisper-large-v3-turbo"
)

// NewLLM builds a Groq LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("GroqLLM", baseURL, defaultLLMModel, cfg)
}
