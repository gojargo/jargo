// Package groq provides Groq's OpenAI-compatible LLM and Whisper STT services.
package groq

import (
	"github.com/gojargo/jargo/provider/openai"
	"github.com/gojargo/jargo/service/stt"
)

const (
	baseURL         = "https://api.groq.com/openai/v1"
	defaultLLMModel = "llama-3.3-70b-versatile"
	defaultSTTModel = "whisper-large-v3-turbo"
)

// NewLLM builds a Groq LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("GroqLLM", baseURL, defaultLLMModel, cfg)
}

// NewSTT builds a Groq Whisper transcription service. It is segmented: a turn
// detector upstream delimits each utterance.
func NewSTT(cfg openai.STTConfig) *stt.SegmentService {
	return openai.NewCompatSTT("GroqSTT", baseURL, defaultSTTModel, cfg)
}
