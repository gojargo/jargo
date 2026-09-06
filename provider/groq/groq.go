// Package groq provides Groq's OpenAI-compatible LLM service and its Whisper
// speech-to-text and text-to-speech services.
package groq

import "errors"

const (
	baseURL         = "https://api.groq.com/openai/v1"
	defaultLLMModel = "openai/gpt-oss-120b"
	defaultSTTModel = "whisper-large-v3-turbo"
)

// errStatus is returned when the API responds with a non-200 status. It is shared
// by the STT and TTS services.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("groq: unexpected status")
