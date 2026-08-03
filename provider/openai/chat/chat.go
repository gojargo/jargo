// Package chat provides OpenAI's LLM, STT and TTS services, plus the
// OpenAI-compatible LLM base that other providers (Groq, Together, Fireworks and
// the rest) wrap with their own base URL, key and default model.
package chat

import (
	"errors"
	"net/http"
)

// defaultLLMBaseURL is the OpenAI API base used by the LLM, STT and TTS services.
const defaultLLMBaseURL = "https://api.openai.com/v1"

// errStatus is returned when the API responds with a non-200 status. It is shared
// by the LLM and STT services.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("openai: unexpected status")

// httpClientOr returns c, or a fresh client with the standard library's defaults
// when c is nil.
func httpClientOr(c *http.Client) *http.Client {
	if c != nil {
		return c
	}
	return &http.Client{}
}
