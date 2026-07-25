// Package together provides Together AI's OpenAI-compatible LLM service and its
// streaming speech-to-text and text-to-speech services.
package together

const (
	baseURL      = "https://api.together.xyz/v1"
	defaultModel = "zai-org/GLM-5.1"
)

// msgType is the discriminator key shared by the STT and TTS client/server
// frames.
const msgType = "type"
