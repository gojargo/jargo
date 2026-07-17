// Package nvidia provides NVIDIA services: an NVIDIA NIM OpenAI-compatible LLM
// (NewLLM) and a Riva streaming speech-to-text service (NewSTT) that talks to
// NVIDIA's hosted ASR endpoint or a locally deployed Riva/NIM model such as
// parakeet.
package nvidia

import "github.com/gojargo/jargo/provider/openai"

const (
	baseURL      = "https://integrate.api.nvidia.com/v1"
	defaultModel = "nvidia/nemotron-3-nano-30b-a3b"
)

// NewLLM builds an NVIDIA NIM LLM service.
func NewLLM(cfg openai.LLMConfig) *openai.LLMService {
	return openai.NewCompatLLM("NvidiaLLM", baseURL, defaultModel, cfg)
}
