// Package nvidia provides NVIDIA services: an NVIDIA NIM OpenAI-compatible LLM
// (NewLLM), a Riva streaming speech-to-text service (NewSTT), and a Riva
// streaming text-to-speech service (NewTTS). Both speech services talk to
// NVIDIA's hosted endpoints or to a locally deployed Riva/NIM model (parakeet
// for recognition, magpie for synthesis), selected through the server address,
// the TLS setting, and the auth fields.
package nvidia

const (
	baseURL      = "https://integrate.api.nvidia.com/v1"
	defaultModel = "nvidia/nemotron-3-nano-30b-a3b"
)
