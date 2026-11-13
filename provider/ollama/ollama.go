// Package ollama provides a client for a local Ollama server through its
// OpenAI-compatible LLM endpoint.
package ollama

const (
	baseURL      = "http://localhost:11434/v1"
	defaultModel = "llama2"
	// placeholderAPIKey is the key sent when none is configured. A local
	// server ignores it, and it keeps the Authorization header well formed.
	placeholderAPIKey = "ollama"
)
