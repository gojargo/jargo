// Package grok provides xAI's (Grok) services: an OpenAI-compatible LLM,
// real-time streaming speech-to-text, and text-to-speech in both a streaming
// WebSocket and a batch HTTP form. The speech services share the voice API
// documented alongside the chat models and authorize with the same API key.
package grok

const (
	baseURL      = "https://api.x.ai/v1"
	defaultModel = "grok-4.20-non-reasoning"
)
