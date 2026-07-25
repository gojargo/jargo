// Package sarvam provides Sarvam's OpenAI-compatible LLM plus its WebSocket
// streaming text-to-speech and speech-to-text services for Indian languages.
package sarvam

// Wire-protocol keys shared by the STT and TTS transports.
const (
	// msgType is the message discriminator key.
	msgType = "type"
	// msgData is the payload field key.
	msgData = "data"
)
