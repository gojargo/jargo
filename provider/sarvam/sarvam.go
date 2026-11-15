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

// Request keys shared by the TTS services.
const (
	// keyText carries the text to synthesize.
	keyText = "text"
	// keyTargetLanguageCode is Sarvam's name for the language.
	keyTargetLanguageCode = "target_language_code"
	// keySpeaker is Sarvam's name for the voice.
	keySpeaker = "speaker"
)
