// Package speech provides Azure AI Speech services. NewTTS is a streaming
// text-to-speech service over the REST endpoint: it sends SSML and requests raw
// 16-bit mono PCM, which streams straight downstream. NewSTT is a streaming
// speech-to-text service speaking Azure's recognition WebSocket protocol
// directly, so it needs no native SDK.
package speech

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

const (
	defaultTTSVoice = "en-US-JennyNeural"
	defaultTTSRate  = 24000
	// ttsUserAgent is sent on every request: Azure requires a non-empty
	// User-Agent and rejects the request otherwise.
	ttsUserAgent = "jargo"
)

// TTSConfig configures the Azure AI Speech TTS service.
type TTSConfig struct {
	// APIKey is the Speech resource key, sent as Ocp-Apim-Subscription-Key.
	// Required.
	APIKey string `validate:"required"`
	// Region is the Speech resource region, e.g. "eastus" or "francecentral".
	// Required unless Host is set.
	Region string
	// Host overrides the full TTS host (for sovereign clouds or custom domains),
	// e.g. https://my-resource.tts.speech.azure.us. Empty derives it from Region.
	Host string
	// Voice is the SSML voice name (e.g. "en-US-JennyNeural"); empty uses a
	// default.
	Voice string
	// Language sets the SSML xml:lang; the zero value derives it from the voice's
	// locale.
	Language language.Language
	// SampleRate is the PCM output rate; empty uses 24 kHz. Must be one Azure
	// offers as raw PCM (8000, 16000, 22050, 24000, 44100, 48000).
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }
