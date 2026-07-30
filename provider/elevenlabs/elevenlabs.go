// Package elevenlabs is a streaming text-to-speech service backed by ElevenLabs.
// The shared TTS base aggregates incoming text into sentences; the service
// synthesizes each and streams raw PCM downstream at the configured rate (48 kHz
// by default, matching the WebRTC Opus rate so no resampling is needed).
//
// There are two transports. NewTTS wraps the HTTP /stream endpoint, one request
// per sentence. NewRealtimeTTS speaks the WebSocket multi-stream-input protocol
// over a single connection held open for the session, so a sentence pays no
// connection setup before its first audio; it also reports word-level timing,
// which lets the assistant context record what was actually spoken when a turn
// is cut short. Prefer it for conversation, where the pause between one
// synthesized sentence and the next is heard.
//
// The package also provides streaming speech-to-text (NewSTT, NewRealtimeSTT).
package elevenlabs

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

const (
	// defaultBaseURL is the ElevenLabs HTTP API base.
	defaultBaseURL = "https://api.elevenlabs.io"
	// defaultSampleRate matches the WebRTC Opus rate so the output transport
	// does not resample. 48 kHz PCM is not tier-gated (only 44.1 kHz requires a
	// Pro plan).
	defaultSampleRate = 48000
	// defaultVoiceID is a public ElevenLabs voice ("Rachel").
	defaultVoiceID = "21m00Tcm4TlvDq8ikWAM"
	// defaultModel is the lowest-latency multilingual model.
	defaultModel = "eleven_flash_v2_5"
	// langFil is the Filipino base language code.
	langFil = "fil"
)

// Config configures the TTS service.
type Config struct {
	// APIKey is the ElevenLabs API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the HTTP API base; empty uses the hosted API.
	BaseURL string
	// VoiceID is the ElevenLabs voice; empty uses a default public voice.
	VoiceID string
	// Model is the ElevenLabs model; empty uses the low-latency flash model.
	Model string
	// SampleRate is the PCM rate requested from ElevenLabs and emitted
	// downstream. Empty uses 48 kHz. Must be a rate ElevenLabs supports
	// (8000, 16000, 22050, 24000, 32000, 44100, 48000).
	SampleRate int
	// Language for multilingual models; the zero value leaves it unset (the
	// model auto-detects). Mapped to ElevenLabs' base code per voice.
	Language language.Language
	// VoiceSettings overrides the voice's default settings when non-nil.
	VoiceSettings *VoiceSettings
	// OptimizeStreamingLatency requests a latency-optimization level (0-4); nil
	// leaves it unset.
	OptimizeStreamingLatency *int
	// ApplyTextNormalization controls spoken-form text normalization
	// ("auto", "on", "off"); empty leaves it unset.
	ApplyTextNormalization string
	// EnableLogging toggles ElevenLabs server-side logging; nil leaves it unset.
	// Set to false for zero-retention mode (enterprise only).
	EnableLogging *bool
	// PronunciationDictionaryLocators applies the given pronunciation
	// dictionaries; empty applies none.
	PronunciationDictionaryLocators []PronunciationDictionaryLocator
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
