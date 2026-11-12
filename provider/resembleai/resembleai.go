// Package resembleai is a streaming text-to-speech service backed by Resemble
// AI's WebSocket. It holds one connection open for the call and sends each
// sentence as a request of its own, numbered so the audio that comes back can be
// told apart. The request sets no_audio_header so the audio arrives without a
// header, as 16-bit PCM by default.
package resembleai

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/wsservice"
)

const (
	defaultURL          = "wss://websocket.cluster.resemble.ai/stream"
	defaultSampleRate   = 22050
	defaultPrecision    = "PCM_16"
	defaultOutputFormat = "wav"
)

// Config configures the Resemble AI TTS service.
type Config struct {
	// APIKey is the Resemble AI API key. Required.
	APIKey string `validate:"required"`
	// Voice is the voice UUID to synthesize with. Required.
	Voice string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// SampleRate is the rate requested and emitted downstream: 8000, 16000,
	// 22050, 32000 or 44100. 0 uses 22.05 kHz.
	SampleRate int
	// Precision is the PCM bit depth: "PCM_32", "PCM_24", "PCM_16" or "MULAW".
	// Empty uses "PCM_16".
	Precision string
	// OutputFormat is the audio format, "wav" or "mp3"; empty uses "wav". The
	// audio is passed on as it arrives, so anything but the default headerless
	// PCM needs decoding downstream.
	OutputFormat string
	// ApplyCustomPronunciations has Resemble AI apply the team's custom
	// pronunciations to matching words in the text. It can change while the
	// pipeline runs.
	ApplyCustomPronunciations bool
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering. Resemble AI never
	// acknowledges the closing handshake, so a nil CloseTimeout sends the close
	// without waiting rather than using the usual default.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
