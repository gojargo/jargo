// Package pockettts is a text-to-speech provider for a local Pocket TTS server
// (Kyutai's pocket-tts, started with `pocket-tts serve`). Pocket TTS is a small
// CPU-only model: the server holds the weights in memory and streams the audio
// back as it is generated, so the first samples arrive well before the sentence
// is finished.
//
// The server chooses its language when it starts (`pocket-tts serve --language
// french_24l`), and the weights are per language, so the language is not part of
// a request and cannot be changed while the server runs. Run one server per
// language a bot speaks. The voice is per request: a built-in name, or a URL of
// a recording to clone.
//
// Kyutai's other self-hosted models, the Delayed Streams Modeling speech-to-text
// and text-to-speech served by moshi-server, are in
// [github.com/gojargo/jargo/provider/kyutai/moshi]. They are a separate server
// speaking a separate protocol.
package pockettts

import (
	"errors"

	"github.com/gojargo/jargo/internal/validate"
)

// Sentinel errors for the synthesis request.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errStatus = errors.New("pockettts: unexpected status")
	errBadWAV = errors.New("pockettts: response is not a WAV stream")
)

const (
	// defaultSampleRate is the rate every model pocket-tts ships runs at. The
	// rate the server actually used is on the WAV header of each response, and
	// the audio is labeled with that; this is what the service declares before
	// the first response arrives.
	defaultSampleRate = 24000
	// ttsPath is the synthesis endpoint of the server.
	ttsPath = "/tts"
	// readChunk is how much of the audio stream is read at a time. It is small
	// enough that the first samples are pushed downstream promptly rather than
	// waiting for a buffer to fill.
	readChunk = 4096
)

// Config configures a Pocket TTS provider.
type Config struct {
	// BaseURL is the base of the local Pocket TTS server, e.g.
	// http://localhost:8000. Required.
	BaseURL string `validate:"required,url"`
	// Voice selects the voice to speak in: a name the server knows ("alba",
	// "estelle", "giovanni"), or an http://, https:// or hf:// URL of a
	// recording to clone. Empty lets the server use the default voice for the
	// language it was started with. A name the server does not know is refused
	// by the server rather than here, since it holds the list.
	Voice string
	// SampleRate is the rate the service declares before it has seen a
	// response; 0 uses 24 kHz, which is what every bundled model runs at. Each
	// response carries the rate it was really generated at, and the audio is
	// labeled with that, so a mismatch costs a warning rather than the wrong
	// playback speed.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
