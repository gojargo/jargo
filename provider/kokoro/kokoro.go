// Package kokoro synthesizes speech with Kokoro, a small open-weight
// text-to-speech model that runs locally on the CPU: no server and no API key.
//
// NewTTS runs the Kokoro v1.0 ONNX export in process, through the ONNX Runtime
// jargo already binds for VAD and turn detection (located from
// JARGO_ONNXRUNTIME_LIB, or by its conventional name). The text is
// phonemized with espeak-ng, whose shared library (libespeak-ng) is bound
// through ebitengine/purego rather than cgo, so the package always builds and
// the library is loaded at run time: from Config.Espeak.LibPath, or by its
// conventional names on the loader's default search path. Distributions
// package it as libespeak-ng1.
//
// The model and voices files are downloaded once, on first use, from the
// kokoro-onnx release into a cache under the user's cache directory, about
// 350 MB together. Config.ModelPath and Config.VoicesPath name files already
// on disk instead.
//
// NewHTTPTTS speaks through a Kokoro-FastAPI server instead, for a deployment
// that runs the model out of process.
package kokoro

import (
	"net/http"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

// EspeakLibPathEnv is the environment variable applications and the eval
// harness read Config.Espeak.LibPath from. This package reads no environment
// itself.
const EspeakLibPathEnv = "JARGO_ESPEAK_LIB"

// Config configures the local Kokoro TTS service.
type Config struct {
	// Voice is the voice to speak in, a name the voices file holds, such as
	// "af_heart". Required. Voices are language-specific: "af_heart" speaks
	// US English whatever the language is set to.
	Voice string
	// Language is the language of the text, which picks how it is
	// phonemized; the zero value uses English.
	Language language.Language
	// Speed is the speech rate multiplier, from 0.5 to 2.0; nil uses 1.0. A
	// rate outside that range is reported when speech is synthesized.
	Speed *float64

	// ModelPath is the Kokoro ONNX model file; empty uses kokoro-v1.0.onnx in
	// CacheDir. A file that does not exist is downloaded there.
	ModelPath string
	// VoicesPath is the voices file; empty uses voices-v1.0.bin in CacheDir.
	// A file that does not exist is downloaded there.
	VoicesPath string
	// CacheDir is where the model and voices files are downloaded; empty
	// uses DefaultCacheDir.
	CacheDir string
	// HTTPClient downloads the files; nil uses a client of its own.
	HTTPClient *http.Client

	// Espeak locates espeak-ng, which phonemizes the text.
	Espeak EspeakConfig

	// SampleRate is the rate the service outputs; 0 uses SampleRate, the
	// rate the model synthesizes at. Another rate is resampled to.
	SampleRate int `validate:"gte=0"`
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
