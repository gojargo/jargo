// Package piper synthesizes speech with Piper, a fast local neural
// text-to-speech system whose voices run on the CPU: no API key.
//
// NewTTS runs a Piper voice in process, in libpiper, Piper's native library.
// It is bound through ebitengine/purego rather than cgo, so the package always
// builds and the library is loaded at run time: from Config.LibraryPath, or by
// its conventional name (libpiper.so, libpiper.dylib, piper.dll) on the
// loader's default search path. libpiper phonemizes the text with the
// espeak-ng it is built with, whose data it reads from Config.EspeakDataPath or
// from share/espeak-ng-data beside the library's lib directory. It runs the
// voice on the ONNX Runtime it is linked against. `make deps-piper` builds it
// under .native/, against the ONNX Runtime jargo's VAD and turn models run on,
// so a process using both loads one copy of the runtime. The dynamic loader
// gives libpiper the runtime already loaded when there is one, so that runtime
// has to be at least the version libpiper was built against.
//
// The voice is downloaded once, on first use, from the rhasspy/piper-voices
// repository on Hugging Face into a cache under the user's cache directory:
// the model (<voice>.onnx) and its configuration (<voice>.onnx.json).
//
// libpiper is licensed under the GPL-3.0. Distributing an application that
// loads it may subject the application to the GPL's source-disclosure terms.
// To keep Piper out of an application's license scope, use NewHTTPTTS, which
// speaks through a separately installed Piper HTTP server
// (python -m piper.http_server).
package piper

import (
	"errors"

	"github.com/gojargo/jargo/service/settings"
)

// LibPathEnv is the environment variable applications and the eval harness
// read Config.LibraryPath from. This package reads no environment itself.
const LibPathEnv = "JARGO_PIPER_LIB"

// errVoiceRequired reports settings that name no voice.
var errVoiceRequired = errors.New("Piper TTS voice must be specified") //nolint:staticcheck // the service's name leads

// synthesisFields are the settings Piper takes with each utterance it
// synthesizes, so a runtime update of any of them applies to the next one.
//
//nolint:gochecknoglobals // a read-only list
var synthesisFields = []string{"speaker_id", "length_scale", "noise_scale", "noise_w_scale"}

// TTSSettings is the configuration of the in-process Piper service that can
// change while the pipeline runs, through a TTSUpdateSettingsFrame.
//
// The synthesis controls take effect with the next utterance. The voice is
// loaded once, when the service is built, so a change to it (or to the model
// or the language) is stored but not applied, and is reported as such.
type TTSSettings struct {
	settings.TTS

	// SpeakerID is which speaker to use, for voices trained on several
	// speakers (such as en_US-libritts_r-medium). It is an index from 0 to the
	// voice's speaker count minus one, listed in the voice's .onnx.json file.
	// Voices with a single speaker ignore it.
	SpeakerID settings.Opt[int] `settings:"speaker_id"`
	// LengthScale is the phoneme length scale: below 1 is faster, above 1 is
	// slower.
	LengthScale settings.Opt[float64] `settings:"length_scale"`
	// NoiseScale is the amount of generator noise, which varies the audio.
	NoiseScale settings.Opt[float64] `settings:"noise_scale"`
	// NoiseWScale is the amount of phoneme width noise, which varies the
	// speaking style.
	NoiseWScale settings.Opt[float64] `settings:"noise_w_scale"`
}

// HTTPTTSSettings is the configuration of the Piper HTTP service that can
// change while the pipeline runs, through a TTSUpdateSettingsFrame. It holds
// the same fields as TTSSettings. Every request is built from it, so any
// change, the voice included, applies from the next utterance on.
type HTTPTTSSettings TTSSettings

// synthesisConfig is the synthesis controls a configuration starts a service
// with; a nil one leaves the voice's own default.
type synthesisConfig struct {
	speakerID   *int
	lengthScale *float64
	noiseScale  *float64
	noiseWScale *float64
}

// newTTSSettings is the settings store a service starts with: no model and no
// language, the voice given, and the synthesis controls given, each of the
// rest held with no value so the voice's own default applies.
func newTTSSettings(voice string, syn synthesisConfig) TTSSettings {
	var s TTSSettings
	s.Model = settings.Cleared[string]()
	s.Voice = settings.Cleared[string]()
	if voice != "" {
		s.Voice = settings.Set(voice)
	}
	s.Language = settings.Cleared[string]()
	s.SpeakerID = optional(syn.speakerID)
	s.LengthScale = optional(syn.lengthScale)
	s.NoiseScale = optional(syn.noiseScale)
	s.NoiseWScale = optional(syn.noiseWScale)
	return s
}

// optional is a setting holding *v, or given no value when v is nil.
func optional[T any](v *T) settings.Opt[T] {
	if v == nil {
		return settings.Cleared[T]()
	}
	return settings.Set(*v)
}

// requireVoice is the voice the settings name, or errVoiceRequired when they
// name none or an empty one.
func requireVoice(o settings.Opt[string]) (string, error) {
	v, ok := o.Value()
	if !ok || v == "" {
		return "", errVoiceRequired
	}
	return v, nil
}
