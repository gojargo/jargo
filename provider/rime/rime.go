// Package rime is a streaming text-to-speech service backed by Rime's HTTP API.
// It requests raw PCM via the Accept header and streams it downstream.
package rime

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	apiURL = "https://users.rime.ai/v1/rime-tts"
	// modelCoda is Rime's current model, and the one whose controls differ from
	// the older mist models.
	modelCoda    = "coda"
	defaultModel = modelCoda
	// defaultLang is English as Rime names it. Rime takes three-letter codes
	// ("eng", "fra", "ger", "spa", "hin"), not the two-letter ones.
	defaultLang = "eng"
	// defaultSampleRate is the PCM rate jargo requests from Rime.
	defaultSampleRate = 24000
)

// Config configures the Rime TTS service. The model-specific controls are
// pointers (or empty strings) and are omitted from the request when unset, and
// only the ones the chosen model accepts are sent at all.
type Config struct {
	// APIKey is the Rime API key. Required.
	APIKey string `validate:"required"`
	// Model is the Rime model id; empty uses Rime's current model. It also
	// selects which of the controls below are sent: the sampling controls go to
	// coda, the bracket and latency controls to the older mist models, and each
	// model rejects the other's.
	Model string
	// Speaker is the voice id; empty asks Rime to pick one.
	Speaker string
	// Lang is the spoken language as Rime names it, a three-letter code such as
	// "eng", "fra", "ger", "spa" or "hin"; empty uses English.
	Lang string
	// SampleRate is the PCM rate requested from Rime and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// RepetitionPenalty penalizes token repetition (coda only, 1.0-2.0).
	RepetitionPenalty *float64
	// Temperature is the sampling temperature (coda only, 0.0-1.0).
	Temperature *float64
	// TopP is the cumulative-probability threshold (coda only, 0.0-1.0).
	TopP *float64
	// SpeedAlpha is the speech-speed multiplier (mistv2 only).
	SpeedAlpha *float64
	// ReduceLatency trades quality for latency (mistv2 only).
	ReduceLatency *bool
	// PauseBetweenBrackets adds pauses between bracketed content (mistv2 only).
	PauseBetweenBrackets *bool
	// PhonemizeBetweenBrackets phonemizes bracketed content (mistv2 only).
	PhonemizeBetweenBrackets *bool
	// InlineSpeedAlpha is inline speed-control markup; empty omits it.
	InlineSpeedAlpha string
	// TimeScaleFactor scales playback speed (coda only); >1 slows down.
	TimeScaleFactor *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
