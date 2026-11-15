// Package rime provides Rime's services: text-to-speech over Rime's HTTP API
// (NewTTS), over its WebSocket JSON API (NewRealtimeTTS) and over its WebSocket
// v1 protobuf API (NewV1TTS), and streaming speech-to-text over its gRPC API.
//
// The HTTP text-to-speech service requests raw PCM via the Accept header and
// streams it downstream. The WebSocket services hold one connection open for
// the call and give each turn a context of its own on it, so the audio of a
// turn the user cut off is told apart from the next one's and dropped. The
// speech-to-text service streams each stretch of speech the VAD reports as its
// own recognition request.
package rime

import (
	"strconv"
	"strings"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
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
	// fieldSpeaker is Rime's name for the voice.
	fieldSpeaker = "speaker"
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
	// NoTextNormalization leaves numbers, dates and abbreviations as written
	// rather than spelling them out first (mistv2 only).
	NoTextNormalization *bool
	// InlineSpeedAlpha is inline speed-control markup; empty omits it.
	InlineSpeedAlpha string
	// TimeScaleFactor scales playback speed (coda only); >1 slows down.
	TimeScaleFactor *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// rimeLanguages names the languages Rime has been verified against the way its
// HTTP and WebSocket JSON APIs take them. Rime accepts both ISO 639-2/3 and ISO
// 639-1 codes, so a language it does not list falls back to its two-letter base
// code.
//
//nolint:gochecknoglobals // read-only lookup table
var rimeLanguages = map[language.Language]string{
	language.Arabic:     "ara",
	language.German:     "ger",
	language.English:    "eng",
	language.Spanish:    "spa",
	language.French:     "fra",
	language.Hindi:      "hin",
	language.Italian:    "ita",
	language.Japanese:   "jpn",
	language.Portuguese: "por",
}

// rimeLanguage maps a language to the code Rime's HTTP and WebSocket JSON APIs
// take ("eng" for English). A region-qualified language resolves through its
// base language. The zero value maps to "", which leaves the language unset.
func rimeLanguage(l language.Language) string {
	if l == "" {
		return ""
	}
	return language.Resolve(l, rimeLanguages, true)
}

// rimeV1Languages names the languages Rime has been verified against as the
// BCP-47 tags its WebSocket v1 API takes.
//
//nolint:gochecknoglobals // read-only lookup table
var rimeV1Languages = map[language.Language]string{
	language.Arabic:     "ar",
	language.German:     "de",
	language.English:    "en",
	language.Spanish:    "es",
	language.French:     "fr",
	language.Hindi:      "hi",
	language.Italian:    "it",
	language.Japanese:   "ja",
	language.Portuguese: "pt",
}

// rimeV1Language maps a language to the base BCP-47 tag Rime's WebSocket v1 API
// takes ("en" for English). A region-qualified language falls back to its base
// tag. The zero value maps to "", which leaves the language unset.
func rimeV1Language(l language.Language) string {
	if l == "" {
		return ""
	}
	return language.Resolve(l, rimeV1Languages, true)
}

// Spell wraps text in Rime's spell function, which has Rime say it letter by
// letter. Both WebSocket services keep a spell function in one piece when they
// split text into sentences.
func Spell(text string) string { return "spell(" + text + ")" }

// PauseTag is Rime's tag for a pause of the given length, in seconds.
func PauseTag(seconds float64) string { return "<" + formatFloat(seconds*1000) + ">" }

// formatFloat writes a number for Rime with a fractional part always shown, so
// 500 is "500.0" and 1.25 is "1.25".
func formatFloat(v float64) string {
	s := strconv.FormatFloat(v, 'f', -1, 64)
	if !strings.ContainsAny(s, ".eEnN") {
		s += ".0"
	}
	return s
}
