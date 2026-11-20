// Package heygen provides HeyGen Voice text-to-speech: a streaming service that
// sends text in and plays audio out over the
// /v3/models/audio/tts/multi-stream-input WebSocket, with word timestamps, and
// an HTTP service streaming raw PCM from POST /v3/models/audio/tts/stream, one
// request per sentence.
//
// See https://developers.heygen.com/docs/voices/heygen-voice-speech.
package heygen

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"

	"github.com/gojargo/jargo/language"
)

const (
	// defaultVoiceID is Nadine, a HeyGen public voice. All HeyGen Voice models
	// serve it.
	defaultVoiceID = "83548de556df48ba8c09c42a57c51d85"
	// defaultModel is the lowest-latency HeyGen Voice model.
	defaultModel = "heygen-voice-1-turbo"
	// defaultSampleRate is the PCM rate emitted downstream when the caller names
	// none: the pipeline's default output rate.
	defaultSampleRate = 24000
	// fallbackSampleRate is the rate HeyGen synthesizes at when it does not
	// offer the service's rate. Its audio is resampled to the service's rate.
	fallbackSampleRate = 24000
	// errorBodyLimit bounds how much of a body that is not an error envelope is
	// reported.
	errorBodyLimit = 500
	// headerAPIKey carries the API key on every request.
	//
	//nolint:gosec // G101: a header name, not a credential
	headerAPIKey = "X-Api-Key"
)

// httpSampleRates are the PCM rates the HTTP endpoint synthesizes at.
//
//nolint:gochecknoglobals // read-only lookup table
var httpSampleRates = []int{8000, 16000, 22050, 24000, 44100, 48000}

// websocketSampleRates are the PCM rates the WebSocket synthesizes at.
//
//nolint:gochecknoglobals // read-only lookup table
var websocketSampleRates = []int{8000, 16000, 22050, 24000, 32000, 44100, 48000}

// heygenLanguages are the languages HeyGen Voice speaks, under the lowercase
// ISO 639-1 codes it takes.
//
//nolint:gochecknoglobals // read-only lookup table
var heygenLanguages = map[language.Language]string{
	language.Arabic:     "ar",
	language.Belarusian: "be",
	language.Bulgarian:  "bg",
	language.Catalan:    "ca",
	language.Czech:      "cs",
	language.Danish:     "da",
	language.German:     "de",
	language.Greek:      "el",
	language.English:    "en",
	language.Spanish:    "es",
	language.Persian:    "fa",
	language.Finnish:    "fi",
	language.French:     "fr",
	language.Hebrew:     "he",
	language.Hindi:      "hi",
	language.Croatian:   "hr",
	language.Hungarian:  "hu",
	language.Indonesian: "id",
	language.Italian:    "it",
	language.Japanese:   "ja",
	language.Korean:     "ko",
	language.Macedonian: "mk",
	language.Malay:      "ms",
	language.Dutch:      "nl",
	language.Polish:     "pl",
	language.Portuguese: "pt",
	language.Romanian:   "ro",
	language.Russian:    "ru",
	language.Slovak:     "sk",
	language.Slovenian:  "sl",
	language.Serbian:    "sr",
	language.Swedish:    "sv",
	language.Tamil:      "ta",
	language.Thai:       "th",
	language.Tagalog:    "tl",
	language.Turkish:    "tr",
	language.Ukrainian:  "uk",
	language.Vietnamese: "vi",
	language.Chinese:    "zh",
}

// heygenLanguage maps a Language to the HeyGen Voice language code.
//
// HeyGen Voice takes lowercase ISO 639-1 codes and has no regional accents, so a
// regional variant ("en-US") resolves to its base code ("en"). A code HeyGen
// Voice does not support is logged, and the services do not send it, so HeyGen
// detects the language.
func heygenLanguage(l language.Language) string {
	code := language.Resolve(l, heygenLanguages, true)
	if !supportedLanguage(code) {
		slog.Warn(fmt.Sprintf("Language code [%s] not applied. HeyGen Voice doesn't support it, "+
			"so HeyGen detects the language.", code))
	}
	return code
}

// supportedLanguage reports whether HeyGen Voice speaks the language code.
// HeyGen refuses a language it does not speak, and detects one left out.
func supportedLanguage(code string) bool {
	for _, c := range heygenLanguages {
		if c == code {
			return true
		}
	}
	return false
}

// synthesisRate is the PCM rate to request from HeyGen for the service's rate:
// that rate when HeyGen offers it, the fallback otherwise, which the service
// resamples.
func synthesisRate(sampleRate int, supported []int) int {
	if slices.Contains(supported, sampleRate) {
		return sampleRate
	}
	slog.Warn(fmt.Sprintf("HeyGen cannot synthesize %d Hz (supports %v); synthesizing at %d Hz "+
		"and resampling on output", sampleRate, supported, fallbackSampleRate))
	return fallbackSampleRate
}

// errorDetail unwraps HeyGen's {"error": {"code", "message", "request_id"}}
// envelope: the error's code, message and request ID, or the start of the raw
// body when it is not an envelope.
func errorDetail(body []byte) string {
	var envelope map[string]any
	_ = json.Unmarshal(body, &envelope)
	detail, _ := envelope["error"].(map[string]any)
	code, hasCode := detail["code"]
	message, hasMessage := detail["message"]
	if !hasCode || !hasMessage {
		return string(body[:min(len(body), errorBodyLimit)])
	}
	requestID, ok := detail["request_id"]
	if !ok || requestID == nil {
		requestID = "none"
	}
	return fmt.Sprintf("%v: %v (request_id: %v)", code, message, requestID)
}
