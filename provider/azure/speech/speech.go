// Package speech provides Azure AI Speech services. NewTTS is a streaming
// text-to-speech service over the REST endpoint: it sends SSML and requests raw
// 16-bit mono PCM, which streams straight downstream. NewRealtimeTTS synthesizes
// over Azure's synthesis WebSocket protocol, streaming the audio and the word
// boundaries of each sentence as they are generated. NewSTT is a streaming
// speech-to-text service speaking Azure's recognition WebSocket protocol. Both
// WebSocket services speak the protocol directly, so they need no native SDK.
package speech

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
)

const (
	defaultTTSRate = 24000
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
	// Voice is the SSML voice name (e.g. "en-US-JennyNeural"); empty uses
	// "en-US-SaraNeural".
	Voice string
	// Language sets the SSML xml:lang; the zero value uses en-US. A base language
	// takes the locale Azure defaults it to.
	Language language.Language
	// Effect names the audio effect processor that compensates for playback
	// distortion on the device the audio is bound for, as SSML's effect attribute
	// on <voice>: "eq_car" for an enclosed vehicle, "eq_telecomhp8k" for
	// narrowband telephony, which is what the 8 kHz telephony pipelines carry.
	// Empty leaves the attribute off.
	Effect string
	// ForceLocale wraps the text in SSML's <lang> element, so a multilingual
	// voice (en-US-EmmaMultilingualNeural, say) speaks in Language rather than
	// the language it reads out of the text. A single-locale voice ignores the
	// element. It also stops the voice switching language part-way, so mixed
	// text is spoken entirely in Language: pair it with a language the voice
	// actually speaks, since it synthesizes nothing at all for one it does not.
	ForceLocale bool
	// VoiceParameters sets the model parameters of an HD voice, as SSML's
	// semicolon-separated parameters attribute on <voice>
	// ("temperature=0.7;top_p=0.8", say). Which keys apply depends on the voice's
	// base model; standard neural voices ignore it. Empty leaves it off.
	VoiceParameters string
	// Style is the speaking style ("cheerful", "sad"); empty speaks in the
	// voice's own.
	Style string
	// StyleDegree is the intensity of the style, from 0.01 to 2.0; empty leaves
	// it at the default.
	StyleDegree string
	// Role is the role the voice plays ("YoungAdultFemale"); empty leaves it
	// unset.
	Role string
	// Rate, Pitch and Volume adjust the prosody ("1.25" or "slow", "+10%" or
	// "high", "+20%" or "loud"); empty leaves each as it is.
	Rate   string
	Pitch  string
	Volume string
	// Emphasis is the level of emphasis ("strong", "moderate", "reduced"); empty
	// adds none.
	Emphasis string
	// SampleRate is the PCM output rate; empty uses 24 kHz. Must be one Azure
	// offers as raw PCM (8000, 16000, 22050, 24000, 44100, 48000).
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }
