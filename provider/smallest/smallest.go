// Package smallest is a streaming text-to-speech service backed by Smallest AI's
// Waves WebSocket. It holds one connection open for the call, sends the
// sentences of a turn on the turn's continuation context so they are spoken as
// one generation, and labels the audio that comes back with the context
// playing, since Smallest's messages name no request.
package smallest

import (
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/wsservice"
)

const (
	defaultURL      = "wss://api.smallest.ai/waves/v1/tts/live"
	modelLightning  = "lightning_v3.1"
	modelLightningP = "lightning_v3.1_pro"
	defaultModel    = modelLightningP
	// defaultLanguage and defaultSampleRate are the same for every model.
	defaultLanguage   = "en"
	defaultSampleRate = 24000
	// defaultOutputFormat is raw PCM, the audio the pipeline carries.
	defaultOutputFormat = "pcm"
)

// modelDefaultVoices is the voice each model speaks with when the caller names
// none. The two models have different voice catalogs, so one default for both
// would name a voice the other does not serve.
//
//nolint:gochecknoglobals // a fixed table
var modelDefaultVoices = map[string]string{
	modelLightning:  "sophia",
	modelLightningP: "meher",
}

// Config configures the Smallest AI TTS service.
type Config struct {
	// APIKey is the Smallest AI API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the Smallest model id; empty uses a default.
	Model string
	// Voice is the voice id; empty uses a default voice.
	Voice string
	// Language is the synthesis language code; empty uses "en".
	Language string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int
	// OutputFormat is the audio format Smallest returns: "pcm", "mp3", "wav",
	// "ulaw" or "alaw". Empty uses "pcm", which is what the pipeline carries.
	OutputFormat string `validate:"omitempty,oneof=pcm mp3 wav ulaw alaw"`
	// WordTimestamps asks for the timing of each word, so the text of each word
	// is pushed as its audio plays and an interruption leaves the context with
	// only what was heard. Nil turns it on. Voices that do not report word
	// timing send none, which leaves it harmless to keep on whatever the voice.
	WordTimestamps *bool
	// MaxBufferDelayMs bounds, in milliseconds (0 to 5000), how long the server
	// buffers a fragment of a turn before speaking it. Nil leaves the server's
	// default, 3000.
	MaxBufferDelayMs *int `validate:"omitempty,gte=0,lte=5000"`
	// Speed multiplies the speaking rate; nil uses the service default.
	Speed *float64
	// MathNotation reads a math operator between digits as a word, so "2 + 2" is
	// spoken as "two plus two" rather than left to the default number reader; nil
	// leaves the API's own default, which is off.
	MathNotation *bool
	// ContentFilter checks the text against Smallest's profanity word list
	// before synthesis, taking this action on a match: "reject" fails the
	// request with a CONTENT_FILTER_BLOCKED error and synthesizes nothing;
	// "flag" synthesizes normally and records the match; "redact" drops the
	// matched term and speaks the rest, failing like "reject" when nothing is
	// left to speak. Empty leaves the API default, which is off.
	ContentFilter string
	// DateFormat is the order all-numeric dates are read in, independent of the
	// synthesis voice: "EU" reads 05/03/2024 day-first ("fifth March, twenty
	// twenty four"), "US" month-first ("May third, twenty twenty four"). Empty
	// leaves the API default, "US".
	DateFormat string
	// NumberPronunciationLanguage is the language code numerals, currency,
	// times and dates are read in, independent of the synthesis language.
	// Empty reads them in the synthesis language.
	NumberPronunciationLanguage string
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
