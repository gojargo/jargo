// Package speechmatics provides Speechmatics' speech services.
//
// Speech-to-text (NewSTT) streams over the Agent STT WebSocket API, which works
// in segments rather than word groups and reports speech and turn events. It
// pushes an InterimTranscriptionFrame for each partial segment and a
// TranscriptionFrame for each final one. With diarization on, each transcript
// carries the speaker Speechmatics attributed it to, and SpeakerFormat can label
// the text with it. Turns are closed either by the pipeline's own VAD
// (TurnDetectionExternal, the default) or by Speechmatics' (TurnDetectionVAD).
package speechmatics

import (
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/wsservice"
)

const (
	defaultURL = "wss://eu2.rt.speechmatics.com/v2/agent"
	// defaultModel is the Agent STT model a session uses when none is named.
	defaultModel = "linden-1"
	// agentPathSegment is the path segment that selects the Agent STT endpoint
	// on a real-time URL.
	agentPathSegment = "agent"
	// appName is the application name reported to Speechmatics as sm-app.
	appName = "jargo"
	// audioEncoding is the encoding of the audio the service sends: 16-bit
	// little-endian PCM.
	audioEncoding = "pcm_s16le"
	// bytesPerSample is the width of one sample of that audio.
	bytesPerSample = 2
	// readLimit bounds a single inbound message; transcript messages are small.
	readLimit = 1 << 20
	// startTimeout bounds the wait for RecognitionStarted.
	startTimeout = 5 * time.Second
	// endOfStreamTimeout bounds the wait for the EndOfTranscript answering
	// EndOfStream when the pipeline stops.
	endOfStreamTimeout = 5 * time.Second
)

// supportedSampleRates are the only rates the Agent STT service accepts.
//
//nolint:gochecknoglobals // read-only lookup table
var supportedSampleRates = []int{8000, 16000}

// errServer wraps an error reported by Speechmatics.
//
//nolint:gochecknoglobals // sentinel error
var errServer = errors.New("speechmatics: server error")

// errModelConflict is reported when Model and the deprecated OperatingPoint
// name different models.
//
//nolint:gochecknoglobals // sentinel error
var errModelConflict = errors.New("speechmatics: model and operating_point differ")

// TurnDetectionMode is how turn boundaries (the end of speech) are detected.
type TurnDetectionMode string

const (
	// TurnDetectionVAD has Speechmatics run its own VAD and close turns
	// itself. It reports where a turn starts and ends, which the turn
	// strategies the service recommends resolve into the user's turn.
	TurnDetectionVAD TurnDetectionMode = "vad"
	// TurnDetectionExternal leaves turn boundaries to the pipeline: the service
	// does not end turns on its own, and closes one when a
	// VADUserStoppedSpeakingFrame arrives, from the user aggregator's VAD say.
	TurnDetectionExternal TurnDetectionMode = "external"
)

// defaultTurnDetectionMode is the mode a service uses when none is given.
const defaultTurnDetectionMode = TurnDetectionExternal

// SpeakerIdentifier maps one or more known speaker identifiers to a label, so
// the engine attributes their words to that label. Identifiers come from a
// previous session and are unique to a Speechmatics account.
type SpeakerIdentifier struct {
	// Label is the label to assign to this speaker, e.g. "Agent".
	Label string `json:"label"`
	// SpeakerIdentifiers are the identifiers associated with this speaker.
	SpeakerIdentifiers []string `json:"speaker_identifiers"`
}

// AdditionalVocabEntry is a word to bias the engine towards, optionally with
// pronunciation hints.
type AdditionalVocabEntry struct {
	// Content is the word or phrase.
	Content string `json:"content"`
	// SoundsLike are alternative pronunciations, written as they sound.
	SoundsLike []string `json:"sounds_like,omitempty"`
}

// Config configures the Speechmatics STT service.
type Config struct {
	// APIKey is the Speechmatics API key, sent as a Bearer token. Required.
	APIKey string `validate:"required"`
	// URL overrides the Agent STT endpoint; empty uses the EU endpoint
	// (wss://eu2.rt.speechmatics.com/v2/agent). An "agent" path segment is
	// appended to a real-time URL that has none.
	URL string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	// Speechmatics accepts 8000 and 16000 Hz, and a session at any other rate is
	// refused for good.
	SampleRate int
	// ShouldInterrupt barges in on the bot when Speechmatics detects that the
	// user started a turn, in TurnDetectionVAD mode; nil defaults to true. It
	// is passed to the turn strategies the service recommends, which own the
	// interruption, so strategies the application configures itself override
	// it.
	ShouldInterrupt *bool

	// The options below may also be changed while the pipeline runs, through an
	// STTUpdateSettingsFrame. A change to SpeakerFormat takes effect at once; a
	// change to any other reopens the session, since Speechmatics is told all
	// of it when a session opens.

	// Language is the transcription language; empty uses English. A regional
	// English (en-GB, en-US, en-AU) also sets the output locale.
	Language language.Language
	// Domain requests a language pack optimized for a domain, e.g. "finance";
	// empty omits it.
	Domain string
	// TurnDetectionMode is how turns are closed; empty uses
	// TurnDetectionExternal.
	TurnDetectionMode TurnDetectionMode `validate:"omitempty,oneof=vad external"`
	// SpeakerFormat labels diarized transcripts with their speaker. It uses
	// {speaker} for the speaker label, {text} for the transcript text, and
	// {speaker_number} for the speaker's order of first appearance in the
	// session (1, 2, ...), e.g. "{speaker}: {text}". Empty leaves the text as
	// it is. It is applied locally, so changing it does not reopen the session.
	SpeakerFormat string
	// KnownSpeakers are labels and identifiers of speakers seen in earlier
	// sessions, so their words are attributed to those labels. They take
	// effect with diarization on.
	KnownSpeakers []SpeakerIdentifier
	// AdditionalVocab are words to bias the engine towards.
	AdditionalVocab []AdditionalVocabEntry
	// Model is the transcription model; empty uses "linden-1".
	Model string
	// EnablePartials includes partial segments in the output, as interim
	// transcripts; nil leaves it to Speechmatics.
	EnablePartials *bool
	// EmitSentences closes a segment at every end of sentence, not only at the
	// end of the turn, so each sentence is its own final transcript; nil leaves
	// it to Speechmatics, which closes segments at the end of the turn.
	EmitSentences *bool
	// PunctuationOverrides overrides the punctuation the engine produces, as
	// Speechmatics' punctuation_overrides object; nil omits it.
	PunctuationOverrides map[string]any
	// TranscriptFilteringConfig post-processes the transcript, as Speechmatics'
	// transcript_filtering_config object: remove_disfluencies drops fillers
	// such as "um", replacements rewrites matched words. Nil omits it.
	TranscriptFilteringConfig map[string]any
	// EnableDiarization has the engine tell speakers apart and attribute words
	// to each. Each transcript then carries the Speechmatics speaker label
	// ("S1", or a label given in KnownSpeakers) as its speaker.
	EnableDiarization *bool
	// SpeakerSensitivity is the diarization sensitivity; higher helps when
	// speakers sound alike. Nil omits it.
	SpeakerSensitivity *float64
	// MaxSpeakers forces the engine to cluster words into at most this many
	// speakers. Nil omits it.
	MaxSpeakers *int
	// PreferCurrentSpeaker gives words close together extra weight to be the
	// same speaker. Nil omits it.
	PreferCurrentSpeaker *bool

	// OperatingPoint is an alias for Model. If both are given they must name
	// the same model.
	//
	// Deprecated: use Model.
	OperatingPoint string
	// MaxDelay has no effect: the Agent STT service takes no delay setting.
	//
	// Deprecated: no replacement.
	MaxDelay float64
	// EndOfUtteranceSilence has no effect: turns are closed as
	// TurnDetectionMode says.
	//
	// Deprecated: use TurnDetectionMode.
	EndOfUtteranceSilence *float64

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.SpeechmaticsTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	_, err := resolveModel(c.Model, c.OperatingPoint)
	return err
}

// resolveModel resolves the transcription model from model and the deprecated
// operatingPoint, either of which may be empty. If both are given they must
// match; if only one is given it wins; if neither, the default model is used.
func resolveModel(model, operatingPoint string) (string, error) {
	if model != "" && operatingPoint != "" && model != operatingPoint {
		return "", fmt.Errorf("%w: model (%q) and operating_point (%q) differ; pass only model, "+
			"operating_point is deprecated", errModelConflict, model, operatingPoint)
	}
	if model == "" && operatingPoint != "" {
		slog.Warn("speechmatics stt: operating_point is deprecated, use model instead")
	}
	switch {
	case model != "":
		return model, nil
	case operatingPoint != "":
		return operatingPoint, nil
	default:
		return defaultModel, nil
	}
}
