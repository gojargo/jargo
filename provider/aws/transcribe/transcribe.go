// Package transcribe is a streaming speech-to-text service backed by Amazon
// Transcribe's StartStreamTranscription API. Audio streams up as AudioEvents
// over a bidirectional event stream and transcripts come back as
// TranscriptEvents, surfaced as interim and final results. Transcribe finalizes
// per audio segment rather than per turn, so it works best behind a turn
// detector. Credentials and region come from the standard AWS chain
// (environment, shared config, or an IAM role) unless set explicitly.
package transcribe

import (
	"time"

	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultLanguage    = "en-US"
	defaultStability   = "high"
	fallbackSampleRate = 16000
)

// Config configures the Amazon Transcribe streaming STT service.
type Config struct {
	// Region is the AWS region (e.g. us-east-1); empty uses the default chain
	// (AWS_REGION, shared config).
	Region string
	// AccessKeyID and SecretAccessKey set static credentials; leave both empty to
	// use the default AWS credential chain (environment, shared config, IAM role).
	AccessKeyID     string
	SecretAccessKey string
	// SessionToken is the optional session token for temporary credentials.
	SessionToken string
	// Language is the BCP-47 language code of the spoken audio (e.g. en-US,
	// fr-FR); empty uses en-US.
	Language string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	// Transcribe supports 8000 or 16000 Hz; other values are clamped to 16000.
	SampleRate int
	// PartialResultsStabilization enables partial-result stabilization, trading a
	// little accuracy for steadier, lower-latency interim results; nil enables it.
	PartialResultsStabilization *bool
	// PartialResultsStability sets the stabilization level when it is on; empty
	// uses "high".
	PartialResultsStability string `validate:"omitempty,oneof=high medium low"`
	// VocabularyName selects a custom vocabulary; empty omits it.
	VocabularyName string
	// VocabularyFilterName selects a custom vocabulary filter; empty omits it.
	VocabularyFilterName string

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.AWSTranscribeTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
