// Package polly is a text-to-speech service backed by Amazon Polly's
// SynthesizeSpeech API. It requests signed 16-bit mono PCM at 8 or 16 kHz
// (Polly caps PCM at 16 kHz) and streams the returned audio downstream.
// Credentials and region come from the standard AWS chain (environment, shared
// config, or an IAM role) unless set explicitly.
package polly

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultVoice      = "Joanna"
	defaultLanguage   = "en-US"
	defaultSampleRate = 16000
	// readChunk is the size of each audio read from the synthesis stream.
	readChunk = 8192
)

// Config configures the Amazon Polly TTS service.
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
	// Voice is the Polly voice id (e.g. Joanna, Matthew); empty uses a default.
	Voice string
	// Engine selects the synthesis engine; empty leaves Polly's default in place.
	Engine string `validate:"omitempty,oneof=standard neural long-form generative"`
	// Language is the BCP-47 language tag applied to the synthesized text
	// (e.g. en-US, fr-FR); empty uses en-US.
	Language string
	// SampleRate is the PCM rate requested from Polly and emitted downstream;
	// Polly supports 8000 or 16000 for PCM. 0 uses 16 kHz.
	SampleRate int `validate:"omitempty,oneof=8000 16000"`
	// Pitch adjusts voice pitch (standard engine only); empty omits it.
	Pitch string
	// Rate adjusts speech rate; empty omits it.
	Rate string
	// Volume adjusts voice volume; empty omits it.
	Volume string
	// LexiconNames lists pronunciation lexicons to apply during synthesis.
	LexiconNames []string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
