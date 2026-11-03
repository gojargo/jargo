// Package novasonic is a speech-to-speech service built on Amazon Nova Sonic,
// served over Bedrock's bidirectional streaming API. A single SigV4-signed
// HTTP/2 duplex stream carries the conversation: input audio streams up and the
// model streams its spoken reply, its transcript, and the user's transcript
// back down. The AWS SDK handles signing, HTTP/2 and event-stream framing; this
// service drives the Nova Sonic event protocol on top.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output, with the user and assistant aggregators on either side. By
// default Nova Sonic takes 16 kHz mono PCM in and returns 24 kHz mono PCM out,
// so run the transport input at 16000 and output at 24000, or set AudioConfig
// to match the transport.
//
// The service does not report the user's turns (UserStartedSpeakingFrame,
// UserStoppedSpeakingFrame): Nova Sonic detects them server-side and does not
// say where they fall. Processors that need them (RTVI speech events, turn
// tracking, user idle and mute strategies) need the user aggregator's own voice
// activity detection, whose turns are approximate.
//
// Sessions have an AWS-imposed time limit of about eight minutes. Session
// continuation (on by default) opens the next session in the background before
// the limit and hands the conversation over to it while the assistant speaks.
package novasonic

import (
	"errors"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultModel = "amazon.nova-2-5-sonic"
	// firstGenerationModel is the original Nova Sonic, which takes no
	// endpointing sensitivity and needs the assistant response trigger.
	firstGenerationModel = "amazon.nova-sonic-v1:0"
	defaultVoice         = "matthew"
	defaultTemperature   = 0.7
	defaultMaxTokens     = 1024
	defaultTopP          = 0.9
)

// The endpointing sensitivities: how quickly Nova Sonic decides the user has
// stopped speaking, HIGH responding most quickly.
const (
	EndpointingSensitivityLow    = "LOW"
	EndpointingSensitivityMedium = "MEDIUM"
	EndpointingSensitivityHigh   = "HIGH"
)

// errNotGenerator is returned by the generation entry point this service does
// not use: it generates continuously rather than answering a conversation.
//
//nolint:gochecknoglobals // sentinel error
var errNotGenerator = errors.New("novasonic: the model generates continuously")

// AudioConfig is the audio the session takes in and sends back. It is fixed
// for the service's lifetime.
type AudioConfig struct {
	// InputSampleRate is the user audio's sample rate in Hz.
	InputSampleRate int `validate:"min=1"`
	// InputSampleSize is the user audio's sample size in bits.
	InputSampleSize int `validate:"min=8"`
	// InputChannelCount is the user audio's channel count.
	InputChannelCount int `validate:"min=1"`
	// OutputSampleRate is the model's audio's sample rate in Hz.
	OutputSampleRate int `validate:"min=1"`
	// OutputSampleSize is the model's audio's sample size in bits.
	OutputSampleSize int `validate:"min=8"`
	// OutputChannelCount is the model's audio's channel count.
	OutputChannelCount int `validate:"min=1"`
}

// DefaultAudioConfig is 16-bit mono at 16 kHz in and 24 kHz out.
func DefaultAudioConfig() AudioConfig {
	return AudioConfig{
		InputSampleRate:    16000,
		InputSampleSize:    16,
		InputChannelCount:  1,
		OutputSampleRate:   24000,
		OutputSampleSize:   16,
		OutputChannelCount: 1,
	}
}

// Config configures the Nova Sonic service. The model, voice, instruction and
// inference fields are the starting values of its Settings.
type Config struct {
	// Region is the AWS region where the service is hosted. Supported regions:
	// us-east-1, us-west-2, eu-north-1, ap-northeast-1. Empty uses the default
	// chain (AWS_REGION, shared config).
	Region string
	// AccessKeyID and SecretAccessKey set static credentials; leave both empty to
	// use the default AWS credential chain (environment, shared config, IAM role).
	AccessKeyID     string
	SecretAccessKey string
	// SessionToken is the optional session token for temporary credentials.
	SessionToken string
	// Model is the Nova Sonic model id; empty uses amazon.nova-2-5-sonic.
	Model string
	// Voice is the output voice id (e.g. matthew, tiffany, amy); empty uses
	// matthew. Some voices are designed for use with a specific language. For
	// the available voices, see
	// https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-language-support.html.
	Voice string
	// Instructions is the system instruction for the session. It wins over a
	// system prompt on the conversation.
	Instructions string
	// Temperature is the sampling temperature; nil uses 0.7.
	Temperature *float64 `validate:"omitempty,min=0"`
	// MaxTokens caps the model's response length; nil uses 1024.
	MaxTokens *int `validate:"omitempty,min=1"`
	// TopP is the nucleus sampling parameter; nil uses 0.9.
	TopP *float64 `validate:"omitempty,min=0,max=1"`
	// EndpointingSensitivity is how quickly Nova Sonic decides the user has
	// stopped speaking: LOW, MEDIUM or HIGH, HIGH responding most quickly. Empty
	// leaves the model's own behavior. Nova 2 Sonic and later take it.
	EndpointingSensitivity string `validate:"omitempty,oneof=LOW MEDIUM HIGH"`
	// AudioConfig is the audio the session takes in and sends back; nil uses
	// DefaultAudioConfig.
	AudioConfig *AudioConfig
	// Tools are tools configured on the service itself, as an alternative to
	// advertising them through the conversation. Their handlers register on
	// their own. A conversation that brings tools of its own replaces them.
	Tools frames.ToolsSchema
	// SessionContinuation configures how sessions are rotated before the AWS
	// time limit; nil uses DefaultSessionContinuationParams, which is on.
	SessionContinuation *SessionContinuationParams
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.AudioConfig != nil {
		return validate.Struct(*c.AudioConfig)
	}
	return nil
}
