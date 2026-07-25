// Package novasonic is a speech-to-speech service built on Amazon Nova Sonic,
// served over Bedrock's bidirectional streaming API. A single SigV4-signed
// HTTP/2 duplex stream carries the conversation: input audio streams up and the
// model streams its spoken reply, its transcript, and the user's transcript
// back down. The AWS SDK handles signing, HTTP/2 and event-stream framing; this
// service drives the Nova Sonic event protocol on top.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output. Nova Sonic takes 16 kHz mono PCM in and returns 24 kHz mono
// PCM out, so run the transport input at 16000 and output at 24000.
package novasonic

import (
	"github.com/gojargo/jargo/internal/validate"
)

const (
	defaultModel     = "amazon.nova-sonic-v1:0"
	defaultVoice     = "matthew"
	inputSampleRate  = 16000
	outputSampleRate = 24000

	keyPromptName  = "promptName"
	keyContentName = "contentName"
	keyMediaType   = "mediaType"
)

// Config configures the Nova Sonic service.
type Config struct {
	// Region is the AWS region (Nova Sonic launched in us-east-1); empty uses the
	// default chain (AWS_REGION, shared config).
	Region string
	// AccessKeyID and SecretAccessKey set static credentials; leave both empty to
	// use the default AWS credential chain (environment, shared config, IAM role).
	AccessKeyID     string
	SecretAccessKey string
	// SessionToken is the optional session token for temporary credentials.
	SessionToken string
	// Model is the Nova Sonic model id; empty uses a current default.
	Model string
	// Voice is the output voice id (e.g. matthew, tiffany, ambre, florian); empty
	// uses a default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// MaxTokens caps the model's response length; 0 leaves the API default.
	MaxTokens int
	// Temperature and TopP are optional sampling controls; nil leaves the API
	// default in place.
	Temperature *float64
	TopP        *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
