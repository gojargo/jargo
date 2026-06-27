// Package bedrock provides a streaming LLM service for Anthropic Claude models
// served through Amazon Bedrock. It reuses the Anthropic provider's streaming
// implementation, swapping API-key auth for AWS SigV4 via the Anthropic SDK's
// Bedrock backend; credentials and region come from the standard AWS chain
// (environment, shared config, or an IAM role) unless set explicitly.
package bedrock

import (
	"context"
	"time"

	"github.com/anthropics/anthropic-sdk-go/bedrock"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/anthropic"
)

// Config configures the Amazon Bedrock LLM provider.
type Config struct {
	// Model is the Bedrock model id or inference-profile id, e.g.
	// "us.anthropic.claude-3-5-haiku-20241022-v1:0". Required.
	Model string `validate:"required"`
	// Region is the AWS region (e.g. us-east-1); empty uses the default chain
	// (AWS_REGION, shared config).
	Region string
	// AccessKeyID and SecretAccessKey set static credentials; leave both empty to
	// use the default AWS credential chain (environment, shared config, IAM role).
	AccessKeyID     string
	SecretAccessKey string
	// SessionToken is the optional session token for temporary credentials.
	SessionToken string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
	// Temperature, TopP and TopK are optional sampling controls; nil leaves the
	// API default in place.
	Temperature *float64
	TopP        *float64
	TopK        *int64
	// RequestTimeout bounds a single request attempt, including the full stream;
	// 0 leaves the SDK default.
	RequestTimeout time.Duration
	// EnablePromptCaching caches the system prompt; nil defaults to true.
	EnablePromptCaching *bool
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// loadOptions builds the AWS config load options from the static fields. An
// empty Region or empty credentials fall back to the default AWS chain.
func (c Config) loadOptions() []func(*awsconfig.LoadOptions) error {
	var opts []func(*awsconfig.LoadOptions) error
	if c.Region != "" {
		opts = append(opts, awsconfig.WithRegion(c.Region))
	}
	if c.AccessKeyID != "" && c.SecretAccessKey != "" {
		opts = append(opts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(c.AccessKeyID, c.SecretAccessKey, c.SessionToken),
		))
	}
	return opts
}

// anthropicConfig maps the Bedrock config onto the Anthropic provider's config.
// APIKey and BaseURL are left empty: auth and addressing come from the Bedrock
// SDK option instead.
func (c Config) anthropicConfig() anthropic.Config {
	return anthropic.Config{
		Model:               c.Model,
		MaxTokens:           c.MaxTokens,
		Temperature:         c.Temperature,
		TopP:                c.TopP,
		TopK:                c.TopK,
		RequestTimeout:      c.RequestTimeout,
		EnablePromptCaching: c.EnablePromptCaching,
	}
}

// NewLLM builds a Bedrock-backed Anthropic LLM service. It loads AWS
// configuration (which may read the environment, shared config files or
// instance metadata), so it takes a context and can fail.
func NewLLM(ctx context.Context, cfg Config) (*anthropic.Service, error) {
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, cfg.loadOptions()...)
	if err != nil {
		return nil, err
	}
	return anthropic.NewLLMWithOptions("BedrockLLM", cfg.anthropicConfig(), bedrock.WithConfig(awsCfg)), nil
}
