package bedrock

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/gojargo/jargo/provider/anthropic"
)

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
// SDK option instead. The interleaved-thinking beta is left off: Bedrock may
// reject a beta the model does not support.
func (c Config) anthropicConfig() anthropic.Config {
	return anthropic.Config{
		Model:                       c.Model,
		MaxTokens:                   c.MaxTokens,
		Temperature:                 c.Temperature,
		TopP:                        c.TopP,
		TopK:                        c.TopK,
		RequestTimeout:              c.RequestTimeout,
		EnablePromptCaching:         c.EnablePromptCaching,
		OmitInterleavedThinkingBeta: true,
	}
}

// NewLLM builds a Bedrock-backed Anthropic LLM service. It loads AWS
// configuration (which may read the environment, shared config files or
// instance metadata), so it takes a context and can fail.
func NewLLM(ctx context.Context, cfg Config) (*anthropic.LLMService, error) {
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, cfg.loadOptions()...)
	if err != nil {
		return nil, err
	}
	return newLLM(cfg, awsCfg), nil
}

// newLLM builds the service from a loaded AWS config. extra is appended after
// the Bedrock option, which is how a test points the service at a stub.
func newLLM(cfg Config, awsCfg aws.Config, extra ...option.RequestOption) *anthropic.LLMService {
	opts := append([]option.RequestOption{bedrock.WithConfig(awsCfg)}, extra...)
	return anthropic.NewLLMWithOptions("BedrockLLM", cfg.anthropicConfig(), opts...)
}
