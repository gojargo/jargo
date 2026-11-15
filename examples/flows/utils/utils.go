// Package utils holds the helper the flows examples share, which keeps them
// focused on the flow itself.
//
// CreateLLM builds the LLM service an example runs, chosen by LLM_PROVIDER:
//
//   - openai: OpenAI Chat Completions, needs OPENAI_API_KEY.
//   - openai_responses: OpenAI Responses, needs OPENAI_API_KEY. The default.
//   - anthropic: needs ANTHROPIC_API_KEY.
//   - google: Gemini, needs GOOGLE_API_KEY.
//   - aws: Bedrock, through the default AWS credential chain (SSO, environment
//     variables or IAM roles). AWS_REGION picks the region, us-west-2 by
//     default.
package utils

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/gojargo/jargo/flows"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/aws/bedrock"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/openai/responses"
	"github.com/gojargo/jargo/service/llm"
)

var (
	// ErrUnsupportedProvider reports an LLM_PROVIDER no service is configured
	// for.
	ErrUnsupportedProvider = errors.New("unsupported LLM provider")
	// ErrMissingAPIKey reports a provider whose API key is not set.
	ErrMissingAPIKey = errors.New("missing API key")
)

// LLM is what the examples need of the service CreateLLM builds: a pipeline
// processor, somewhere to register tools, and a one-shot inference for the
// flow's conversation summaries.
type LLM interface {
	processor.Processor
	flows.Inferencer
	RegisterFunction(name string, h llm.FunctionCallHandler, opts ...llm.RegisterOption)
}

// providerConfig is how one provider is built.
type providerConfig struct {
	name         string
	keyVar       string
	defaultModel string
	region       string
	build        func(ctx context.Context, apiKey, model, region string) (LLM, error)
}

// providers are the supported providers, in the order they are listed.
//
//nolint:gochecknoglobals // the fixed provider table
var providers = []providerConfig{
	{
		name: "openai", keyVar: "OPENAI_API_KEY", defaultModel: "gpt-4.1",
		build: func(_ context.Context, apiKey, model, _ string) (LLM, error) {
			return chat.NewLLM(chat.LLMConfig{APIKey: apiKey, Model: model}), nil
		},
	},
	{
		name: defaultProvider, keyVar: "OPENAI_API_KEY", defaultModel: "gpt-4.1",
		build: func(_ context.Context, apiKey, model, _ string) (LLM, error) {
			return responses.NewLLM(responses.Config{APIKey: apiKey, Model: model}), nil
		},
	},
	{
		name: "anthropic", keyVar: "ANTHROPIC_API_KEY", defaultModel: "claude-sonnet-4-6",
		build: func(_ context.Context, apiKey, model, _ string) (LLM, error) {
			return anthropic.NewLLM(anthropic.Config{APIKey: apiKey, Model: model}), nil
		},
	},
	{
		name: "google", keyVar: "GOOGLE_API_KEY", defaultModel: "gemini-3.6-flash",
		build: func(_ context.Context, apiKey, model, _ string) (LLM, error) {
			return gemini.NewLLM(gemini.Config{APIKey: apiKey, Model: model}), nil
		},
	},
	{
		// AWS uses the default credential chain, so it has no API key.
		name: awsProvider, defaultModel: "us.anthropic.claude-sonnet-4-6", region: "us-west-2",
		build: func(ctx context.Context, _, model, region string) (LLM, error) {
			return bedrock.NewLLM(ctx, bedrock.Config{
				Model:       model,
				Region:      region,
				Temperature: new(0.8),
			})
		},
	},
}

const (
	// defaultProvider is the provider used when LLM_PROVIDER is not set.
	defaultProvider = "openai_responses"
	// awsProvider is the provider that authenticates through the AWS
	// credential chain rather than an API key.
	awsProvider = "aws"
)

// CreateLLM builds an LLM service from the environment.
//
// provider names the provider; empty reads LLM_PROVIDER, which defaults to
// openai_responses. model names the model; empty uses the provider's default.
//
//	model, err := utils.CreateLLM(ctx, "", "")            // LLM_PROVIDER, or OpenAI Responses
//	model, err := utils.CreateLLM(ctx, "anthropic", "")   // a specific provider
//	model, err := utils.CreateLLM(ctx, "openai", "gpt-6") // a specific provider and model
func CreateLLM(ctx context.Context, provider, model string) (LLM, error) {
	if provider == "" {
		var ok bool
		if provider, ok = os.LookupEnv("LLM_PROVIDER"); !ok {
			provider = defaultProvider
		}
	}
	provider = strings.ToLower(provider)

	idx := slices.IndexFunc(providers, func(p providerConfig) bool { return p.name == provider })
	if idx < 0 {
		names := make([]string, 0, len(providers))
		for _, p := range providers {
			names = append(names, p.name)
		}
		return nil, fmt.Errorf("%w: %s (available: %s)", ErrUnsupportedProvider, provider, strings.Join(names, ", "))
	}
	cfg := providers[idx]

	// AWS uses the default credential chain, so there is no key to read.
	var apiKey string
	if provider != awsProvider && cfg.keyVar != "" {
		apiKey = os.Getenv(cfg.keyVar)
		if apiKey == "" {
			return nil, fmt.Errorf("%w: %s for provider: %s", ErrMissingAPIKey, cfg.keyVar, provider)
		}
	}

	// Use the model given, or the provider's default.
	selectedModel := model
	if selectedModel == "" {
		selectedModel = cfg.defaultModel
	}

	region := cfg.region
	if provider == awsProvider {
		if r, ok := os.LookupEnv("AWS_REGION"); ok {
			region = r
		}
	}

	return cfg.build(ctx, apiKey, selectedModel, region)
}
