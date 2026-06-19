// Package anthropic is a streaming LLM service backed by the Anthropic API. The
// shared LLM base brackets the response with start/end frames; this service
// streams the text deltas. It defaults to Claude Haiku for low latency and
// caches the system prompt.
package anthropic

import (
	"context"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// defaultMaxTokens keeps spoken responses short and snappy.
const defaultMaxTokens = 1024

// Config configures the LLM service.
type Config struct {
	// APIKey is the Anthropic API key; empty uses the ANTHROPIC_API_KEY env var.
	APIKey string
	// Model is the model id; empty uses Claude Haiku 4.5.
	Model string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
}

// Service is a streaming Anthropic LLM processor.
type Service struct {
	*llm.Base
	client    sdk.Client
	model     sdk.Model
	maxTokens int64
}

// NewLLM builds an Anthropic LLM service.
func NewLLM(cfg Config) *Service {
	var opts []option.RequestOption
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	s := &Service{
		client:    sdk.NewClient(opts...),
		model:     sdk.ModelClaudeHaiku4_5,
		maxTokens: defaultMaxTokens,
	}
	if cfg.Model != "" {
		s.model = cfg.Model
	}
	if cfg.MaxTokens > 0 {
		s.maxTokens = int64(cfg.MaxTokens)
	}
	s.Base = llm.New("AnthropicLLM", s)
	return s
}

// Generate streams a response for the conversation, emitting each text delta.
func (s *Service) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	params := sdk.MessageNewParams{
		Model:     s.model,
		MaxTokens: s.maxTokens,
		Messages:  toMessages(convo.Messages()),
	}
	if system := convo.System(); system != "" {
		// Cache the system prompt so repeated turns reuse it.
		params.System = []sdk.TextBlockParam{{
			Text:         system,
			CacheControl: sdk.NewCacheControlEphemeralParam(),
		}}
	}

	stream := s.client.Messages.NewStreaming(ctx, params)
	for stream.Next() {
		event := stream.Current()
		delta, ok := event.AsAny().(sdk.ContentBlockDeltaEvent)
		if !ok {
			continue
		}
		text, ok := delta.Delta.AsAny().(sdk.TextDelta)
		if !ok || text.Text == "" {
			continue
		}
		if err := emit(text.Text); err != nil {
			return err
		}
	}
	return stream.Err()
}

// toMessages converts the conversation into Anthropic message params.
func toMessages(msgs []frames.Message) []sdk.MessageParam {
	out := make([]sdk.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case frames.RoleUser:
			out = append(out, sdk.NewUserMessage(sdk.NewTextBlock(m.Text)))
		case frames.RoleAssistant:
			out = append(out, sdk.NewAssistantMessage(sdk.NewTextBlock(m.Text)))
		default:
			// The system prompt is sent separately, not as a message.
		}
	}
	return out
}
