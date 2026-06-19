// Package anthropic is a streaming LLM service backed by the Anthropic API. It
// consumes an LLMContextFrame and emits the response as an
// LLMFullResponseStartFrame, a stream of LLMTextFrames, and an
// LLMFullResponseEndFrame.
//
// It defaults to Claude Haiku for low latency and caches the system prompt.
package anthropic

import (
	"context"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
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
	*processor.Base
	client    sdk.Client
	model     sdk.Model
	maxTokens int64
}

// New builds an Anthropic LLM service.
func New(cfg Config) *Service {
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
	s.Base = processor.New("AnthropicLLM", s)
	return s
}

// ProcessFrame runs the LLM on each LLMContextFrame and forwards other frames.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if cf, ok := f.(*frames.LLMContextFrame); ok {
		return s.run(ctx, cf.Context)
	}
	return s.PushFrame(ctx, f, dir)
}

// run streams a response for the conversation, emitting response frames. It runs
// under the process goroutine's context, so an interruption cancels the stream.
func (s *Service) run(ctx context.Context, convo *frames.LLMContext) error {
	if err := s.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}

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
		if err := s.PushFrame(ctx, frames.NewLLMTextFrame(text.Text), processor.Downstream); err != nil {
			return err
		}
	}
	if err := stream.Err(); err != nil && ctx.Err() == nil {
		s.PushError(ctx, "anthropic streaming failed", err, false)
	}

	return s.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
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
