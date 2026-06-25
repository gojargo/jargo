// Package anthropic is a streaming LLM service backed by the Anthropic API. The
// shared LLM base brackets the response with start/end frames; this service
// streams the text deltas. It defaults to Claude Haiku for low latency and
// caches the system prompt.
package anthropic

import (
	"context"
	"encoding/json"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
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

// GenerateWithTools streams a response that may request tool calls. It emits
// text deltas to the sink as they arrive and, once the turn completes, reports
// each tool-use block the model produced. The conversation's tools are sent on
// the request, and any tool-use / tool-result turns already in the context are
// replayed as the matching Anthropic blocks.
func (s *Service) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	params := sdk.MessageNewParams{
		Model:     s.model,
		MaxTokens: s.maxTokens,
		Messages:  toMessages(convo.Messages()),
	}
	if tools := convo.Tools(); len(tools) > 0 {
		params.Tools = toTools(tools)
	}
	if system := convo.System(); system != "" {
		params.System = []sdk.TextBlockParam{{
			Text:         system,
			CacheControl: sdk.NewCacheControlEphemeralParam(),
		}}
	}

	var acc sdk.Message
	stream := s.client.Messages.NewStreaming(ctx, params)
	for stream.Next() {
		event := stream.Current()
		if err := acc.Accumulate(event); err != nil {
			return err
		}
		// Emit text deltas live so first-token latency is unaffected by the
		// tool-use blocks accumulated for the post-stream harvest below.
		if delta, ok := event.AsAny().(sdk.ContentBlockDeltaEvent); ok {
			if text, ok := delta.Delta.AsAny().(sdk.TextDelta); ok && text.Text != "" {
				if err := sink.Text(text.Text); err != nil {
					return err
				}
			}
		}
	}
	if err := stream.Err(); err != nil {
		return err
	}
	for _, blk := range acc.Content {
		if blk.Type == "tool_use" {
			if err := sink.Tool(frames.ToolCall{ID: blk.ID, Name: blk.Name, Args: blk.Input}); err != nil {
				return err
			}
		}
	}
	return nil
}

// toTools converts the context's tools into Anthropic tool params. Each tool's
// Parameters JSON-Schema object supplies the input schema's properties and
// required fields.
func toTools(tools []frames.Tool) []sdk.ToolUnionParam {
	out := make([]sdk.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		var schema struct {
			Properties json.RawMessage `json:"properties"`
			Required   []string        `json:"required"`
		}
		if len(t.Parameters) > 0 {
			_ = json.Unmarshal(t.Parameters, &schema)
		}
		tool := &sdk.ToolParam{
			Name:        t.Name,
			InputSchema: sdk.ToolInputSchemaParam{Required: schema.Required},
		}
		if t.Description != "" {
			tool.Description = param.NewOpt(t.Description)
		}
		if len(schema.Properties) > 0 {
			tool.InputSchema.Properties = schema.Properties
		}
		out = append(out, sdk.ToolUnionParam{OfTool: tool})
	}
	return out
}

// toMessages converts the conversation into Anthropic message params. Tool turns
// become the assistant(tool_use) and user(tool_result) blocks the API requires.
func toMessages(msgs []frames.Message) []sdk.MessageParam {
	out := make([]sdk.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		switch {
		case len(m.ToolResults) > 0:
			blocks := make([]sdk.ContentBlockParamUnion, 0, len(m.ToolResults))
			for _, r := range m.ToolResults {
				blocks = append(blocks, sdk.NewToolResultBlock(r.ID, r.Content, r.IsError))
			}
			out = append(out, sdk.NewUserMessage(blocks...))
		case len(m.ToolCalls) > 0:
			blocks := make([]sdk.ContentBlockParamUnion, 0, len(m.ToolCalls)+1)
			if m.Text != "" {
				blocks = append(blocks, sdk.NewTextBlock(m.Text))
			}
			for _, c := range m.ToolCalls {
				input := any(c.Args)
				if len(c.Args) == 0 {
					input = json.RawMessage("{}")
				}
				blocks = append(blocks, sdk.NewToolUseBlock(c.ID, input, c.Name))
			}
			out = append(out, sdk.NewAssistantMessage(blocks...))
		case m.Role == frames.RoleUser:
			out = append(out, sdk.NewUserMessage(sdk.NewTextBlock(m.Text)))
		case m.Role == frames.RoleAssistant:
			out = append(out, sdk.NewAssistantMessage(sdk.NewTextBlock(m.Text)))
		default:
			// The system prompt is sent separately, not as a message.
		}
	}
	return out
}
