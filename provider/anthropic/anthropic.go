// Package anthropic is a streaming LLM service backed by the Anthropic API. The
// shared LLM base brackets the response with start/end frames; this service
// streams the text deltas. It defaults to Claude Haiku for low latency and
// caches the system prompt.
package anthropic

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/llm"
)

// defaultMaxTokens keeps spoken responses short and snappy.
const defaultMaxTokens = 1024

// ThinkingConfig configures extended thinking on each request. Type is the mode:
// "enabled" (a fixed BudgetTokens, older models), "adaptive" (the model decides
// how much to think, 4.6+ models), or "disabled" (no thinking). BudgetTokens
// applies only to "enabled". Leaving Config.Thinking nil omits the parameter, so
// the model's own default applies — note adaptive thinking is on by default on
// Sonnet 5 / Opus 4.8, so set "disabled" for the lowest latency.
type ThinkingConfig struct {
	Type         string `validate:"required,oneof=enabled disabled adaptive"`
	BudgetTokens int
}

// Config configures the LLM service.
type Config struct {
	// APIKey is the Anthropic API key; empty uses the ANTHROPIC_API_KEY env var.
	APIKey string
	// BaseURL overrides the API base (e.g. a proxy or compatible gateway); empty
	// uses the SDK default.
	BaseURL string
	// Model is the model id; empty uses Claude Haiku 4.5.
	Model string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
	// Temperature, TopP and TopK are optional sampling controls. A nil value
	// leaves the API default in place; they are pointers so a deliberate zero is
	// distinguishable from "unset".
	Temperature *float64
	TopP        *float64
	TopK        *int64
	// Thinking configures extended thinking. Nil omits the parameter (model
	// default). For voice, "disabled" avoids the latency of adaptive thinking on
	// models where it is on by default.
	Thinking *ThinkingConfig `validate:"omitempty"`
	// RequestTimeout bounds a single request attempt, including the full stream;
	// 0 leaves the SDK default. Keep it generously above the expected response
	// time, since for a streaming request it caps the whole response.
	RequestTimeout time.Duration
	// MaxRetries overrides how many times a failed request is retried before the
	// stream begins (transient connection errors, 429s, 5xx); 0 leaves the SDK
	// default of two retries. Mid-stream failures are not retried.
	MaxRetries int
	// EnablePromptCaching caches the system prompt with an ephemeral cache
	// breakpoint so repeated turns reuse it; nil defaults to true (jargo caches
	// for latency). Set to false to disable caching.
	EnablePromptCaching *bool
	// Extra sets arbitrary additional top-level request-body fields not modeled
	// above (e.g. beta parameters), applied to every request.
	Extra map[string]any
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// Service is a streaming Anthropic LLM processor.
type Service struct {
	*llm.Base
	client    sdk.Client
	model     sdk.Model
	maxTokens int64
	// Sampling controls, applied to each request. A zero param.Opt is omitted
	// from the request, leaving the API default.
	temperature param.Opt[float64]
	topP        param.Opt[float64]
	topK        param.Opt[int64]
	// cachePrompt gates the ephemeral cache breakpoint on the system prompt.
	cachePrompt bool
	// thinking is the extended-thinking config sent on each request; thinkingSet
	// reports whether it was configured (an unset union means "omit").
	thinking    sdk.ThinkingConfigParamUnion
	thinkingSet bool
}

// NewLLM builds an Anthropic LLM service.
func NewLLM(cfg Config) *Service {
	return NewLLMWithOptions("AnthropicLLM", cfg)
}

// NewLLMWithOptions builds an Anthropic LLM service named name with extra SDK
// request options appended. It backs alternative Anthropic backends — such as
// Amazon Bedrock or Google Vertex — that authorize and address requests through
// an SDK option rather than an API key.
func NewLLMWithOptions(name string, cfg Config, extra ...option.RequestOption) *Service {
	var opts []option.RequestOption
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	if cfg.RequestTimeout > 0 {
		opts = append(opts, option.WithRequestTimeout(cfg.RequestTimeout))
	}
	if cfg.MaxRetries > 0 {
		opts = append(opts, option.WithMaxRetries(cfg.MaxRetries))
	}
	for k, v := range cfg.Extra {
		opts = append(opts, option.WithJSONSet(k, v))
	}
	opts = append(opts, extra...)
	s := &Service{
		client:      sdk.NewClient(opts...),
		model:       sdk.ModelClaudeHaiku4_5,
		maxTokens:   defaultMaxTokens,
		cachePrompt: true,
	}
	if cfg.Model != "" {
		s.model = cfg.Model
	}
	if cfg.EnablePromptCaching != nil {
		s.cachePrompt = *cfg.EnablePromptCaching
	}
	if cfg.MaxTokens > 0 {
		s.maxTokens = int64(cfg.MaxTokens)
	}
	if cfg.Temperature != nil {
		s.temperature = param.NewOpt(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		s.topP = param.NewOpt(*cfg.TopP)
	}
	if cfg.TopK != nil {
		s.topK = param.NewOpt(*cfg.TopK)
	}
	if cfg.Thinking != nil {
		switch cfg.Thinking.Type {
		case "disabled":
			s.thinking = sdk.ThinkingConfigParamUnion{OfDisabled: &sdk.ThinkingConfigDisabledParam{}}
			s.thinkingSet = true
		case "adaptive":
			s.thinking = sdk.ThinkingConfigParamUnion{OfAdaptive: &sdk.ThinkingConfigAdaptiveParam{}}
			s.thinkingSet = true
		case "enabled":
			s.thinking = sdk.ThinkingConfigParamOfEnabled(int64(cfg.Thinking.BudgetTokens))
			s.thinkingSet = true
		}
	}
	s.Base = llm.New(name, s)
	s.Base.SetModel(s.model)
	return s
}

// newParams builds the request params shared by both generation paths: model,
// token cap, the converted conversation, sampling controls and the cached
// system prompt.
func (s *Service) newParams(convo *frames.LLMContext) sdk.MessageNewParams {
	messages := toMessages(convo.Messages())
	// Models without assistant-prefill support reject a request whose message
	// list ends with an assistant message; give them a trailing user turn.
	if !supportsPrefill(s.model) {
		messages = ensureLastMessageIsUser(messages)
	}
	params := sdk.MessageNewParams{
		Model:       s.model,
		MaxTokens:   s.maxTokens,
		Messages:    messages,
		Temperature: s.temperature,
		TopP:        s.topP,
		TopK:        s.topK,
	}
	if s.thinkingSet {
		params.Thinking = s.thinking
	}
	if system := convo.System(); system != "" {
		block := sdk.TextBlockParam{Text: system}
		if s.cachePrompt {
			// Cache the system prompt so repeated turns reuse it.
			block.CacheControl = sdk.NewCacheControlEphemeralParam()
		}
		params.System = []sdk.TextBlockParam{block}
	}
	return params
}

// toUsage converts the SDK's per-request usage into the pipeline's token usage.
func toUsage(u sdk.Usage) frames.LLMTokenUsage {
	return frames.LLMTokenUsage{
		PromptTokens:        u.InputTokens,
		CompletionTokens:    u.OutputTokens,
		CacheReadTokens:     u.CacheReadInputTokens,
		CacheCreationTokens: u.CacheCreationInputTokens,
		TotalTokens:         u.InputTokens + u.OutputTokens,
	}
}

// Generate streams a response for the conversation, emitting each text delta.
// When usage metrics are enabled it accumulates the stream so it can report the
// turn's token usage once the response completes.
func (s *Service) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	report := s.UsageMetricsEnabled()
	var acc sdk.Message
	stream := s.client.Messages.NewStreaming(ctx, s.newParams(convo))
	for stream.Next() {
		event := stream.Current()
		if report {
			if err := acc.Accumulate(event); err != nil {
				return err
			}
		}
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
	if err := stream.Err(); err != nil {
		return err
	}
	if report {
		return s.PushTokenUsage(ctx, toUsage(acc.Usage))
	}
	return nil
}

// GenerateWithTools streams a response that may request tool calls. It emits
// text deltas to the sink as they arrive and, once the turn completes, reports
// each tool-use block the model produced. The conversation's tools are sent on
// the request, and any tool-use / tool-result turns already in the context are
// replayed as the matching Anthropic blocks.
func (s *Service) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	params := s.newParams(convo)
	if tools := convo.Tools(); len(tools) > 0 {
		params.Tools = toTools(tools)
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
	if s.UsageMetricsEnabled() {
		return s.PushTokenUsage(ctx, toUsage(acc.Usage))
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

// prefillSupportedPatterns lists the Claude models that still accept a request
// whose message list ends with an assistant message (assistant prefill).
// Anthropic dropped prefill support in the 4.6-generation models, so this is a
// frozen legacy set; any Claude model not matching it is assumed to reject
// prefill. Patterns are matched as substrings so Bedrock ids like
// "us.anthropic.claude-sonnet-4-6-v1:0" are handled alongside direct ids.
//
//nolint:gochecknoglobals // fixed lookup table
var prefillSupportedPatterns = []string{
	"claude-2",
	"claude-instant",
	"claude-3",
	"claude-opus-4-0",
	"claude-opus-4-1",
	"claude-sonnet-4-0",
	"claude-sonnet-4-5",
	"claude-haiku-4-5",
}

// supportsPrefill reports whether model accepts an assistant-prefilled request.
// Non-Claude models are unaffected and reported as supporting it (so nothing is
// injected); a Claude model is supported only when it matches the frozen legacy
// set above.
func supportsPrefill(model string) bool {
	if !strings.Contains(model, "claude") {
		return true
	}
	for _, p := range prefillSupportedPatterns {
		if strings.Contains(model, p) {
			return true
		}
	}
	return false
}

// ensureLastMessageIsUser appends a minimal user message when the list ends with
// an assistant message, so a model without prefill support accepts the request.
// The "." is a language-neutral no-op user turn; the stored context is untouched.
func ensureLastMessageIsUser(msgs []sdk.MessageParam) []sdk.MessageParam {
	if n := len(msgs); n > 0 && msgs[n-1].Role == sdk.MessageParamRoleAssistant {
		return append(msgs, sdk.NewUserMessage(sdk.NewTextBlock(".")))
	}
	return msgs
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
