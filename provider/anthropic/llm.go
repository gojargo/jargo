package anthropic

import (
	"context"
	"encoding/json"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

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

// requestOptions builds the SDK client options for cfg, with extra appended.
func requestOptions(cfg Config, extra ...option.RequestOption) []option.RequestOption {
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
	return append(opts, extra...)
}

// NewLLMWithOptions builds an Anthropic LLM service named name with extra SDK
// request options appended. It backs alternative Anthropic backends — such as
// Amazon Bedrock or Google Vertex — that authorize and address requests through
// an SDK option rather than an API key.
func NewLLMWithOptions(name string, cfg Config, extra ...option.RequestOption) *Service {
	s := &Service{
		client:      sdk.NewClient(requestOptions(cfg, extra...)...),
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
	params.System = s.systemBlocks(convo)
	return params
}

// systemBlocks renders the system prompt, with the cache breakpoint on the part
// of it that survives between turns.
//
// A cached prefix is only reused while it stays byte-identical, and everything
// after the breakpoint is free to vary without disturbing it. The recalled
// context a memory service refreshes every turn is exactly that: putting it
// inside the breakpoint rewrites the cache on every request and never reads one
// back, which costs more than not caching at all.
func (s *Service) systemBlocks(convo *frames.LLMContext) []sdk.TextBlockParam {
	if !s.cachePrompt {
		system := convo.System()
		if system == "" {
			return nil
		}
		return []sdk.TextBlockParam{{Text: system}}
	}

	stable, volatile := convo.SystemParts()
	blocks := make([]sdk.TextBlockParam, 0, 2)
	if stable != "" {
		blocks = append(blocks, sdk.TextBlockParam{
			Text:         stable,
			CacheControl: sdk.NewCacheControlEphemeralParam(),
		})
	}
	if volatile != "" {
		blocks = append(blocks, sdk.TextBlockParam{Text: volatile})
	}
	if len(blocks) == 0 {
		return nil
	}
	return blocks
}

// toUsage converts the SDK's per-request usage into the pipeline's token usage.
func toUsage(u sdk.Usage) frames.LLMTokenUsage {
	return frames.LLMTokenUsage{
		PromptTokens:        u.InputTokens,
		CompletionTokens:    u.OutputTokens,
		CacheReadTokens:     u.CacheReadInputTokens,
		CacheCreationTokens: u.CacheCreationInputTokens,
		// The input count is reported net of the cache, so the cached tokens are
		// added back: the total stays the gross figure, comparable with a
		// service whose provider supplies it that way already.
		TotalTokens: u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens + u.OutputTokens,
	}
}

// Generate streams a response for the conversation, emitting each text delta.
// When usage metrics are enabled it accumulates the stream so it can report the
// turn's token usage once the response completes.
func (s *Service) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	report := s.UsageMetricsEnabled()
	var acc sdk.Message
	s.StartTTFBMetrics()
	stream := s.client.Messages.NewStreaming(ctx, s.newParams(convo))
	s.StopTTFBMetrics()
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
	s.StartTTFBMetrics()
	stream := s.client.Messages.NewStreaming(ctx, params)
	s.StopTTFBMetrics()
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
