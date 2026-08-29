package anthropic

import (
	"context"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/gojargo/jargo/adapter"
	anthropicadapter "github.com/gojargo/jargo/adapter/anthropic"
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
	// adapter converts the conversation into the request Anthropic takes.
	adapter anthropicadapter.Adapter
	// cachePrompt gates the ephemeral cache breakpoints on the prompt.
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
		display := sdk.ThinkingConfigAdaptiveDisplay(cfg.Thinking.Display)
		switch cfg.Thinking.Type {
		case "disabled":
			s.thinking = sdk.ThinkingConfigParamUnion{OfDisabled: &sdk.ThinkingConfigDisabledParam{}}
			s.thinkingSet = true
		case "adaptive":
			s.thinking = sdk.ThinkingConfigParamUnion{
				OfAdaptive: &sdk.ThinkingConfigAdaptiveParam{Display: display},
			}
			s.thinkingSet = true
		case "enabled":
			enabled := sdk.ThinkingConfigParamOfEnabled(int64(cfg.Thinking.BudgetTokens))
			enabled.OfEnabled.Display = sdk.ThinkingConfigEnabledDisplay(cfg.Thinking.Display)
			s.thinking = enabled
			s.thinkingSet = true
		}
	} else if sonnetThinksByDefault(s.model) {
		// Sonnet 5 and later think adaptively whenever a request omits the
		// parameter, which for real-time voice can add seconds before the first
		// answer token. Only the Sonnet line, Anthropic's speed tier: Opus and
		// Fable are left at the provider default, since choosing one of those is
		// a decision to reason.
		s.thinking = sdk.ThinkingConfigParamUnion{OfDisabled: &sdk.ThinkingConfigDisabledParam{}}
		s.thinkingSet = true
	}
	s.Base = llm.New(name, s)
	s.Base.SetModel(s.model)
	return s
}

// newParams builds the request params shared by both generation paths: model,
// token cap, the converted conversation, sampling controls and the system
// prompt.
func (s *Service) newParams(
	convo *frames.LLMContext, opts adapter.Options,
) (sdk.MessageNewParams, error) {
	opts.EnablePromptCaching = s.cachePrompt
	// A model without assistant-prefill support rejects a request whose message
	// list ends with an assistant message; give it a trailing user turn.
	opts.EnsureLastMessageIsUser = !supportsPrefill(s.model)
	p, err := s.adapter.LLMInvocationParams(convo, opts)
	if err != nil {
		return sdk.MessageNewParams{}, err
	}
	params := sdk.MessageNewParams{
		Model:       s.model,
		MaxTokens:   s.maxTokens,
		Messages:    p.Messages,
		System:      p.System,
		Tools:       p.Tools,
		Temperature: s.temperature,
		TopP:        s.topP,
		TopK:        s.topK,
	}
	if s.thinkingSet {
		params.Thinking = s.thinking
	}
	return params, nil
}

// toUsage converts the SDK's per-request usage into the pipeline's token usage.
func toUsage(u sdk.Usage) frames.LLMTokenUsage {
	return frames.LLMTokenUsage{
		PromptTokens:        u.InputTokens,
		CompletionTokens:    u.OutputTokens,
		CacheReadTokens:     new(u.CacheReadInputTokens),
		CacheCreationTokens: new(u.CacheCreationInputTokens),
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
	params, err := s.newParams(convo, adapter.Options{SystemInstruction: s.SystemInstruction()})
	if err != nil {
		return err
	}
	// A text-only generation advertises no tools, so the model has nothing to
	// call and a reported call would be the provider inventing one.
	params.Tools = nil
	s.StartTTFBMetrics()
	stream := s.client.Messages.NewStreaming(ctx, params)
	for stream.Next() {
		event := stream.Current()
		if carriesModelOutput(event) {
			s.StopTTFBMetrics()
		}
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
		return llm.AsCompletionTimeout(ctx, err)
	}
	if report {
		return s.PushTokenUsage(ctx, toUsage(acc.Usage))
	}
	return nil
}

// carriesModelOutput reports whether a stream event holds output the model
// produced. The events that open the stream (message_start, ping) carry none,
// so TTFB ends at the first content block. A thinking block counts: reasoning
// is output, so it ends TTFB just as answer text would.
func carriesModelOutput(event sdk.MessageStreamEventUnion) bool {
	switch event.AsAny().(type) {
	case sdk.ContentBlockStartEvent, sdk.ContentBlockDeltaEvent:
		return true
	default:
		return false
	}
}

// GenerateWithTools streams a response that may request tool calls. It emits
// text deltas to the sink as they arrive and, once the turn completes, reports
// each tool-use block the model produced. The conversation's tools are sent on
// the request, and any tool-use / tool-result turns already in the context are
// replayed as the matching Anthropic blocks.
func (s *Service) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	params, err := s.newParams(convo, adapter.Options{SystemInstruction: s.SystemInstruction()})
	if err != nil {
		return err
	}

	var acc sdk.Message
	s.StartTTFBMetrics()
	stream := s.client.Messages.NewStreaming(ctx, params)
	for stream.Next() {
		event := stream.Current()
		if carriesModelOutput(event) {
			s.StopTTFBMetrics()
		}
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
		return llm.AsCompletionTimeout(ctx, err)
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

// RunInference answers the conversation once, off to the side of the pipeline:
// no streaming, no frames, just the text. It implements llm.Inferencer.
func (s *Service) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	params, err := s.newParams(
		convo, adapter.Options{SystemInstruction: opts.SystemInstruction},
	)
	if err != nil {
		return "", err
	}
	if opts.MaxTokens > 0 {
		params.MaxTokens = int64(opts.MaxTokens)
	}
	// An inference wants an answer, not a tool call, so the toolset is left off.
	params.Tools = nil
	msg, err := s.client.Messages.New(ctx, params)
	if err != nil {
		return "", llm.AsCompletionTimeout(ctx, err)
	}
	for _, blk := range msg.Content {
		if text := blk.Text; text != "" {
			return text, nil
		}
	}
	return "", nil
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

// LLMAdapter returns the adapter this service converts through, so the base can
// add the tools it implements itself to what every request advertises. It
// implements llm.AdapterHolder.
func (s *Service) LLMAdapter() llm.BuiltinToolHolder { return &s.adapter }

// MessagesForLogging renders the conversation as this provider will see it, for
// the generation span. It implements llm.TraceRenderer.
func (s *Service) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	return s.adapter.MessagesForLogging(convo)
}

// ToolsForLogging renders the toolset as this provider will see it, for the
// generation span. It implements llm.TraceRenderer.
func (s *Service) ToolsForLogging(schema frames.ToolsSchema) []any {
	return adapter.ToolsForLogging(&s.adapter, schema)
}
