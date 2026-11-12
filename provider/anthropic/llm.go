package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/gojargo/jargo/adapter"
	anthropicadapter "github.com/gojargo/jargo/adapter/anthropic"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
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
	// systemPromptCacheTTL is how long the system prompt's cache entry lives;
	// empty leaves Anthropic's default.
	systemPromptCacheTTL string
	// thinking is the extended-thinking config sent on each request; thinkingSet
	// reports whether it was configured (an unset union means "omit").
	thinking    sdk.ThinkingConfigParamUnion
	thinkingSet bool
	// effort is the output_config.effort sent on each request; empty omits it.
	effort string
	// extraConfiguresThinking reports whether Extra sets thinking or an effort,
	// which, like the configured ones, wins over the thinking default.
	extraConfiguresThinking bool
	// omitInterleavedThinking leaves the interleaved-thinking beta off every
	// request, for a backend that does not take it.
	omitInterleavedThinking bool
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

		omitInterleavedThinking: cfg.OmitInterleavedThinkingBeta,
	}
	if cfg.Model != "" {
		s.model = cfg.Model
	}
	if cfg.EnablePromptCaching != nil {
		s.cachePrompt = *cfg.EnablePromptCaching
	}
	s.systemPromptCacheTTL = cfg.SystemPromptCacheTTL
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
		case thinkingDisabled:
			s.thinking = sdk.ThinkingConfigParamUnion{OfDisabled: &sdk.ThinkingConfigDisabledParam{}}
			s.thinkingSet = true
		case thinkingBetweenTools:
			s.thinking = sdk.ThinkingConfigParamUnion{OfBetweenTools: &sdk.ThinkingConfigBetweenToolsParam{}}
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
	}
	s.effort = cfg.Effort
	s.extraConfiguresThinking = extraConfiguresThinking(cfg.Extra)
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
	opts.SystemPromptCacheTTL = s.systemPromptCacheTTL
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
	if s.effort != "" {
		params.OutputConfig.Effort = sdk.OutputConfigEffort(s.effort)
	}
	// Applied last, so an explicit thinking or effort setting, from the config
	// or from Extra, wins over the low-latency default.
	s.applyThinkingDefault(&params)
	return params, nil
}

// extraConfiguresThinking reports whether the extra request fields set thinking
// or an effort, either of which decides how much the model thinks.
func extraConfiguresThinking(extra map[string]any) bool {
	if _, ok := extra["thinking"]; ok {
		return true
	}
	if _, ok := extra["output_config.effort"]; ok {
		return true
	}
	outputConfig, ok := extra["output_config"].(map[string]any)
	if !ok {
		return false
	}
	_, ok = outputConfig["effort"]
	return ok
}

// applyThinkingDefault sends the model its thinking default when no reasoning
// is configured.
//
// It applies the model's entry in thinkingDefaults, if it has one. A request
// that already sets thinking or an effort, from the config or from Extra, is
// left as it is, since the two together decide how much the model thinks. So is
// every request when UseProviderThinkingDefault is set. The first request for
// each model logs the setting sent, or that the model runs at Anthropic's
// default, warning for an unlisted Claude model.
func (s *Service) applyThinkingDefault(params *sdk.MessageNewParams) {
	if s.thinkingSet || s.effort != "" || s.extraConfiguresThinking {
		return
	}
	name := modelName(s.model)
	var setting string
	if d, ok := thinkingDefaults[name]; ok && !s.UseProviderThinkingDefault() {
		switch {
		case d.effort != "":
			params.OutputConfig.Effort = sdk.OutputConfigEffort(d.effort)
			setting = fmt.Sprintf(`output_config={"effort":%q}`, d.effort)
		case d.thinking == thinkingBetweenTools:
			params.Thinking = sdk.ThinkingConfigParamUnion{OfBetweenTools: &sdk.ThinkingConfigBetweenToolsParam{}}
			setting = fmt.Sprintf(`thinking={"type":%q}`, d.thinking)
		default:
			params.Thinking = sdk.ThinkingConfigParamUnion{OfDisabled: &sdk.ThinkingConfigDisabledParam{}}
			setting = fmt.Sprintf(`thinking={"type":%q}`, d.thinking)
		}
	}
	s.ReportThinkingDefault(s.model, setting, strings.HasPrefix(name, "claude-"),
		"Anthropic", "Thinking or Effort in anthropic.Config")
}

// reportUsage reports a response's usage, unless it counted no tokens at all:
// an all-zero report says nothing about what the response cost.
func (s *Service) reportUsage(ctx context.Context, u sdk.Usage) error {
	if u.InputTokens == 0 && u.OutputTokens == 0 && u.CacheCreationInputTokens == 0 && u.CacheReadInputTokens == 0 {
		return nil
	}
	return s.PushTokenUsage(ctx, toUsage(u))
}

// toUsage converts the SDK's per-request usage into the pipeline's token usage.
func toUsage(u sdk.Usage) frames.LLMTokenUsage {
	out := frames.LLMTokenUsage{
		PromptTokens:        u.InputTokens,
		CompletionTokens:    u.OutputTokens,
		CacheReadTokens:     new(u.CacheReadInputTokens),
		CacheCreationTokens: new(u.CacheCreationInputTokens),
		// The input count is reported net of the cache, so the cached tokens are
		// added back: the total stays the gross figure, comparable with a
		// service whose provider supplies it that way already.
		TotalTokens: u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens + u.OutputTokens,
	}
	// Thinking tokens are part of the output count, reported apart when the
	// response says how many there were.
	if u.OutputTokensDetails.JSON.ThinkingTokens.Valid() {
		out.ReasoningTokens = new(u.OutputTokensDetails.ThinkingTokens)
	}
	return out
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
	stream := s.client.Messages.NewStreaming(ctx, params, s.requestOpts()...)
	for stream.Next() {
		event := stream.Current()
		if carriesModelOutput(event) {
			s.StopTTFBMetrics(ctx)
		}
		if report {
			if err := acc.Accumulate(event); err != nil {
				return err
			}
		}
		if err := s.pushThought(ctx, event); err != nil {
			return err
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
		return s.reportUsage(ctx, acc.Usage)
	}
	return nil
}

// pushThought pushes the thought frames a stream event carries, if any.
//
// A thinking block opens a thought, each thinking delta is a chunk of its text,
// and the signature delta Anthropic sends just before the block closes ends it,
// carrying the signature with it. The thought is asked back into the
// conversation as this provider's own message: a thinking block has to be
// handed back on the next turn, signature intact, for the model to carry its
// reasoning across a tool call.
func (s *Service) pushThought(ctx context.Context, event sdk.MessageStreamEventUnion) error {
	var f frames.Frame
	switch ev := event.AsAny().(type) {
	case sdk.ContentBlockStartEvent:
		if ev.ContentBlock.Type != "thinking" {
			return nil
		}
		start := frames.NewLLMThoughtStartFrame()
		start.AppendToContext = true
		start.LLM = s.adapter.IDForLLMSpecificMessages()
		f = start
	case sdk.ContentBlockDeltaEvent:
		switch d := ev.Delta.AsAny().(type) {
		case sdk.ThinkingDelta:
			f = frames.NewLLMThoughtTextFrame(d.Thinking)
		case sdk.SignatureDelta:
			end := frames.NewLLMThoughtEndFrame()
			end.Signature = d.Signature
			f = end
		default:
			return nil
		}
	default:
		return nil
	}
	return s.PushFrame(ctx, f, processor.Downstream)
}

// interleavedThinkingBeta is the Anthropic beta that lets a model think between
// the tool calls of one turn, rather than only before the first of them.
const interleavedThinkingBeta = "interleaved-thinking-2025-05-14"

// interleavedThinking opts a request into interleaved thinking. Every request
// carries it, whether or not thinking is configured: it changes nothing for a
// model that does not think, and a model that thinks by default needs it to
// keep reasoning across a sequence of tool calls. It is added to any beta the
// caller already asked for rather than replacing it.
func interleavedThinking() option.RequestOption {
	return option.WithHeaderAdd("anthropic-beta", interleavedThinkingBeta)
}

// requestOpts is what every request this service makes is sent with: the
// interleaved-thinking beta, unless the service was built for a backend that
// does not take it.
func (s *Service) requestOpts() []option.RequestOption {
	if s.omitInterleavedThinking {
		return nil
	}
	return []option.RequestOption{interleavedThinking()}
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

// timeEvent ends the TTFB at the first event carrying model output, and the
// TTFAT at a tool call's start: a turn that only calls tools produces no answer
// text, so the call itself is what the caller gets and TTFAT ends there rather
// than going unmeasured.
func (s *Service) timeEvent(ctx context.Context, event sdk.MessageStreamEventUnion) {
	if carriesModelOutput(event) {
		s.StopTTFBMetrics(ctx)
	}
	if start, ok := event.AsAny().(sdk.ContentBlockStartEvent); ok && start.ContentBlock.Type == "tool_use" {
		s.StopTTFATMetrics(ctx)
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
	stream := s.client.Messages.NewStreaming(ctx, params, s.requestOpts()...)
	for stream.Next() {
		event := stream.Current()
		s.timeEvent(ctx, event)
		if err := acc.Accumulate(event); err != nil {
			return err
		}
		if err := s.pushThought(ctx, event); err != nil {
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
		return s.reportUsage(ctx, acc.Usage)
	}
	return nil
}

// RunInference answers the conversation once, off to the side of the pipeline:
// no streaming, no frames, just the text. It implements llm.Inferencer.
func (s *Service) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	// The call's own instruction, or else the service's: an inference is still
	// this service answering.
	instruction := opts.SystemInstruction
	if instruction == "" {
		instruction = s.SystemInstruction()
	}
	// Built the way a streamed turn is, tools included: the API refuses a
	// conversation holding tool_use or tool_result blocks without the tools.
	params, err := s.newParams(convo, adapter.Options{SystemInstruction: instruction})
	if err != nil {
		return "", err
	}
	if opts.MaxTokens > 0 {
		params.MaxTokens = int64(opts.MaxTokens)
	}
	if schema := s.CheckResponseSchema(ctx, opts.ResponseSchema); schema != nil {
		var m map[string]any
		if err = json.Unmarshal(schema, &m); err != nil {
			return "", fmt.Errorf("anthropic: response schema: %w", err)
		}
		params.OutputConfig.Format = sdk.JSONOutputFormatParam{Schema: m}
	}
	msg, err := s.client.Messages.New(ctx, params, s.requestOpts()...)
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

// SupportsResponseSchema implements llm.ResponseSchemaSupporter.
func (s *Service) SupportsResponseSchema() bool { return true }

// modelVersion reads the generation out of a Claude model id: "claude-3-5-sonnet"
// is 3.5 and "claude-sonnet-4-5" 4.5. A minor version is at most two digits, so
// a date suffix is not taken for one.
//
//nolint:gochecknoglobals // a compiled pattern
var modelVersion = regexp.MustCompile(`claude(?:-[a-z]+)?-(\d+)(?:-(\d{1,2})(?:[^\d]|$))?`)

// ModelSupportsResponseSchema implements llm.ResponseSchemaSupporter.
// Structured outputs arrived with the 4.5 models. A model id without a
// version, such as a preview, is assumed to support them.
func (s *Service) ModelSupportsResponseSchema(model string) bool {
	m := modelVersion.FindStringSubmatch(model)
	if m == nil {
		return true
	}
	major, _ := strconv.Atoi(m[1])
	minor, _ := strconv.Atoi(m[2])
	return major > 4 || (major == 4 && minor >= 5)
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
