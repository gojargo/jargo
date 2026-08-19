package chat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/llm"
	errs "github.com/gojargo/jargo/utils/errors"
)

const (
	defaultLLMModel = "gpt-4o-mini"
	// defaultRetryTimeout bounds the first attempt when a service was built to
	// retry a request that times out.
	defaultRetryTimeout = 5 * time.Second
)

// The message roles the chat-completions API defines. They are aliases of the
// adapter's own, which is where the wire format lives.
const (
	RoleSystem    = openai.RoleSystem
	RoleUser      = openai.RoleUser
	RoleAssistant = openai.RoleAssistant
	RoleTool      = openai.RoleTool
	RoleDeveloper = openai.RoleDeveloper
)

// LLMConfig configures an OpenAI (or OpenAI-compatible) LLM service. The
// sampling controls are pointers so a deliberate zero is distinguishable from
// "unset"; a nil value is omitted from the request, leaving the API default.
type LLMConfig struct {
	// APIKey is the API key. It is not marked required because OpenAI-compatible
	// endpoints vary — some (e.g. a local Ollama) need none — so the caller
	// supplies it when the endpoint requires one.
	APIKey string
	// BaseURL overrides the API base (e.g. an OpenAI-compatible endpoint).
	BaseURL string
	// Model is the model id; empty uses the provider default.
	Model string
	// MaxTokens caps the response length; 0 omits it, leaving the API default.
	MaxTokens int
	// MaxCompletionTokens caps the completion length on models that require it in
	// place of MaxTokens; nil omits it.
	MaxCompletionTokens *int
	// Temperature is the sampling temperature (0.0 to 2.0); nil omits it.
	Temperature *float64
	// TopP is the nucleus-sampling parameter (0.0 to 1.0); nil omits it.
	TopP *float64
	// FrequencyPenalty penalizes frequent tokens (-2.0 to 2.0); nil omits it.
	FrequencyPenalty *float64
	// PresencePenalty penalizes already-present tokens (-2.0 to 2.0); nil omits it.
	PresencePenalty *float64
	// Seed requests deterministic sampling for a fixed seed; nil omits it.
	Seed *int
	// ServiceTier selects the tier the request is served under (e.g. "auto",
	// "flex", "priority") on an endpoint that offers them; empty omits it.
	ServiceTier string
	// RetryOnTimeout runs a request that takes longer than RetryTimeout a second
	// time, that one unbounded. It is for an endpoint that occasionally stalls on
	// the first token, where waiting out the stall costs more than asking again.
	RetryOnTimeout bool
	// RetryTimeout is how long the first attempt may take before RetryOnTimeout
	// gives up on it. Zero uses five seconds. It bounds nothing unless
	// RetryOnTimeout is set.
	RetryTimeout time.Duration
	// Extra sets arbitrary additional request-body fields not modeled above
	// (e.g. provider-specific parameters), applied to every request.
	Extra map[string]any
}

// RequestShaper customizes how a chat-completions request is addressed and
// authorized, so an OpenAI-compatible deployment with a different URL layout or
// auth scheme (e.g. Azure OpenAI) can reuse this implementation. The default
// shaper targets <baseURL>/chat/completions with a Bearer token.
type RequestShaper interface {
	// Endpoint returns the full chat-completions URL for baseURL, including any
	// query string.
	Endpoint(baseURL string) string
	// Authorize sets the authorization headers for apiKey on req.
	Authorize(req *http.Request, apiKey string)
}

// defaultShaper is the standard OpenAI addressing and Bearer authorization.
type defaultShaper struct{}

func (defaultShaper) Endpoint(baseURL string) string { return baseURL + "/chat/completions" }

func (defaultShaper) Authorize(req *http.Request, apiKey string) {
	req.Header.Set("Authorization", "Bearer "+apiKey)
}

// LLMService is a streaming OpenAI-compatible chat-completions LLM processor.
type LLMService struct {
	*llm.Base
	cfg    LLMConfig
	http   *http.Client
	shaper RequestShaper
	// adapter converts the conversation into the request this endpoint takes.
	adapter adapter.LLMAdapter[openai.Params, openai.Tool]
	// How this endpoint departs from OpenAI's own API, fixed at construction.
	noDeveloperRole bool
}

// Validate reports whether the configuration is usable.
func (c LLMConfig) Validate() error { return validate.Struct(c) }

// Compat describes an OpenAI-compatible endpoint: the label its service runs
// under, where it lives, and the ways it departs from OpenAI's own API. The
// zero value of every optional field is OpenAI's own behavior, so a provider
// only states what differs.
type Compat struct {
	// Name is the processor label the service is known by in logs, metrics and
	// traces.
	Name string
	// BaseURL is the API base used when the config leaves it empty.
	BaseURL string
	// DefaultModel is the model used when the config leaves it empty.
	DefaultModel string
	// Shaper addresses and authorizes the request. Nil means OpenAI's own
	// layout: <base>/chat/completions with a Bearer token.
	Shaper RequestShaper
	// NoDeveloperRole marks an endpoint with no developer role. Its messages are
	// sent as user messages instead, which is what carries an asynchronous
	// tool's late results to a model that would otherwise reject the role.
	NoDeveloperRole bool
	// Adapter converts the conversation into the request this endpoint takes. It
	// is for an endpoint that constrains the shape of a conversation beyond what
	// OpenAI's schema says: such an adapter embeds the OpenAI one and rewrites
	// what it produced. Nil converts the conversation as OpenAI itself takes it.
	Adapter adapter.LLMAdapter[openai.Params, openai.Tool]
	// Base configures the shared LLM base this service is built on.
	Base []llm.Option
}

// NewLLM builds an OpenAI LLM service.
func NewLLM(cfg LLMConfig) *LLMService {
	return NewCompatLLM(Compat{
		Name:         "OpenAILLM",
		BaseURL:      defaultLLMBaseURL,
		DefaultModel: defaultLLMModel,
	}, cfg)
}

// NewCompatLLM builds an LLM service for an OpenAI-compatible endpoint
// described by c.
func NewCompatLLM(c Compat, cfg LLMConfig) *LLMService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = c.BaseURL
	}
	if cfg.Model == "" {
		cfg.Model = c.DefaultModel
	}
	if cfg.RetryTimeout == 0 {
		cfg.RetryTimeout = defaultRetryTimeout
	}
	shaper := c.Shaper
	if shaper == nil {
		shaper = defaultShaper{}
	}
	a := c.Adapter
	if a == nil {
		a = &openai.Adapter{}
	}
	s := &LLMService{
		cfg:             cfg,
		http:            &http.Client{},
		shaper:          shaper,
		adapter:         a,
		noDeveloperRole: c.NoDeveloperRole,
	}
	s.Base = llm.New(c.Name, s, c.Base...)
	s.Base.SetModel(cfg.Model)
	return s
}

// The chat-completions wire types. They live in the adapter, which is what
// converts a conversation into them; these aliases keep them reachable under
// this package's name.
type (
	// Message is one message of the conversation as the chat-completions API
	// takes it.
	Message = openai.Message
	// ToolCall is an assistant tool-call entry on a message.
	ToolCall = openai.ToolCall
	// ToolCallFunction is the function a tool call invokes, with its arguments as
	// the raw JSON string the model produced.
	ToolCallFunction = openai.ToolCallFunction
)

type chatRequest struct {
	Model               string         `json:"model"`
	Messages            []Message      `json:"messages"`
	Stream              bool           `json:"stream"`
	StreamOptions       *streamOptions `json:"stream_options,omitempty"`
	MaxTokens           int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int           `json:"max_completion_tokens,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	TopP                *float64       `json:"top_p,omitempty"`
	FrequencyPenalty    *float64       `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64       `json:"presence_penalty,omitempty"`
	Seed                *int           `json:"seed,omitempty"`
	ServiceTier         string         `json:"service_tier,omitempty"`
	Tools               []openai.Tool  `json:"tools,omitempty"`
	ToolChoice          string         `json:"tool_choice,omitempty"`
}

// streamOptions asks for the token counts to be reported on the stream. Without
// it a streamed completion carries no usage at all.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// encodeBody marshals the request, merging any extra fields over the modeled
// ones. The merge cost is paid only when extra is non-empty.
func encodeBody(req chatRequest, extra map[string]any) ([]byte, error) {
	if len(extra) == 0 {
		return json.Marshal(req)
	}
	return openai.MergeExtra(req, extra)
}

type toolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatDelta struct {
	Content   string          `json:"content"`
	ToolCalls []toolCallDelta `json:"tool_calls"`
}

// chatUsage is the token accounting a completion reports. It arrives on a chunk
// of its own at the end of the stream, or repeated on every chunk as a running
// total, depending on the provider.
type chatUsage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

// tokenUsage converts the reported counts into the shape the pipeline carries.
func (u chatUsage) tokenUsage() frames.LLMTokenUsage {
	out := frames.LLMTokenUsage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.PromptTokensDetails != nil {
		out.CacheReadTokens = new(u.PromptTokensDetails.CachedTokens)
	}
	if u.CompletionTokensDetails != nil {
		out.ReasoningTokens = new(u.CompletionTokensDetails.ReasoningTokens)
	}
	return out
}

type chatChunk struct {
	// Model is the model that answered, which is more specific than the one
	// asked for: an alias resolves to a dated build here.
	Model   string     `json:"model"`
	Usage   *chatUsage `json:"usage"`
	Choices []struct {
		Delta chatDelta `json:"delta"`
	} `json:"choices"`
}

// textSink adapts a text-only generation to the streaming loop, which is shared
// with the tool-capable one. A request advertising no tools gives the model
// nothing to call, so a reported call would be the provider inventing one.
type textSink struct{ emit llm.Emit }

func (t textSink) Text(text string) error { return t.emit(text) }

func (textSink) Tool(frames.ToolCall) error { return nil }

// Generate streams a chat completion, emitting each content delta.
func (s *LLMService) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	reqBody, err := s.baseRequest(convo, adapter.Options{SystemInstruction: s.SystemInstruction()})
	if err != nil {
		return err
	}
	// A text-only generation advertises no tools, so the model has nothing to
	// call and a reported call would be the provider inventing one.
	reqBody.Tools = nil
	reqBody.ToolChoice = ""
	return s.generate(ctx, reqBody, textSink{emit})
}

// GenerateWithTools streams a tool-capable completion. It emits text deltas to
// the sink as they arrive and, once the stream completes, reports each tool call
// the model produced. The conversation's tools are sent on the request, and any
// tool turns already in the context are replayed as the matching messages.
func (s *LLMService) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	reqBody, err := s.baseRequest(convo, adapter.Options{SystemInstruction: s.SystemInstruction()})
	if err != nil {
		return err
	}
	return s.generate(ctx, reqBody, sink)
}

// baseRequest builds the streaming request shared by both generation paths.
func (s *LLMService) baseRequest(
	convo *frames.LLMContext, opts adapter.Options,
) (chatRequest, error) {
	p, err := s.params(convo, opts)
	if err != nil {
		return chatRequest{}, err
	}
	return chatRequest{
		Model:               s.cfg.Model,
		Messages:            p.Messages,
		Tools:               p.Tools,
		ToolChoice:          p.ToolChoice,
		Stream:              true,
		StreamOptions:       &streamOptions{IncludeUsage: true},
		MaxTokens:           s.cfg.MaxTokens,
		MaxCompletionTokens: s.cfg.MaxCompletionTokens,
		Temperature:         s.cfg.Temperature,
		TopP:                s.cfg.TopP,
		FrequencyPenalty:    s.cfg.FrequencyPenalty,
		PresencePenalty:     s.cfg.PresencePenalty,
		Seed:                s.cfg.Seed,
		ServiceTier:         s.cfg.ServiceTier,
	}, nil
}

// generate sends one streaming request and feeds what comes back to sink.
func (s *LLMService) generate(ctx context.Context, reqBody chatRequest, sink llm.Sink) error {
	body, err := encodeBody(reqBody, s.cfg.Extra)
	if err != nil {
		return err
	}
	s.StartTTFBMetrics()
	resp, err := s.send(ctx, body)
	s.StopTTFBMetrics()
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}
	return s.consume(ctx, resp.Body, sink)
}

// consume reads the stream: text and tool-call fragments reach the sink as they
// arrive, whole calls once the stream ends.
//
// Providers differ over how often they report the token counts, some once at
// the end and some as a running total on every chunk, so the latest is held and
// reported when the stream finishes: one report per completion either way. It is
// reported even when the stream fails or is cut off part way, because the
// tokens were spent regardless.
func (s *LLMService) consume(ctx context.Context, body io.Reader, sink llm.Sink) error {
	c := &toolCoalescer{calls: map[int]*toolAccumulator{}}
	var usage *chatUsage
	scanErr := llm.ScanSSE(body, func(data string) error {
		var chunk chatChunk
		if json.Unmarshal([]byte(data), &chunk) == nil {
			if chunk.Usage != nil {
				usage = chunk.Usage
			}
			if chunk.Model != "" {
				s.SetFullModelName(chunk.Model)
			}
			// The chunk carrying the counts has no choices of its own.
			if len(chunk.Choices) > 0 {
				return c.add(chunk.Choices[0].Delta, sink)
			}
		}
		return nil // Skip empty or malformed chunks.
	})
	if usage != nil && s.UsageMetricsEnabled() {
		if err := s.PushTokenUsage(ctx, usage.tokenUsage()); err != nil {
			return err
		}
	}
	if scanErr != nil {
		return scanErr
	}
	return c.emit(sink)
}

// send posts the encoded body to the chat-completions endpoint. A service built
// to retry a timed-out request bounds the wait for the first attempt and, if
// nothing has come back by then, asks again with no bound at all: the point is
// to give up on an attempt that has stalled, not on one that is answering
// slowly.
func (s *LLMService) send(ctx context.Context, body []byte) (*http.Response, error) {
	if !s.cfg.RetryOnTimeout {
		return s.sendOnce(ctx, body, 0)
	}
	resp, err := s.sendOnce(ctx, body, s.cfg.RetryTimeout)
	if err == nil || !errors.Is(err, llm.ErrCompletionTimeout) || ctx.Err() != nil {
		return resp, err
	}
	slog.DebugContext(ctx, "retrying the completion the endpoint did not start in time",
		"service", s.Name(), "waited", s.cfg.RetryTimeout)
	return s.sendOnce(ctx, body, 0)
}

// sendOnce makes one attempt. A non-zero bound applies to the wait for the
// response only: once the response has started the bound is lifted, so a long
// completion is never cut off part way. A timeout is reported as one, which is
// what a switcher fails over on.
func (s *LLMService) sendOnce(ctx context.Context, body []byte, bound time.Duration) (*http.Response, error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	var expired *time.Timer
	if bound > 0 {
		expired = time.AfterFunc(bound, cancel)
	}

	req, err := http.NewRequestWithContext(
		attemptCtx, http.MethodPost, s.shaper.Endpoint(s.cfg.BaseURL), bytes.NewReader(body),
	)
	if err != nil {
		cancel()
		return nil, err
	}
	s.shaper.Authorize(req, s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.http.Do(req)
	if err != nil {
		cancel()
		// The parent still being live is what says the bound expired rather than
		// the turn being interrupted.
		if ctx.Err() == nil && (attemptCtx.Err() != nil || os.IsTimeout(err)) {
			return nil, fmt.Errorf("%w: %w", llm.ErrCompletionTimeout, err)
		}
		return nil, err
	}
	if expired != nil {
		expired.Stop()
	}
	// The body is read after this returns, so the attempt is released when the
	// caller closes it rather than here.
	resp.Body = cancelOnClose{ReadCloser: resp.Body, cancel: cancel}
	return resp, nil
}

// cancelOnClose releases an attempt's context once its body is closed.
type cancelOnClose struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (c cancelOnClose) Close() error {
	err := c.ReadCloser.Close()
	c.cancel()
	return err
}

// toolAccumulator coalesces the streamed fragments of one tool call.
type toolAccumulator struct {
	id, name string
	args     strings.Builder
}

// toolCoalescer assembles tool calls from streamed deltas, keyed by index and
// remembering arrival order.
type toolCoalescer struct {
	calls map[int]*toolAccumulator
	order []int
}

// add folds one delta in: text deltas go straight to the sink, tool-call
// fragments accumulate by index.
func (c *toolCoalescer) add(delta chatDelta, sink llm.Sink) error {
	if len(delta.ToolCalls) == 0 {
		return sink.Text(delta.Content)
	}
	for _, d := range delta.ToolCalls {
		a := c.calls[d.Index]
		if a == nil {
			a = &toolAccumulator{}
			c.calls[d.Index] = a
			c.order = append(c.order, d.Index)
		}
		if d.ID != "" {
			a.id = d.ID
		}
		a.name += d.Function.Name
		a.args.WriteString(d.Function.Arguments)
	}
	return nil
}

// emit reports the assembled calls to the sink in arrival order, defaulting
// empty arguments to an empty JSON object.
//
// A call whose arguments did not arrive as valid JSON is dropped rather than
// passed on: nothing downstream can act on arguments that cannot be read, and a
// handler given them would fail on its own account, which the model would read
// as the tool having failed.
func (c *toolCoalescer) emit(sink llm.Sink) error {
	for _, idx := range c.order {
		a := c.calls[idx]
		if a.name == "" {
			continue
		}
		args := a.args.String()
		if args == "" {
			args = "{}"
		}
		if !json.Valid([]byte(args)) {
			slog.Warn("dropping a function call whose arguments are not valid JSON",
				"function", a.name, "tool_call_id", a.id, "arguments", args)
			continue
		}
		if err := sink.Tool(frames.ToolCall{ID: a.id, Name: a.name, Args: json.RawMessage(args)}); err != nil {
			return err
		}
	}
	return nil
}

// RunInference answers the conversation once, off to the side of the pipeline:
// no streaming, no frames, just the text. It implements llm.Inferencer.
func (s *LLMService) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	reqBody, err := s.baseRequest(convo, adapter.Options{SystemInstruction: opts.SystemInstruction})
	if err != nil {
		return "", err
	}
	reqBody.Stream = false
	reqBody.StreamOptions = nil
	// An inference wants an answer, not a tool call, so the toolset is left off.
	reqBody.Tools = nil
	reqBody.ToolChoice = ""
	if opts.MaxTokens > 0 {
		// Whichever bound this service states is the one to override; a model
		// that reads only one of the two is told through the field it reads.
		if reqBody.MaxCompletionTokens != nil {
			reqBody.MaxCompletionTokens = &opts.MaxTokens
		} else {
			reqBody.MaxTokens = opts.MaxTokens
		}
	}
	body, err := encodeBody(reqBody, s.cfg.Extra)
	if err != nil {
		return "", err
	}
	resp, err := s.send(ctx, body)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
		return "", err
	}
	if len(completion.Choices) == 0 {
		return "", nil
	}
	return completion.Choices[0].Message.Content, nil
}

// params converts the conversation into what this endpoint takes.
func (s *LLMService) params(
	convo *frames.LLMContext, opts adapter.Options,
) (openai.Params, error) {
	opts.ConvertDeveloperToUser = s.noDeveloperRole
	return s.adapter.LLMInvocationParams(convo, opts)
}

// LLMAdapter returns the adapter this service converts through, so the base can
// add the tools it implements itself to what every request advertises. It
// implements llm.AdapterHolder.
func (s *LLMService) LLMAdapter() llm.BuiltinToolHolder {
	holder, ok := s.adapter.(llm.BuiltinToolHolder)
	if !ok {
		return nil
	}
	return holder
}

// MessagesForLogging renders the conversation as this endpoint will see it, for
// the generation span. It implements llm.TraceRenderer.
func (s *LLMService) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	return s.adapter.MessagesForLogging(convo)
}

// ToolsForLogging renders the toolset as this endpoint will see it, for the
// generation span. It implements llm.TraceRenderer.
func (s *LLMService) ToolsForLogging(schema frames.ToolsSchema) []any {
	return adapter.ToolsForLogging(s.adapter, schema)
}
