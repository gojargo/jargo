// Package llm is the shared base for streaming LLM services. A concrete service
// embeds *Base and implements Generator; the base handles the LLMContextFrame
// lifecycle and brackets the streamed response as an LLMFullResponseStartFrame,
// a stream of LLMTextFrames, and an LLMFullResponseEndFrame.
//
// This keeps every provider down to the part that differs — turning a
// conversation into a stream of text deltas — while the frame contract lives in
// one place.
//
// A service that also supports tool calling implements ToolGenerator. When the
// context carries tools, the base streams text, dispatches each requested call to
// a registered handler, and emits the function-call frames; the assistant
// aggregator commits the results and re-triggers generation for the model's reply.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/telemetry/metrics"
	"github.com/gojargo/jargo/telemetry/tracing"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// missingFunctionMessage is the tool result returned when the model calls a
// function no handler is registered for. It is a terminal result rather than an
// error, so the call completes normally and the model is told the function is
// unavailable instead of being left waiting.
const missingFunctionMessage = "The function `%s` is not currently available."

// Emit pushes a chunk of generated text downstream as an LLMTextFrame. A
// Generator calls it once per delta received from the provider; empty deltas are
// ignored.
type Emit func(text string) error

// Generator produces a streaming completion for a conversation. The
// implementation streams text deltas to emit until the response completes or ctx
// is canceled (an interruption). A returned error is reported upstream.
type Generator interface {
	Generate(ctx context.Context, convo *frames.LLMContext, emit Emit) error
}

// Sink receives the streamed output of a tool-capable generation: text deltas
// via Text and each requested tool call via Tool.
type Sink interface {
	Text(text string) error
	Tool(call frames.ToolCall) error
}

// ToolGenerator is implemented by services that support tool calling. It streams
// text to sink.Text and reports each tool call the model requests to sink.Tool.
// It returns when the model's turn completes — either a final text answer or a
// request to call tools — or when ctx is canceled.
type ToolGenerator interface {
	GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink Sink) error
}

// FunctionCallResultCallback delivers a tool call's result to the conversation.
// props tunes how the result is applied and may be nil for the defaults: a final
// result, on which the aggregator decides whether to re-generate.
//
// It is how a handler reports, rather than returning a value, because a call can
// have more than one thing to say. A handler registered to survive interruptions
// calls it repeatedly with props.IsFinal false to stream progress, and once more
// without to finish.
type FunctionCallResultCallback func(
	ctx context.Context, result string, props *frames.FunctionCallResultProperties,
) error

// FunctionCallParams is everything a tool handler is given for one call.
type FunctionCallParams struct {
	// FunctionName is the name of the tool being called.
	FunctionName string
	// ToolCallID identifies this invocation.
	ToolCallID string
	// Arguments is the raw JSON the model produced for the call.
	Arguments json.RawMessage
	// LLM is the service running the call.
	LLM *Base
	// Context is the conversation the call was made in.
	Context *frames.LLMContext
	// Result delivers what the call produced. A handler that returns without
	// calling it reports nothing, and the call is left showing as in progress
	// until it is canceled.
	Result FunctionCallResultCallback
}

// FunctionCallHandler runs one tool call, reporting what it produced through
// params.Result. A returned error is reported as a non-fatal pipeline error and
// no result reaches the conversation, so a handler that wants the model to see a
// failure should report the failure as its result instead of returning it.
//
// A handler that does blocking work must honor ctx: it is canceled when the call
// is, which for an ordinary tool means the moment the user interrupts.
type FunctionCallHandler func(ctx context.Context, params FunctionCallParams) error

// catchAllFunction is the registry key of the handler that takes every call no
// named handler claims. Registering under it is how one handler serves a whole
// toolset, a proxy to another system for instance.
const catchAllFunction = ""

// registryItem is a registered handler and the call options it was registered
// with.
type registryItem struct {
	name                 string
	handler              FunctionCallHandler
	cancelOnInterruption bool
	// timeout bounds one call to this function, overriding the service's own.
	// Zero means the service's.
	timeout time.Duration
}

// RegisterOption configures how a registered function's calls are run.
type RegisterOption func(*registryItem)

// WithCancelOnInterruption sets whether a call to this function is canceled when
// the user interrupts. It is true by default.
//
// Registering with false makes the tool asynchronous: the model carries on with
// the conversation rather than waiting, the call outlives the turn that made it,
// and everything the handler reports arrives as async-tool messages the model
// reads on a later turn.
func WithCancelOnInterruption(cancel bool) RegisterOption {
	return func(i *registryItem) { i.cancelOnInterruption = cancel }
}

// WithTimeout bounds how long one call to this function may take before it is
// given up on, overriding whatever the service was built with. Zero restores the
// service's own bound.
func WithTimeout(d time.Duration) RegisterOption {
	return func(i *registryItem) { i.timeout = d }
}

// Option configures an LLM Base at construction.
type Option func(*Base)

// WithFunctionCallTimeout bounds how long any tool call may take before it is
// given up on. A call that overruns reports no result of its own, which records
// it as completed and leaves generation to whatever comes next rather than
// answering on the tool's behalf.
//
// Zero, the default, means no bound: a call runs until it reports, is canceled,
// or the session ends. WithTimeout overrides it for one function.
func WithFunctionCallTimeout(d time.Duration) Option {
	return func(b *Base) { b.callTimeout = d }
}

// WithSequentialFunctionCalls runs the calls of one response one after another
// rather than at the same time. Use it when the tools share something that does
// not take concurrent use, or when a later call is only meaningful once an
// earlier one has taken effect.
func WithSequentialFunctionCalls() Option {
	return func(b *Base) { b.runInParallel = false }
}

// WithUngroupedFunctionCalls makes each tool result re-run generation on its own
// instead of the batch answering once its last call reports. The model then
// answers as many times as it made calls, which is rarely what a spoken
// conversation wants, so grouping is the default.
func WithUngroupedFunctionCalls() Option {
	return func(b *Base) { b.groupCalls = false }
}

// FunctionCallsHandler is notified about the tool calls of one response.
type FunctionCallsHandler func(ctx context.Context, calls []frames.ToolCall)

// functionCall is one call in flight, tracked so it can be canceled.
type functionCall struct {
	item       registryItem
	name       string
	toolCallID string
	args       json.RawMessage
	convo      *frames.LLMContext
	groupID    string

	cancel context.CancelFunc
	// canceled stops the result callback from reporting after the call was
	// canceled. A goroutine cannot be killed the way a task can, so a handler
	// that ignores cancellation and reports anyway is silenced here: the
	// conversation already records the call as canceled.
	canceled bool
	// stopTimeout disarms the bound on how long the call may take. Reporting
	// anything at all disarms it: a tool that has spoken is not stuck.
	stopTimeout func()
}

// SettingsHolder is an optional interface a Generator implements when part of
// what it was built with can change while the pipeline runs: the model, the
// system prompt, the sampling knobs. The value returned is the provider's own
// store, a pointer to a settings value, which an update is merged into.
type SettingsHolder interface {
	Settings() any
}

// SettingsUpdater is an optional interface a Generator implements to act on a
// settings change, with what changed and what each field held before. A
// Generator that holds settings without implementing this still has them
// updated; it picks them up on the next generation, which is enough for a
// service that reads its settings each time it generates.
type SettingsUpdater interface {
	SettingsHolder
	UpdateSettings(ctx context.Context, changed settings.Changed) error
}

// Base is the shared LLM processor. It runs the embedded Generator on each
// LLMContextFrame and surrounds the streamed text with response start/end
// frames. When the context carries tools and the generator supports them, it
// runs the tool loop instead.
type Base struct {
	*processor.Base
	gen Generator

	// modelMu guards model, which labels the metrics and can change mid-call.
	modelMu sync.Mutex
	model   string // reported as a span attribute; set by the provider via SetModel

	handlersMu sync.RWMutex
	handlers   map[string]registryItem

	// How the tool calls of one response are run, fixed at construction.
	callTimeout   time.Duration
	runInParallel bool
	groupCalls    bool
	// asyncToolCancellation offers the model a built-in tool for abandoning an
	// asynchronous call, while any asynchronous tool is registered.
	asyncToolCancellation bool
	// cancelToolActive tracks whether that tool is currently offered, so it is
	// added and withdrawn exactly once. Guarded by handlersMu.
	cancelToolActive bool

	// Tool calls in flight, keyed by tool call id. They run off the frame path,
	// on a lifetime of their own, so that a handler taking its time does not hold
	// up the frames queued behind it, including the speech that covers the wait.
	callsMu     sync.Mutex
	calls       map[string]*functionCall
	callsCtx    context.Context
	callsCancel context.CancelFunc
	callsWG     sync.WaitGroup
	// sequential carries the calls of a response to the runner that takes them
	// one at a time. Nil unless the service was built to run them that way.
	sequential chan *functionCall

	// Notifications about the calls of a response, set by the application.
	eventsMu   sync.RWMutex
	onStarted  FunctionCallsHandler
	onCanceled FunctionCallsHandler

	ttfbMu    sync.Mutex
	ttfbStart time.Time
	ttfb      time.Duration
	hasTTFB   bool
}

// New builds an LLM Base named name driven by gen. The concrete service passes
// itself as gen and embeds the returned Base.
func New(name string, gen Generator, opts ...Option) *Base {
	b := &Base{
		gen:           gen,
		calls:         make(map[string]*functionCall),
		runInParallel: true,
		groupCalls:    true,
	}
	for _, opt := range opts {
		opt(b)
	}
	b.Base = processor.New(name, b)
	return b
}

// OnFunctionCallsStarted registers a callback run when the model requests tool
// calls, with the calls of that response. It runs off the frame path, so what it
// does cannot hold up the pipeline.
func (b *Base) OnFunctionCallsStarted(h FunctionCallsHandler) {
	b.eventsMu.Lock()
	b.onStarted = h
	b.eventsMu.Unlock()
}

// OnFunctionCallsCanceled registers a callback run when tool calls are
// canceled, with the calls that were. It runs off the frame path.
func (b *Base) OnFunctionCallsCanceled(h FunctionCallsHandler) {
	b.eventsMu.Lock()
	b.onCanceled = h
	b.eventsMu.Unlock()
}

// notify runs one of the two callbacks on a goroutine of its own, tracked so
// Cleanup waits for it.
func (b *Base) notify(ctx context.Context, h FunctionCallsHandler, calls []frames.ToolCall) {
	if h == nil || len(calls) == 0 {
		return
	}
	b.callsWG.Go(func() { h(ctx, calls) })
}

// Setup opens the lifetime the tool calls run under. It is the session's, not
// the turn's: a call must be able to outlive the frame that started it, so that
// canceling it is a decision this service makes rather than a side effect of the
// turn ending.
func (b *Base) Setup(ctx context.Context, s processor.Setup) error {
	if err := b.Base.Setup(ctx, s); err != nil {
		return err
	}
	b.callsCtx, b.callsCancel = context.WithCancel(ctx)
	if !b.runInParallel {
		b.sequential = make(chan *functionCall, sequentialQueueDepth)
		b.callsWG.Go(b.runSequentially)
	}
	return nil
}

// sequentialQueueDepth bounds the calls waiting for the sequential runner. It is
// generous: a response requesting more calls than this at once would be
// remarkable, and a full queue only makes the service wait its turn to enqueue.
const sequentialQueueDepth = 64

// runSequentially takes queued calls one at a time, so a service built with
// WithSequentialFunctionCalls never has two handlers running at once.
func (b *Base) runSequentially() {
	for {
		select {
		case <-b.callsCtx.Done():
			return
		case call := <-b.sequential:
			b.runFunctionCall(call)
		}
	}
}

// Cleanup cancels every tool call still running and waits for the handlers to
// return.
func (b *Base) Cleanup(ctx context.Context) error {
	if b.callsCancel != nil {
		b.callsCancel()
	}
	b.callsWG.Wait()
	return b.Base.Cleanup(ctx)
}

// SetModel records the model id the service generates with, reported as the
// llm.model span attribute. A provider calls it during construction.
func (b *Base) SetModel(model string) {
	b.modelMu.Lock()
	defer b.modelMu.Unlock()
	b.model = model
}

// modelName is the identifier the generation is measured and priced against.
func (b *Base) modelName() string {
	b.modelMu.Lock()
	defer b.modelMu.Unlock()
	return b.model
}

// PushTokenUsage emits a MetricsFrame carrying token usage downstream. A service
// calls it after a generation, gated on UsageMetricsEnabled, so the conversion
// from the provider's usage shape happens only when metrics are collected.
func (b *Base) PushTokenUsage(ctx context.Context, u frames.LLMTokenUsage) error {
	tracing.SetTokenUsage(ctx, u)
	metrics.RecordTokens(ctx, b.Name(), b.modelName(), u.PromptTokens, u.CompletionTokens)
	f := frames.NewMetricsFrame(frames.LLMUsageMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: b.Name(), Model: b.modelName()},
		Value:           u,
	})
	return b.PushFrame(ctx, f, processor.Downstream)
}

// StartTTFBMetrics starts this generation's time-to-first-byte clock. A service
// calls it immediately before issuing the request. It measures nothing once the
// pipeline has had the only measurement it asked for, which leaves the clock
// unstarted and so reports no time to first byte for this generation.
func (b *Base) StartTTFBMetrics() {
	armed := b.BeginTTFB()
	b.ttfbMu.Lock()
	defer b.ttfbMu.Unlock()
	b.ttfbStart = time.Time{}
	b.ttfb = 0
	b.hasTTFB = false
	if armed {
		b.ttfbStart = time.Now()
	}
}

// StopTTFBMetrics records time to first byte. A service calls it as soon as the
// provider's response stream is open, which is the moment the model starts
// answering — a turn that only requests tools streams no text, and so would
// otherwise report no TTFB at all. Only the first call in a generation counts.
func (b *Base) StopTTFBMetrics() {
	b.ttfbMu.Lock()
	defer b.ttfbMu.Unlock()
	if b.hasTTFB || b.ttfbStart.IsZero() {
		return
	}
	b.ttfb = time.Since(b.ttfbStart)
	b.hasTTFB = true
}

// ttfbMetrics returns the recorded time to first byte, and whether the service
// recorded one at all.
func (b *Base) ttfbMetrics() (time.Duration, bool) {
	b.ttfbMu.Lock()
	defer b.ttfbMu.Unlock()
	return b.ttfb, b.hasTTFB
}

// emitTiming records the generation's time-to-first-byte and processing time to
// the span and to OpenTelemetry (always) and, when in-band metrics are enabled,
// downstream as a MetricsFrame for the RTVI client.
func (b *Base) emitTiming(ctx context.Context, span trace.Span, processing time.Duration) {
	ttfb, hadTTFB := b.ttfbMetrics()
	model := b.modelName()
	metrics.RecordProcessing(ctx, "llm", b.Name(), model, processing.Seconds())
	if hadTTFB {
		span.SetAttributes(attribute.Int64("llm.ttfb_ms", ttfb.Milliseconds()))
		metrics.RecordTTFB(ctx, "llm", b.Name(), model, ttfb.Seconds())
	}
	if !b.MetricsEnabled() {
		return
	}
	base := frames.BaseMetricsData{Processor: b.Name(), Model: model}
	data := []frames.MetricsData{frames.ProcessingMetricsData{BaseMetricsData: base, Value: processing}}
	if hadTTFB {
		data = append(data, frames.TTFBMetricsData{BaseMetricsData: base, Value: ttfb})
	}
	_ = b.PushFrame(ctx, frames.NewMetricsFrame(data...), processor.Downstream)
}

// startSpan opens the generation span, tagging it with the service name and
// model. The returned context carries the span so PushTokenUsage and any nested
// work attach to it.
func (b *Base) startSpan(ctx context.Context) (context.Context, trace.Span) {
	ctx, span := tracing.Tracer().Start(b.Tracing().Parent(ctx), "llm")
	span.SetAttributes(attribute.String("llm.service", b.Name()))
	if model := b.modelName(); model != "" {
		span.SetAttributes(attribute.String("llm.model", model))
	}
	return ctx, span
}

// traceRequest tags the generation span with the gen_ai.* request attributes and
// the serialized input messages, so a trace backend (e.g. Langfuse) renders the
// prompt. Mirrors Pipecat's add_llm_span_attributes (input + gen_ai.request.*).
func (b *Base) traceRequest(span trace.Span, convo *frames.LLMContext) {
	span.SetAttributes(
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.output.type", "text"),
		attribute.Bool("stream", true),
	)
	if model := b.modelName(); model != "" {
		span.SetAttributes(attribute.String("gen_ai.request.model", model))
	}
	if in := traceMessages(convo); in != "" {
		span.SetAttributes(attribute.String("input", in))
	}
}

// traceOutput records the model's reply text as the span's output attribute.
func traceOutput(span trace.Span, text string) {
	if text != "" {
		span.SetAttributes(attribute.String("output", text))
	}
}

// traceMessages renders the context (system prompt plus conversation) as a JSON
// array of role/content messages for the span's input attribute — the shape
// Langfuse and the gen_ai convention read.
func traceMessages(convo *frames.LLMContext) string {
	type msg struct {
		Role        string              `json:"role"`
		Content     string              `json:"content,omitempty"`
		ToolCalls   []frames.ToolCall   `json:"tool_calls,omitempty"`
		ToolResults []frames.ToolResult `json:"tool_results,omitempty"`
	}
	var out []msg
	if sys := convo.System(); sys != "" {
		out = append(out, msg{Role: "system", Content: sys})
	}
	for _, m := range convo.Messages() {
		out = append(out, msg{
			Role:        string(m.Role),
			Content:     m.Text,
			ToolCalls:   m.ToolCalls,
			ToolResults: m.ToolResults,
		})
	}
	data, err := json.Marshal(out)
	if err != nil {
		return ""
	}
	return string(data)
}

// RegisterFunction registers a handler for the named tool. During a tool-capable
// generation, a call to that tool is dispatched to the handler. By default the
// call is canceled when the user interrupts; see WithCancelOnInterruption.
//
// Registering under the empty name registers a catch-all: it takes every call no
// named handler claims, which is how one handler serves a whole toolset.
func (b *Base) RegisterFunction(name string, h FunctionCallHandler, opts ...RegisterOption) {
	if name == CancelAsyncToolName {
		slog.Error("refusing to register a reserved built-in tool name",
			"service", b.Name(), "function", name)
		return
	}
	item := registryItem{name: name, handler: h, cancelOnInterruption: true}
	for _, opt := range opts {
		opt(&item)
	}
	b.handlersMu.Lock()
	if b.handlers == nil {
		b.handlers = make(map[string]registryItem)
	}
	b.handlers[name] = item
	b.handlersMu.Unlock()

	// Registering the first asynchronous tool is what brings the built-in
	// cancellation tool in.
	b.syncAsyncToolCancellation()
}

// UnregisterFunction removes the handler registered for name, reporting whether
// there was one.
//
// It stops the tool being handled, not being offered. To withdraw a tool
// properly, push an [frames.LLMSetToolsFrame] with the tool left out: that stops
// advertising it as well, rather than leaving the model free to call something
// that answers only that it is unavailable.
func (b *Base) UnregisterFunction(name string) bool {
	b.handlersMu.Lock()
	_, ok := b.handlers[name]
	delete(b.handlers, name)
	b.handlersMu.Unlock()

	// Withdrawing the last asynchronous tool takes the built-in cancellation
	// tool away with it.
	b.syncAsyncToolCancellation()
	return ok
}

// HasFunction reports whether a handler is registered for name. A catch-all
// counts: it would take the call.
func (b *Base) HasFunction(name string) bool {
	b.handlersMu.RLock()
	defer b.handlersMu.RUnlock()
	if _, ok := b.handlers[name]; ok {
		return true
	}
	_, ok := b.handlers[catchAllFunction]
	return ok
}

// lookupFunction returns the entry that runs a call to name: its own handler, or
// else the catch-all. It reports false when nothing claims the call, which the
// caller answers with the missing-function handler.
func (b *Base) lookupFunction(name string) (registryItem, bool) {
	b.handlersMu.RLock()
	item, ok := b.handlers[name]
	if !ok {
		item, ok = b.handlers[catchAllFunction]
	}
	b.handlersMu.RUnlock()
	if !ok {
		return registryItem{}, false
	}
	// The catch-all is registered under the empty name; a call runs under the
	// name the model actually used.
	item.name = name
	return item, true
}

// logMissingFunction says why a call had no handler, telling apart the two cases
// that lead here. A tool the conversation advertises with nothing behind it is
// almost always a wiring mistake, and reads as an error. A tool nothing
// advertises is the model inventing one, which an application can do little
// about beyond answering that it is unavailable.
func (b *Base) logMissingFunction(ctx context.Context, name string, convo *frames.LLMContext) {
	for _, t := range convo.Tools() {
		if t.Name == name {
			slog.ErrorContext(ctx, "tool is advertised to the model but has no handler",
				"service", b.Name(), "function", name)
			return
		}
	}
	slog.WarnContext(ctx, "model called a tool that is not in the advertised toolset",
		"service", b.Name(), "function", name)
}

// missingFunctionHandler answers a call to an unregistered tool with a terminal
// result naming it.
func missingFunctionHandler(ctx context.Context, params FunctionCallParams) error {
	return params.Result(ctx, fmt.Sprintf(missingFunctionMessage, params.FunctionName), nil)
}

// updateSettings merges an update into the provider's own settings and lets it
// act on what changed. There is no reconnection here: a generation is a request
// of its own, so a provider reads its settings the next time it generates.
func (b *Base) updateSettings(ctx context.Context, f *frames.LLMUpdateSettingsFrame) {
	holder, ok := b.gen.(SettingsHolder)
	if !ok {
		slog.Warn("settings update for a service whose provider has none", "service", b.Name())
		return
	}
	store := holder.Settings()

	delta, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, store)
	if err != nil {
		b.PushError(ctx, "llm: settings update", err, false)
		return
	}
	if !ok {
		return
	}

	changed, err := settings.Apply(store, delta)
	if err != nil {
		b.PushError(ctx, "llm: settings update", err, false)
		return
	}
	if len(changed) == 0 {
		return
	}
	slog.Info("updated settings", "service", b.Name(), "fields", changed.String())

	if changed.Has("model") {
		// The model labels the tokens this service reports and is what they are
		// priced against, so a model that changed mid-call has to relabel what
		// follows.
		name, _ := settings.Get(store, "model")
		model, _ := name.(string)
		b.SetModel(model)
	}

	if updater, ok := b.gen.(SettingsUpdater); ok {
		if err := updater.UpdateSettings(ctx, changed); err != nil {
			b.PushError(ctx, "llm: settings update", err, false)
		}
	}
}

// ProcessFrame runs the generator on each LLMContextFrame and forwards other
// frames untouched.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMContextFrame:
		return b.run(ctx, fr.Context)
	case *frames.InterruptionFrame:
		b.cancelFunctionCalls(ctx)
		return b.PushFrame(ctx, f, dir)
	case *frames.StartFrame:
		if err := b.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		b.broadcastMetadata(ctx)
		return nil
	case *frames.LLMUpdateSettingsFrame:
		if !fr.TargetsService(b) {
			// Meant for another service; leave it untouched for that one.
			return b.PushFrame(ctx, f, dir)
		}
		b.updateSettings(ctx, fr)
		return nil
	default:
		return b.PushFrame(ctx, f, dir)
	}
}

// broadcastMetadata pushes the LLM service's metadata frame downstream at
// pipeline start so downstream processors can discover the service.
func (b *Base) broadcastMetadata(ctx context.Context) {
	_ = b.PushFrame(ctx, frames.NewLLMServiceMetadataFrame(b.Name()), processor.Downstream)
}

// run streams a response for the conversation, choosing the tool loop when the
// context carries tools and the generator supports them. It runs under the
// process goroutine's context, so an interruption cancels the in-flight work.
func (b *Base) run(ctx context.Context, convo *frames.LLMContext) error {
	// Bring what this service adds on its own account into line with what is
	// registered, before anything reads the conversation to build the request.
	b.applyAsyncToolCancellation(convo)
	if len(convo.Tools()) > 0 {
		if tg, ok := b.gen.(ToolGenerator); ok {
			return b.runWithTools(ctx, convo, tg)
		}
		slog.Warn("LLM service does not support tools; tools ignored", "processor", b.Name())
	}
	return b.runText(ctx, convo)
}

// runText is the text-only path: brackets the streamed deltas with response
// start/end frames.
func (b *Base) runText(ctx context.Context, convo *frames.LLMContext) error {
	ctx, span := b.startSpan(ctx)
	defer span.End()
	b.traceRequest(span, convo)
	if err := b.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	start := time.Now()
	var out strings.Builder
	emit := func(text string) error {
		if text == "" {
			return nil
		}
		out.WriteString(text)
		return b.PushFrame(ctx, frames.NewLLMTextFrame(text), processor.Downstream)
	}
	if err := b.gen.Generate(ctx, convo, emit); err != nil && ctx.Err() == nil {
		span.RecordError(err)
		b.PushError(ctx, "llm generation failed", err, false)
	}
	traceOutput(span, out.String())
	b.emitTiming(ctx, span, time.Since(start))
	return b.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// sink adapts per-pass closures to the Sink interface.
type sink struct {
	text func(string) error
	tool func(frames.ToolCall) error
}

func (s sink) Text(t string) error          { return s.text(t) }
func (s sink) Tool(c frames.ToolCall) error { return s.tool(c) }

// runWithTools runs one tool-capable inference: it streams text and, when the
// model requests tools, dispatches each call to its handler and emits the
// function-call frames. It does not loop — the assistant aggregator commits the
// results to the context and re-triggers generation, so the next inference reads a
// context whose tool calls are balanced. The base writes nothing to the context.
func (b *Base) runWithTools(ctx context.Context, convo *frames.LLMContext, tg ToolGenerator) error {
	ctx, span := b.startSpan(ctx)
	defer span.End()
	span.SetAttributes(attribute.Bool("llm.tools", true))
	b.traceRequest(span, convo)
	if err := b.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	start := time.Now()
	var preamble strings.Builder
	var calls []frames.ToolCall
	s := sink{
		text: func(t string) error {
			if t == "" {
				return nil
			}
			preamble.WriteString(t)
			return b.PushFrame(ctx, frames.NewLLMTextFrame(t), processor.Downstream)
		},
		tool: func(c frames.ToolCall) error {
			calls = append(calls, c)
			return nil
		},
	}
	if err := tg.GenerateWithTools(ctx, convo, s); err != nil && ctx.Err() == nil {
		b.PushError(ctx, "llm generation failed", err, false)
	} else if ctx.Err() == nil && len(calls) > 0 {
		b.eventsMu.RLock()
		started := b.onStarted
		b.eventsMu.RUnlock()
		b.notify(ctx, started, calls)
		if err := b.Broadcast(ctx, func() frames.Frame {
			return frames.NewFunctionCallsStartedFrame(calls)
		}); err != nil {
			return err
		}
		b.runFunctionCalls(ctx, convo, calls)
	}
	traceOutput(span, preamble.String())
	b.emitTiming(ctx, span, time.Since(start))
	return b.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// runFunctionCalls starts every call the model requested this turn and returns
// without waiting. A handler runs for as long as the work it does takes. Running
// it here, on the goroutine that processes this service's frames, would hold up
// every frame queued behind it, including the speech a bot plays to cover the
// wait.
//
// The calls share a group id, which is how the aggregator recognizes them as one
// batch and answers the batch with a single inference once the last of them has
// reported, rather than once per call.
func (b *Base) runFunctionCalls(ctx context.Context, convo *frames.LLMContext, calls []frames.ToolCall) {
	groupID := ""
	if b.groupCalls {
		groupID = uuid.NewString()
	}
	for _, c := range calls {
		item, claimed := b.lookupFunction(c.Name)
		if !claimed {
			// Nothing claims it, so say which of the two ways that happened before
			// the missing-function handler answers the model.
			b.logMissingFunction(ctx, c.Name, convo)
			item = registryItem{name: c.Name, handler: missingFunctionHandler, cancelOnInterruption: true}
		}
		call := &functionCall{
			item:       item,
			name:       c.Name,
			toolCallID: c.ID,
			args:       c.Args,
			convo:      convo,
			groupID:    groupID,
		}
		if b.runInParallel {
			b.callsWG.Go(func() { b.runFunctionCall(call) })
			continue
		}
		// Sequential: one runner takes them in turn, so no two handlers overlap.
		select {
		case b.sequential <- call:
		case <-b.callsCtx.Done():
			return
		}
	}
}

// runFunctionCall executes one tool call: it announces the call as in progress,
// which is what writes it and the message answering it to the conversation, then
// dispatches it to its handler.
//
// The call runs under a context of its own, derived from the session's rather
// than the turn's, so that whether it survives an interruption is decided by how
// it was registered and not by the turn ending.
func (b *Base) runFunctionCall(call *functionCall) {
	ctx, cancel := context.WithCancel(b.callsCtx)
	defer cancel()

	b.callsMu.Lock()
	call.cancel = cancel
	b.calls[call.toolCallID] = call
	b.callsMu.Unlock()
	defer func() {
		b.callsMu.Lock()
		// Only if it is still this call: a cancellation may already have dropped
		// it, and the model may have called the same tool again since.
		if b.calls[call.toolCallID] == call {
			delete(b.calls, call.toolCallID)
		}
		b.callsMu.Unlock()
	}()

	slog.DebugContext(ctx, "calling function", "service", b.Name(),
		"function", call.name, "tool_call_id", call.toolCallID)

	if err := b.Broadcast(ctx, func() frames.Frame {
		return frames.NewFunctionCallInProgressFrame(
			call.toolCallID, call.name, call.args, call.item.cancelOnInterruption, call.groupID,
		)
	}); err != nil {
		return
	}

	report := b.resultCallback(call)
	stopTimeout := b.startCallTimeout(ctx, call, report)
	defer stopTimeout()
	b.callsMu.Lock()
	call.stopTimeout = stopTimeout
	b.callsMu.Unlock()

	params := FunctionCallParams{
		FunctionName: call.name,
		ToolCallID:   call.toolCallID,
		Arguments:    call.args,
		LLM:          b,
		Context:      call.convo,
		Result:       report,
	}
	if err := call.item.handler(ctx, params); err != nil {
		// The conversation is told nothing. A handler that failed has no result to
		// report, and inventing one would put words in the tool's mouth; the call
		// stays in progress until it is canceled.
		b.PushError(ctx, fmt.Sprintf("error executing function call [%s]: %v", call.name, err), err, false)
	}
}

// startCallTimeout arms the bound on how long a call may take, returning the
// function that disarms it. A call that overruns reports no result of its own,
// which records it as completed and leaves generation to whatever comes next
// rather than answering on the tool's behalf. The handler is not canceled: it
// may still be doing something worth finishing, and the result callback ignores
// it from here.
//
// It returns a no-op when no bound applies, so a call with none costs nothing.
func (b *Base) startCallTimeout(
	ctx context.Context, call *functionCall, report FunctionCallResultCallback,
) func() {
	timeout := call.item.timeout
	if timeout == 0 {
		timeout = b.callTimeout
	}
	if timeout <= 0 {
		return func() {}
	}
	timer := time.AfterFunc(timeout, func() {
		slog.WarnContext(ctx, "function call timed out", "service", b.Name(),
			"function", call.name, "tool_call_id", call.toolCallID, "timeout", timeout)
		_ = report(ctx, "", nil)
	})
	return func() { timer.Stop() }
}

// resultCallback builds the callback the handler reports through. It is bound to
// the call rather than to a turn, so a result reports against the call that
// produced it however much has happened since.
func (b *Base) resultCallback(call *functionCall) FunctionCallResultCallback {
	return func(ctx context.Context, result string, props *frames.FunctionCallResultProperties) error {
		b.callsMu.Lock()
		canceled := call.canceled
		b.callsMu.Unlock()
		if canceled {
			return nil
		}
		if !props.Final() && call.item.cancelOnInterruption {
			slog.WarnContext(ctx, "intermediate result from a tool that is not asynchronous",
				"service", b.Name(), "function", call.name, "tool_call_id", call.toolCallID)
			return nil
		}
		// A tool that has spoken is not stuck, so reporting anything at all,
		// intermediate results included, disarms the bound on how long it may take.
		b.callsMu.Lock()
		stop := call.stopTimeout
		b.callsMu.Unlock()
		if stop != nil {
			stop()
		}
		return b.Broadcast(ctx, func() frames.Frame {
			f := frames.NewFunctionCallResultFrame(call.toolCallID, call.name, call.args, result)
			f.Properties = props
			return f
		})
	}
}

// cancelFunctionCalls cancels every call in flight that was registered to be
// canceled on interruption, and announces each one so the conversation records
// it as canceled rather than as still running. A call registered to survive an
// interruption is left to finish.
func (b *Base) cancelFunctionCalls(ctx context.Context) {
	b.callsMu.Lock()
	var canceled []*functionCall
	for id, call := range b.calls {
		if !call.item.cancelOnInterruption {
			continue
		}
		call.canceled = true
		delete(b.calls, id)
		canceled = append(canceled, call)
	}
	b.callsMu.Unlock()

	dropped := make([]frames.ToolCall, 0, len(canceled))
	for _, call := range canceled {
		slog.DebugContext(ctx, "canceling function call", "service", b.Name(),
			"function", call.name, "tool_call_id", call.toolCallID)
		call.cancel()
		_ = b.Broadcast(ctx, func() frames.Frame {
			return frames.NewFunctionCallCancelFrame(call.toolCallID, call.name)
		})
		dropped = append(dropped, frames.ToolCall{ID: call.toolCallID, Name: call.name, Args: call.args})
	}

	b.eventsMu.RLock()
	h := b.onCanceled
	b.eventsMu.RUnlock()
	b.notify(ctx, h, dropped)
}

// CanGenerateMetrics reports that this service times inference and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (b *Base) CanGenerateMetrics() bool { return true }
