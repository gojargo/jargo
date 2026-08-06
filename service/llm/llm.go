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
	"errors"
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
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// ErrStopTurn is returned by a ToolHandler to end the current turn after
// recording the tool result instead of generating a further model response. Use
// it for tools that conclude the interaction, such as ending the session.
var ErrStopTurn = errors.New("stop turn")

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

// ToolHandler runs a tool call and returns the content to feed back to the model
// as the tool result. A handler that does blocking work must honor ctx; if it
// ignores cancellation an interruption is delayed until it returns.
type ToolHandler func(ctx context.Context, args json.RawMessage) (string, error)

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
	handlers   map[string]ToolHandler

	ttfbMu    sync.Mutex
	ttfbStart time.Time
	ttfb      time.Duration
	hasTTFB   bool
}

// New builds an LLM Base named name driven by gen. The concrete service passes
// itself as gen and embeds the returned Base.
func New(name string, gen Generator) *Base {
	b := &Base{gen: gen}
	b.Base = processor.New(name, b)
	return b
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
// generation, a call to that tool is dispatched to the handler.
func (b *Base) RegisterFunction(name string, h ToolHandler) {
	b.handlersMu.Lock()
	defer b.handlersMu.Unlock()
	if b.handlers == nil {
		b.handlers = make(map[string]ToolHandler)
	}
	b.handlers[name] = h
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
		started := frames.NewFunctionCallsStartedFrame(preamble.String(), calls)
		if err := b.PushFrame(ctx, started, processor.Downstream); err != nil {
			return err
		}
		b.runTools(ctx, calls)
	}
	traceOutput(span, preamble.String())
	b.emitTiming(ctx, span, time.Since(start))
	return b.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// runTools starts each tool call on its own goroutine and returns. A handler
// runs for as long as the work it does takes, and running it here — on the
// goroutine that processes this service's frames — would hold up every frame
// queued behind it, including the speech a bot plays to cover the wait. The
// assistant aggregator holds the results until the last call has returned, so
// the calls finishing out of order costs nothing.
func (b *Base) runTools(ctx context.Context, calls []frames.ToolCall) {
	for _, c := range calls {
		go b.runTool(ctx, c)
	}
}

// runTool executes one tool call, emitting an in-progress frame and a result
// frame. A handler that returns ErrStopTurn marks its result so the aggregator
// does not re-trigger generation. Nothing is reported once ctx is canceled: an
// interruption already balanced the call the aggregator was waiting on, and a
// late result would be a second one.
func (b *Base) runTool(ctx context.Context, c frames.ToolCall) {
	if ctx.Err() != nil {
		return
	}
	inProgress := frames.NewFunctionCallInProgressFrame(c.ID, c.Name)
	if err := b.PushFrame(ctx, inProgress, processor.Downstream); err != nil {
		return
	}
	result, isErr, stopTurn := b.invoke(ctx, c)
	if ctx.Err() != nil {
		return
	}
	resultFrame := frames.NewFunctionCallResultFrame(c.ID, c.Name, result, isErr)
	resultFrame.RunLLM = !stopTurn
	_ = b.PushFrame(ctx, resultFrame, processor.Downstream)
}

// invoke dispatches a tool call to its handler, returning the result content,
// whether it was an error, and whether the turn should stop without
// re-generating (a handler that returned ErrStopTurn).
func (b *Base) invoke(ctx context.Context, c frames.ToolCall) (result string, isError, stop bool) {
	b.handlersMu.RLock()
	h := b.handlers[c.Name]
	b.handlersMu.RUnlock()
	if h == nil {
		return fmt.Sprintf("unknown tool %q", c.Name), true, false
	}
	out, err := h(ctx, c.Args)
	if errors.Is(err, ErrStopTurn) {
		return out, false, true
	}
	if err != nil {
		return err.Error(), true, false
	}
	return out, false, false
}

// CanGenerateMetrics reports that this service times inference and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (b *Base) CanGenerateMetrics() bool { return true }
