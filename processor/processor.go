// Package processor defines the frame processor: the building block of a jargo
// pipeline. Processors link into a chain, receive frames, process them, and
// push them on to the next or previous processor. Each processor handles system
// frames with priority, processes data and control frames in order on its own
// goroutine, and can be interrupted.
package processor

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/telemetry/metrics"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel/attribute"
)

// Direction is the direction a frame flows through the pipeline.
type Direction int

const (
	// Downstream is the direction from input toward output.
	Downstream Direction = iota
	// Upstream is the direction from output back toward input.
	Upstream
)

// String returns "downstream" or "upstream".
func (d Direction) String() string {
	if d == Upstream {
		return "upstream"
	}
	return "downstream"
}

// processCancelTimeout bounds how long an interruption or cleanup waits for a
// processor's in-flight frame to finish after its context is canceled. It
// guards against a ProcessFrame implementation that ignores cancellation.
const processCancelTimeout = 3 * time.Second

// Running is the pipeline a processor belongs to, for the few things a
// processor needs from it that the frame path cannot express on its own. It is
// an interface rather than the concrete task because the task is built on top of
// processors, so naming that type here would be circular.
type Running interface {
	// Flush blocks until every frame queued ahead of the call has traveled the
	// whole pipeline, so the pipeline has settled. Never call it from the
	// goroutine that processes frames: the probe has to pass through this
	// processor to complete its round trip.
	Flush(ctx context.Context) error
}

// Setup carries the shared components a processor needs, propagated down the
// pipeline when it is set up.
type Setup struct {
	// Clock is the pipeline clock used for timing.
	Clock clock.Clock
	// Observers watch every frame handed between processors.
	Observers []Observer
	// Running is the pipeline this processor is part of. Nil for a processor
	// driven outside a pipeline task.
	Running Running
	// Tracing is the session's tracing state: the conversation span the trace
	// hangs from and the turn being spoken. Processors parent their spans to it,
	// so a span raised away from the frame path still lands under the turn it
	// belongs to. Nil when the pipeline is not traced.
	Tracing *tracing.TracingContext
}

// Processor is a node in a pipeline. Concrete processors embed *Base, which
// provides every method here except a custom ProcessFrame.
type Processor interface {
	// ID is a process-unique identifier for this processor.
	ID() uint64
	// Name is a human-readable label, "<name>#<id>".
	Name() string

	// Next is the downstream processor, or nil.
	Next() Processor
	// Prev is the upstream processor, or nil.
	Prev() Processor

	// Processors are the sub-processors this processor contains. Only a
	// compound processor (a pipeline, a parallel pipeline) has any; every
	// other processor reports none.
	Processors() []Processor
	// EntryProcessors are the processors a frame entering a compound
	// processor reaches first. A pipeline is a processor itself, so an entry
	// processor can be a pipeline in turn. Every other processor reports none.
	EntryProcessors() []Processor
	// ProcessorsWithMetrics are the processors below this one that report
	// metrics, collected recursively.
	ProcessorsWithMetrics() []Processor
	// CanGenerateMetrics reports whether this processor reports metrics. It is
	// false for everything but a service.
	CanGenerateMetrics() bool
	// Link sets next as this processor's downstream neighbor and this
	// processor as next's upstream neighbor.
	Link(next Processor)

	// Setup wires the processor with shared components and starts its
	// goroutines. It must be called before frames are queued.
	Setup(ctx context.Context, s Setup) error
	// Cleanup stops the processor's goroutines and releases resources.
	Cleanup(ctx context.Context) error

	// QueueFrame hands a frame to this processor for processing.
	QueueFrame(ctx context.Context, f frames.Frame, dir Direction) error
	// ProcessFrame processes a frame. The base implementation handles system
	// lifecycle frames; concrete processors override it and call the base
	// first.
	ProcessFrame(ctx context.Context, f frames.Frame, dir Direction) error
	// PushFrame sends a frame to the neighboring processor in dir.
	PushFrame(ctx context.Context, f frames.Frame, dir Direction) error
}

//nolint:gochecknoglobals // process-wide id source
var idCounter atomic.Uint64

func nextID() uint64 { return idCounter.Add(1) }

// Option configures a Base at construction.
type Option func(*Base)

// WithDirectMode makes a processor process frames immediately on the caller's
// goroutine instead of queueing them. It is used for routing processors (a
// pipeline and its source and sink) that only forward frames.
func WithDirectMode() Option {
	return func(b *Base) { b.directMode = true }
}

// Base implements Processor. Embed it in a concrete processor and pass the
// concrete value as self so the base can dispatch to the overridden
// ProcessFrame:
//
//	type Echo struct{ *processor.Base }
//
//	func NewEcho() *Echo {
//	    e := &Echo{}
//	    e.Base = processor.New("Echo", e)
//	    return e
//	}
//
//	func (e *Echo) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
//	    if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
//	        return err
//	    }
//	    return e.PushFrame(ctx, f, dir)
//	}
type Base struct {
	id   uint64
	name string
	self Processor

	next, prev Processor

	directMode bool
	clock      clock.Clock
	observers  []Observer
	tracing    *tracing.TracingContext
	running    Running

	// Lifetime context for the processor's goroutines, canceled on Cleanup.
	baseCtx    context.Context
	baseCancel context.CancelFunc

	// Input goroutine: handles system frames immediately and forwards data and
	// control frames to the process queue.
	inputQueue *queue
	inputWG    sync.WaitGroup

	// Pause gates. Each loop checks its gate after taking a frame off its queue
	// and before handling it, so a pause takes effect from the next frame on.
	// The events are created with the goroutine they hold.
	pauseMu           sync.Mutex
	blockSystemFrames bool
	inputEvent        *event
	blockFrames       bool
	processEvent      *event

	startedMu sync.Mutex
	started   bool

	// Metrics flags captured from the StartFrame. They are written once on the
	// input goroutine before the process goroutine is created in start(), which
	// establishes the happens-before for reads from ProcessFrame.
	metricsEnabled      bool
	usageMetricsEnabled bool
	// reportOnlyInitialTTFB asks for the first time-to-first-byte of the run and
	// no more, for a caller who wants the figure a call opened with rather than
	// one per turn.
	reportOnlyInitialTTFB bool

	// ttfbMu guards armTTFB, which unlike the flags above is written on every
	// measurement rather than once at the start.
	ttfbMu sync.Mutex
	// armTTFB says whether another time-to-first-byte may be measured. It stays
	// true throughout unless only the initial one was asked for, in which case
	// arming the first measurement is what clears it.
	armTTFB bool

	cancelMu  sync.Mutex
	canceling bool

	// Process goroutine: handles data and control frames in order. It is
	// created on StartFrame and recreated on interruption.
	procMu      sync.Mutex
	procQueue   *queue
	procRunning bool
	procCancel  context.CancelFunc
	procDone    chan struct{}

	curMu    sync.Mutex
	curFrame frames.Frame
}

// New builds a Base named name. self is the embedding processor, used to
// dispatch to its ProcessFrame; pass nil for a plain pass-through that does not
// override ProcessFrame.
func New(name string, self Processor, opts ...Option) *Base {
	b := &Base{
		id:         nextID(),
		inputQueue: newQueue(),
		procQueue:  newQueue(),
		// Armed before any StartFrame, so a measurement started early is not
		// declined for a restriction nothing has asked for yet.
		armTTFB: true,
	}
	b.name = fmt.Sprintf("%s#%d", name, b.id)
	for _, opt := range opts {
		opt(b)
	}
	if self != nil {
		b.self = self
	} else {
		b.self = b
	}
	return b
}

// ID implements Processor.
func (b *Base) ID() uint64 { return b.id }

// Name implements Processor.
func (b *Base) Name() string { return b.name }

// Processors implements Processor. A plain processor contains none; a compound
// processor overrides this.
func (b *Base) Processors() []Processor { return nil }

// EntryProcessors implements Processor. A plain processor has none; a compound
// processor overrides this.
func (b *Base) EntryProcessors() []Processor { return nil }

// ProcessorsWithMetrics implements Processor. A plain processor contains
// nothing, so it reports nothing; a compound processor overrides this and
// collects from what it contains.
func (b *Base) ProcessorsWithMetrics() []Processor { return nil }

// CanGenerateMetrics implements Processor. A processor reports no metrics unless
// it is a service, which overrides this.
func (b *Base) CanGenerateMetrics() bool { return false }

// Next implements Processor.
func (b *Base) Next() Processor { return b.next }

// Prev implements Processor.
func (b *Base) Prev() Processor { return b.prev }

// Link implements Processor.
func (b *Base) Link(next Processor) {
	b.next = next
	if sp, ok := next.(interface{ setPrev(Processor) }); ok {
		sp.setPrev(b.self)
	}
}

func (b *Base) setPrev(p Processor) { b.prev = p }

// Clock returns the pipeline clock, available after Setup.
func (b *Base) Clock() clock.Clock { return b.clock }

// Tracing returns the session's tracing state, available after Setup. It is nil
// when the pipeline is not traced, which its methods handle: parent a span with
// Tracing().Parent(ctx) without checking.
func (b *Base) Tracing() *tracing.TracingContext { return b.tracing }

// Setup implements Processor. It stores shared components and starts the input
// goroutine (unless the processor is in direct mode).
func (b *Base) Setup(ctx context.Context, s Setup) error {
	b.clock = s.Clock
	b.observers = s.Observers
	b.tracing = s.Tracing
	b.running = s.Running
	b.baseCtx, b.baseCancel = context.WithCancel(ctx)
	if !b.directMode {
		b.pauseMu.Lock()
		b.inputEvent = newEvent()
		b.pauseMu.Unlock()
		b.inputWG.Add(1)
		go b.inputLoop()
	}
	return nil
}

// Cleanup implements Processor. It stops the process and input goroutines.
func (b *Base) Cleanup(ctx context.Context) error {
	b.cancelProcessTask()
	if b.baseCancel != nil {
		b.baseCancel()
	}
	b.inputWG.Wait()
	return nil
}

// QueueFrame implements Processor.
func (b *Base) QueueFrame(ctx context.Context, f frames.Frame, dir Direction) error {
	if b.isCanceling() {
		return nil
	}
	if b.directMode {
		return b.processFrame(ctx, item{frame: f, dir: dir})
	}
	b.inputQueue.push(item{frame: f, dir: dir})
	return nil
}

// inputLoop handles every frame queued to the processor. System frames are
// processed immediately; data and control frames are forwarded to the process
// queue for in-order processing.
func (b *Base) inputLoop() {
	defer b.inputWG.Done()
	for {
		it, ok := b.inputQueue.get(b.baseCtx)
		if !ok {
			return
		}
		if !b.waitWhilePaused(b.baseCtx, true) {
			return
		}
		if _, isSystem := it.frame.(frames.SystemFrame); isSystem {
			_ = b.processFrame(b.baseCtx, it)
		} else {
			b.procQueue.push(it)
		}
	}
}

// processLoop handles data and control frames in order. It runs under ctx,
// which is canceled to interrupt the processor.
func (b *Base) processLoop(ctx context.Context, done chan struct{}) {
	defer close(done)
	for {
		b.setCurFrame(nil)
		it, ok := b.procQueue.get(ctx)
		if !ok {
			return
		}
		b.setCurFrame(it.frame)
		if !b.waitWhilePaused(ctx, false) {
			return
		}
		_ = b.processFrame(ctx, it)
	}
}

// processFrame dispatches a frame to the concrete processor and turns a
// processing error into an ErrorFrame pushed upstream.
func (b *Base) processFrame(ctx context.Context, it item) error {
	b.notifyProcess(it.frame, it.dir)
	if err := b.self.ProcessFrame(ctx, it.frame, it.dir); err != nil {
		b.PushError(ctx, fmt.Sprintf("error processing frame: %v", err), err, false)
		return err
	}
	return nil
}

// ProcessFrame implements Processor. It handles the system frames that drive a
// processor's lifecycle: StartFrame, InterruptionFrame and CancelFrame. A
// concrete processor overrides this, calls the base first, then forwards the
// frame with PushFrame.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir Direction) error {
	switch fr := f.(type) {
	case *frames.StartFrame:
		b.metricsEnabled = fr.EnableMetrics
		b.usageMetricsEnabled = fr.EnableUsageMetrics
		b.reportOnlyInitialTTFB = fr.ReportOnlyInitialTTFB
		b.ttfbMu.Lock()
		b.armTTFB = true
		b.ttfbMu.Unlock()
		b.start()
	case *frames.InterruptionFrame:
		b.startInterruption()
	case *frames.CancelFrame:
		b.cancel()
	case *frames.FrameProcessorPauseFrame:
		b.pauseIfAddressed(fr.Processor)
	case *frames.FrameProcessorPauseUrgentFrame:
		b.pauseIfAddressed(fr.Processor)
	case *frames.FrameProcessorResumeFrame:
		b.resumeIfAddressed(fr.Processor)
	case *frames.FrameProcessorResumeUrgentFrame:
		b.resumeIfAddressed(fr.Processor)
	}
	return nil
}

// HasQueuedFrame reports whether a frame satisfying match is still waiting in
// this processor's in-order queue, behind the one being handled now. A
// processor uses it to tell that more of the same work is already on its way, so
// it can hold off on an action until the last of it arrives rather than
// repeating the action once per frame.
//
// Only data and control frames are considered. A system frame is handled the
// moment it is queued and so never waits.
func (b *Base) HasQueuedFrame(match func(frames.Frame) bool) bool {
	return b.procQueue.hasFrame(match)
}

// MetricsEnabled reports whether performance-metrics collection was enabled by
// the StartFrame. It is valid once the processor has received its StartFrame.
func (b *Base) MetricsEnabled() bool { return b.metricsEnabled }

// BeginTTFB reports whether a time-to-first-byte measurement should be started,
// and records that one was. A service calls it where it would start the clock,
// and measures nothing when it reports false.
//
// It answers true every time unless the StartFrame asked for only the initial
// TTFB, in which case the first measurement is the only one armed: a caller who
// wants the figure the call opened with gets it once rather than on every turn.
func (b *Base) BeginTTFB() bool {
	b.ttfbMu.Lock()
	defer b.ttfbMu.Unlock()
	if !b.armTTFB {
		return false
	}
	b.armTTFB = !b.reportOnlyInitialTTFB
	return true
}

// UsageMetricsEnabled reports whether usage-metrics collection was enabled by
// the StartFrame. It is valid once the processor has received its StartFrame.
func (b *Base) UsageMetricsEnabled() bool { return b.usageMetricsEnabled }

// PushFrame implements Processor. It forwards a frame to the neighbor in dir.
// Frames pushed before the processor has received its StartFrame are dropped.
func (b *Base) PushFrame(ctx context.Context, f frames.Frame, dir Direction) error {
	if !b.checkStarted(f) {
		return nil
	}
	switch dir {
	case Downstream:
		if b.next != nil {
			b.notifyPush(f, dir, b.next)
			return b.next.QueueFrame(ctx, f, dir)
		}
	case Upstream:
		if b.prev != nil {
			b.notifyPush(f, dir, b.prev)
			return b.prev.QueueFrame(ctx, f, dir)
		}
	}
	return nil
}

// FlushPipeline blocks until every frame queued ahead of the call has traveled
// the whole pipeline. Use it to let the pipeline settle, after an interruption
// say, before injecting new work.
//
// Never call it from the goroutine that processes frames: the probe it waits on
// has to pass through this processor to complete its round trip, so a processor
// blocking its own frame path would wait forever. A processor that needs this
// runs it from a goroutine of its own.
//
// It is a no-op for a processor driven outside a pipeline task, which has
// nothing to drain.
func (b *Base) FlushPipeline(ctx context.Context) error {
	if b.running == nil {
		return nil
	}
	return b.running.Flush(ctx)
}

// Broadcast sends a frame both downstream and upstream, so an event that the
// whole pipeline has to see reaches processors on either side of this one.
//
// build is called once per direction: the two halves are distinct frames paired
// by BroadcastSiblingID. The directions are processed on separate goroutines, so
// a single shared frame would be mutated concurrently, and a consumer that sees
// both halves can recognize the pair rather than reporting the event twice.
func (b *Base) Broadcast(ctx context.Context, build func() frames.Frame) error {
	down, up := build(), build()
	down.Base().SetBroadcastSiblingID(up.ID())
	up.Base().SetBroadcastSiblingID(down.ID())

	if err := b.PushFrame(ctx, down, Downstream); err != nil {
		return err
	}
	return b.PushFrame(ctx, up, Upstream)
}

// PushTokenUsage reports LLM token usage measured by a service that does not run
// through the LLM base — a realtime (speech-to-speech) service that receives a
// usage event from its provider. It opens a short "llm" span carrying the
// gen_ai.usage.* attributes, records the aggregate token counts as metrics, and
// emits a MetricsFrame downstream for in-band consumers (e.g. an RTVI client).
// The caller passes the model id and gates the call on UsageMetricsEnabled, so
// the conversion from the provider's usage shape happens only when metrics are
// collected.
func (b *Base) PushTokenUsage(ctx context.Context, model string, u frames.LLMTokenUsage) error {
	ctx, span := tracing.Tracer().Start(b.tracing.Parent(ctx), "llm")
	span.SetAttributes(attribute.String("llm.service", b.name))
	if model != "" {
		span.SetAttributes(attribute.String("llm.model", model))
	}
	tracing.SetTokenUsage(ctx, u)
	span.End()
	metrics.RecordTokens(ctx, b.name, model, u.PromptTokens, u.CompletionTokens)
	f := frames.NewMetricsFrame(frames.LLMUsageMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: b.name, Model: model},
		Value:           u,
	})
	return b.PushFrame(ctx, f, Downstream)
}

// PushError builds an ErrorFrame for msg and pushes it upstream.
func (b *Base) PushError(ctx context.Context, msg string, err error, fatal bool) {
	ef := frames.NewErrorFrame(msg)
	ef.Fatal = fatal
	ef.Err = err
	ef.Source = b.self
	slog.Error("processor error", "processor", b.name, "msg", msg, "err", err, "fatal", fatal)
	_ = b.PushFrame(ctx, ef, Upstream)
}

func (b *Base) start() {
	b.startedMu.Lock()
	b.started = true
	b.startedMu.Unlock()
	b.createProcessTask()
}

func (b *Base) cancel() {
	b.setCanceling(true)
	b.cancelProcessTask()
}

// startInterruption interrupts in-order processing. If the frame currently
// being processed is uninterruptible it is left to finish and only the queued
// interruptible frames are flushed; otherwise the process goroutine is
// canceled and recreated.
func (b *Base) startInterruption() {
	if b.directMode {
		return
	}
	if isUninterruptible(b.currentFrame()) {
		b.procQueue.reset()
		return
	}
	b.cancelProcessTask()
	b.createProcessTask()
}

// createProcessTask starts the process goroutine if it is not already running.
func (b *Base) createProcessTask() {
	if b.directMode {
		return
	}
	b.procMu.Lock()
	defer b.procMu.Unlock()
	if b.procRunning {
		return
	}
	b.procQueue.reset()
	// A fresh process goroutine starts unpaused, with its own gate: a pause that
	// was in effect for the goroutine being replaced must not hold the new one.
	b.pauseMu.Lock()
	b.blockFrames = false
	b.processEvent = newEvent()
	b.pauseMu.Unlock()
	ctx, cancel := context.WithCancel(b.baseCtx)
	done := make(chan struct{})
	b.procCancel = cancel
	b.procDone = done
	b.procRunning = true
	go b.processLoop(ctx, done)
}

// cancelProcessTask cancels the process goroutine and waits for it to exit,
// bounded by processCancelTimeout.
func (b *Base) cancelProcessTask() {
	if b.directMode {
		return
	}
	b.procMu.Lock()
	if !b.procRunning {
		b.procMu.Unlock()
		return
	}
	cancel := b.procCancel
	done := b.procDone
	b.procRunning = false
	b.procMu.Unlock()

	cancel()
	select {
	case <-done:
	case <-time.After(processCancelTimeout):
		slog.Warn("timed out canceling process goroutine", "processor", b.name)
	}
}

func (b *Base) checkStarted(f frames.Frame) bool {
	b.startedMu.Lock()
	started := b.started
	b.startedMu.Unlock()
	if !started {
		slog.Error("frame pushed before StartFrame", "processor", b.name, "frame", f.Name())
	}
	return started
}

func (b *Base) isCanceling() bool {
	b.cancelMu.Lock()
	defer b.cancelMu.Unlock()
	return b.canceling
}

func (b *Base) setCanceling(v bool) {
	b.cancelMu.Lock()
	b.canceling = v
	b.cancelMu.Unlock()
}

func (b *Base) setCurFrame(f frames.Frame) {
	b.curMu.Lock()
	b.curFrame = f
	b.curMu.Unlock()
}

func (b *Base) currentFrame() frames.Frame {
	b.curMu.Lock()
	defer b.curMu.Unlock()
	return b.curFrame
}

func isUninterruptible(f frames.Frame) bool {
	if f == nil {
		return false
	}
	_, ok := f.(frames.Uninterruptible)
	return ok
}

var _ Processor = (*Base)(nil)
