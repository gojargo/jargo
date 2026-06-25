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

// Setup carries the shared components a processor needs, propagated down the
// pipeline when it is set up.
type Setup struct {
	// Clock is the pipeline clock used for timing.
	Clock clock.Clock
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

	// Lifetime context for the processor's goroutines, canceled on Cleanup.
	baseCtx    context.Context
	baseCancel context.CancelFunc

	// Input goroutine: handles system frames immediately and forwards data and
	// control frames to the process queue.
	inputQueue *queue
	inputWG    sync.WaitGroup

	startedMu sync.Mutex
	started   bool

	// Metrics flags captured from the StartFrame. They are written once on the
	// input goroutine before the process goroutine is created in start(), which
	// establishes the happens-before for reads from ProcessFrame.
	metricsEnabled      bool
	usageMetricsEnabled bool

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

// Setup implements Processor. It stores shared components and starts the input
// goroutine (unless the processor is in direct mode).
func (b *Base) Setup(ctx context.Context, s Setup) error {
	b.clock = s.Clock
	b.baseCtx, b.baseCancel = context.WithCancel(ctx)
	if !b.directMode {
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
		_ = b.processFrame(ctx, it)
	}
}

// processFrame dispatches a frame to the concrete processor and turns a
// processing error into an ErrorFrame pushed upstream.
func (b *Base) processFrame(ctx context.Context, it item) error {
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
		b.start()
	case *frames.InterruptionFrame:
		b.startInterruption()
	case *frames.CancelFrame:
		b.cancel()
	}
	return nil
}

// MetricsEnabled reports whether performance-metrics collection was enabled by
// the StartFrame. It is valid once the processor has received its StartFrame.
func (b *Base) MetricsEnabled() bool { return b.metricsEnabled }

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
			return b.next.QueueFrame(ctx, f, dir)
		}
	case Upstream:
		if b.prev != nil {
			return b.prev.QueueFrame(ctx, f, dir)
		}
	}
	return nil
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
