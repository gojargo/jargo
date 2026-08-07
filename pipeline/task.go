package pipeline

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel/attribute"
)

// Default sample rates used for the StartFrame when not overridden.
const (
	defaultAudioInSampleRate  = 16000
	defaultAudioOutSampleRate = 24000
)

// TaskParams configures a Task.
type TaskParams struct {
	// Clock is the pipeline clock; a system clock is used when nil.
	Clock clock.Clock
	// AudioInSampleRate is the StartFrame input sample rate; default 16000.
	AudioInSampleRate int
	// AudioOutSampleRate is the StartFrame output sample rate; default 24000.
	AudioOutSampleRate int
	// EnableMetrics enables performance-metrics collection across the pipeline.
	EnableMetrics bool
	// EnableUsageMetrics enables usage-metrics collection (e.g. LLM token usage)
	// across the pipeline.
	EnableUsageMetrics bool
	// ReportOnlyInitialTTFB reports each service's first time-to-first-byte and
	// no more, for a consumer who wants the figure the call opened with rather
	// than one per turn. It only applies when EnableMetrics is set.
	ReportOnlyInitialTTFB bool
	// SendInitialEmptyMetrics sends a zeroed MetricsFrame for every processor
	// that reports metrics once the pipeline is ready, so a consumer knows which
	// processors to expect metrics from before any have been measured; nil
	// defaults to true. It only applies when EnableMetrics is set.
	SendInitialEmptyMetrics *bool
	// OnReachedDownstream, if set, is called for every frame that reaches the
	// end of the pipeline.
	OnReachedDownstream func(frames.Frame)
	// OnReachedUpstream, if set, is called for every frame that reaches the
	// start of the pipeline.
	OnReachedUpstream func(frames.Frame)
	// Observers watch every frame reaching either end of the pipeline. They are
	// notified after the OnReached callbacks.
	Observers []Observer
	// EnableTurnTracking tracks the conversation's turns and raises a span for
	// each; nil defaults to true. Turning it off leaves a traced task with the
	// conversation span and the service spans beneath it. It says nothing about
	// an untraced task, which has nowhere to report turns to.
	EnableTurnTracking *bool
	// EnableTracing opens a span for the conversation and one for each turn
	// beneath it, and gives the processors the tracing context their own spans
	// hang from, so a session is one trace shaped like the conversation it
	// recorded. It needs a TracerProvider installed (see telemetry/tracing);
	// without one the spans are no-ops.
	EnableTracing bool
	// ConversationID names the traced conversation; empty generates one.
	ConversationID string
	// AdditionalSpanAttributes are set on the conversation span, on top of the
	// ones the task sets itself. They are where the keys a trace backend reads
	// from the root span belong — a session id, a user id, tags.
	AdditionalSpanAttributes []attribute.KeyValue
}

// Task runs a pipeline for a single session. It drives the lifecycle: it sends
// the StartFrame, waits for the pipeline to be ready, pushes queued frames, and
// shuts the pipeline down once an EndFrame or CancelFrame has traveled all the
// way through.
type Task struct {
	pipeline *Pipeline
	source   processor.Processor
	sink     processor.Processor
	params   TaskParams
	clk      clock.Clock

	pushQueue *frameQueue

	// observers are the caller's, plus the ones the task registers itself to
	// track and trace turns.
	observers []Observer
	// tracing is the session's tracing state, handed to the processors at setup;
	// nil when the task is not tracing.
	tracing *tracing.TracingContext
	// turnTrace writes the conversation and turn spans; nil when not tracing.
	turnTrace *observers.TurnTrace

	startOnce sync.Once
	startSig  chan struct{}
	endOnce   sync.Once
	endSig    chan struct{}

	mu        sync.Mutex
	finished  bool
	canceling bool
}

// NewTask wraps pipe in a Task. pipe is usually a *Pipeline but may be any
// processor.
func NewTask(pipe processor.Processor, params TaskParams) *Task {
	if params.AudioInSampleRate == 0 {
		params.AudioInSampleRate = defaultAudioInSampleRate
	}
	if params.AudioOutSampleRate == 0 {
		params.AudioOutSampleRate = defaultAudioOutSampleRate
	}
	t := &Task{
		params:    params,
		clk:       params.Clock,
		pushQueue: newFrameQueue(),
		startSig:  make(chan struct{}),
		endSig:    make(chan struct{}),
	}
	if t.clk == nil {
		t.clk = clock.NewSystem()
	}
	t.buildObservers()
	// The source observes upstream frames, the sink observes downstream frames.
	// They bracket the user pipeline so the task can inject and observe frames.
	t.source = processor.NewSource("Task::Source", t.sourcePush)
	t.sink = processor.NewSink("Task::Sink", t.sinkPush)
	t.pipeline = build(t.source, t.sink, []processor.Processor{pipe})
	return t
}

// buildObservers assembles the observers the pipeline runs with: the caller's,
// and — when the task traces — the turn tracing, plus the turn tracking and
// latency measurement that feed it.
func (t *Task) buildObservers() {
	t.observers = append([]Observer(nil), t.params.Observers...)
	if !t.params.EnableTracing {
		return
	}
	t.tracing = tracing.NewTracingContext()
	t.turnTrace = observers.NewTurnTrace(observers.TurnTraceConfig{
		Tracing:        t.tracing,
		ConversationID: t.params.ConversationID,
		Attributes:     t.params.AdditionalSpanAttributes,
	})
	t.observers = append(t.observers, t.turnTrace)
	if t.params.EnableTurnTracking != nil && !*t.params.EnableTurnTracking {
		return
	}
	t.observers = append(t.observers,
		observers.NewTurnTracking(observers.TurnTrackingConfig{
			OnTurnStarted: t.turnTrace.TurnStarted,
			OnTurnEnded:   t.turnTrace.TurnEnded,
		}),
		observers.NewUserBotLatency(observers.LatencyConfig{
			OnLatency: t.turnTrace.LatencyMeasured,
		}),
	)
}

// Tracing is the session's tracing state: the conversation span the trace hangs
// from and the turn being spoken. It is nil unless the task traces, and is what
// the processors are given at setup so their spans land under the right turn.
func (t *Task) Tracing() *tracing.TracingContext { return t.tracing }

// QueueFrame queues a frame to be pushed downstream through the pipeline.
func (t *Task) QueueFrame(f frames.Frame) { t.pushQueue.push(f) }

// QueueFrames queues several frames to be pushed downstream, in order.
func (t *Task) QueueFrames(fs []frames.Frame) {
	for _, f := range fs {
		t.pushQueue.push(f)
	}
}

// StopWhenDone schedules the pipeline to stop once all queued frames have been
// processed, by queueing an EndFrame.
func (t *Task) StopWhenDone() { t.QueueFrame(frames.NewEndFrame()) }

// Flush waits for the pipeline to drain: it queues a PipelineFlushFrame probe
// and blocks until the probe has traveled down to the sink and back up to the
// source, meaning every frame queued ahead of it has been processed. Use it to
// let the pipeline settle — after an interruption, say — before injecting new
// work. It returns ctx.Err() if ctx is done first.
func (t *Task) Flush(ctx context.Context) error {
	probe := frames.NewPipelineFlushFrame()
	t.QueueFrame(probe)
	select {
	case <-probe.Done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Cancel stops the pipeline immediately by queueing a CancelFrame.
func (t *Task) Cancel() { t.cancelWithReason("") }

// cancelWithReason queues a CancelFrame carrying reason, at most once.
func (t *Task) cancelWithReason(reason string) {
	t.mu.Lock()
	if t.canceling {
		t.mu.Unlock()
		return
	}
	t.canceling = true
	t.mu.Unlock()
	cancel := frames.NewCancelFrame()
	cancel.Reason = reason
	t.QueueFrame(cancel)
}

// HasFinished reports whether the task has finished running.
func (t *Task) HasFinished() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.finished
}

// Run sets up the pipeline and drives it until an EndFrame or CancelFrame
// completes its journey through the pipeline, or ctx is canceled. It then
// cleans up the pipeline. Run blocks until the pipeline has finished.
func (t *Task) Run(ctx context.Context) error {
	t.mu.Lock()
	if t.finished {
		t.mu.Unlock()
		return nil
	}
	t.mu.Unlock()

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	setup := processor.Setup{Clock: t.clk, Observers: t.observers, Tracing: t.tracing, Running: t}
	if err := t.pipeline.Setup(runCtx, setup); err != nil {
		return err
	}

	runErr := t.runLoop(runCtx)

	// Clean up with a fresh context so a canceled runCtx does not abort the
	// goroutine shutdown.
	_ = t.pipeline.Cleanup(context.Background())

	// The conversation span closes last, once the processors have stopped and
	// the spans raised beneath it have ended.
	t.turnTrace.EndConversation()

	t.mu.Lock()
	t.finished = true
	t.mu.Unlock()
	return runErr
}

// runLoop sends the StartFrame, waits for the pipeline to be ready, then pushes
// queued frames until a pipeline-ending frame has traveled through.
func (t *Task) runLoop(ctx context.Context) error {
	t.clk.Start()

	start := frames.NewStartFrame()
	start.AudioInSampleRate = t.params.AudioInSampleRate
	start.AudioOutSampleRate = t.params.AudioOutSampleRate
	start.EnableMetrics = t.params.EnableMetrics
	start.EnableUsageMetrics = t.params.EnableUsageMetrics
	start.ReportOnlyInitialTTFB = t.params.ReportOnlyInitialTTFB
	if err := t.pipeline.QueueFrame(ctx, start, processor.Downstream); err != nil {
		return err
	}

	select {
	case <-t.startSig:
	case <-ctx.Done():
		return ctx.Err()
	}

	if t.params.EnableMetrics && boolValue(t.params.SendInitialEmptyMetrics, true) {
		if err := t.pipeline.QueueFrame(ctx, t.initialMetricsFrame(), processor.Downstream); err != nil {
			return err
		}
	}

	for {
		f, ok := t.pushQueue.get(ctx)
		if !ok {
			return ctx.Err()
		}
		if err := t.pipeline.QueueFrame(ctx, f, processor.Downstream); err != nil {
			return err
		}
		if isPipelineEnd(f) {
			select {
			case <-t.endSig:
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		}
	}
}

// initialMetricsFrame builds one MetricsFrame carrying a zeroed time to first
// byte and processing time for every processor in the pipeline that reports
// metrics, so a consumer knows the full set before any have been measured.
func (t *Task) initialMetricsFrame() *frames.MetricsFrame {
	var data []frames.MetricsData
	for _, p := range t.pipeline.ProcessorsWithMetrics() {
		base := frames.BaseMetricsData{Processor: p.Name()}
		data = append(data,
			frames.TTFBMetricsData{BaseMetricsData: base},
			frames.ProcessingMetricsData{BaseMetricsData: base},
		)
	}
	return frames.NewMetricsFrame(data...)
}

// boolValue returns *p, or def when p is nil.
func boolValue(p *bool, def bool) bool {
	if p == nil {
		return def
	}
	return *p
}

// sinkPush observes frames reaching the end of the pipeline and signals the
// lifecycle events the run loop waits on.
func (t *Task) sinkPush(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	// The flush probe reached the sink. Bounce the same instance back upstream so
	// it returns to the source and the round trip covers both directions.
	if probe, ok := f.(*frames.PipelineFlushFrame); ok {
		return t.sink.PushFrame(ctx, probe, processor.Upstream)
	}

	// A worker frame pushed downstream — the documented default, so frames queued
	// ahead of it are processed first — has now reached the end. Send a fresh
	// instance back upstream so it arrives at the source, which converts it into
	// the pipeline-wide frame.
	if back := workerFrameEcho(f); back != nil {
		return t.sink.PushFrame(ctx, back, processor.Upstream)
	}

	switch f.(type) {
	case *frames.StartFrame:
		t.startOnce.Do(func() { close(t.startSig) })
	case *frames.EndFrame, *frames.CancelFrame, *frames.StopFrame:
		t.endOnce.Do(func() { close(t.endSig) })
	}
	if t.params.OnReachedDownstream != nil {
		t.params.OnReachedDownstream(f)
	}
	return nil
}

// sourcePush observes frames reaching the start of the pipeline and converts the
// worker frames a processor pushed upstream into the corresponding pipeline-wide
// frame. A fatal error cancels the pipeline.
func (t *Task) sourcePush(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	// The flush probe completed its round trip: everything queued ahead of it has
	// been processed, so release whoever is waiting on it.
	if probe, ok := f.(*frames.PipelineFlushFrame); ok {
		probe.CloseDone()
		return nil
	}

	if ef, ok := f.(*frames.ErrorFrame); ok && ef.Fatal {
		t.Cancel()
	}

	switch fr := f.(type) {
	case *frames.EndWorkerFrame:
		// End gracefully: queue an EndFrame so frames already queued flush before
		// the pipeline stops.
		end := frames.NewEndFrame()
		end.Reason = fr.Reason
		t.QueueFrame(end)
	case *frames.CancelWorkerFrame:
		// Stop right away, without flushing.
		t.cancelWithReason(fr.Reason)
	case *frames.StopWorkerFrame:
		// Stop once queued frames are flushed, leaving processors running.
		t.QueueFrame(frames.NewStopFrame())
	case *frames.InterruptionWorkerFrame:
		// Queue straight into the pipeline rather than the push queue, which may
		// be blocked waiting for a pipeline-ending frame to finish traversing.
		if err := t.pipeline.QueueFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream); err != nil {
			return err
		}
	}
	if t.params.OnReachedUpstream != nil {
		t.params.OnReachedUpstream(f)
	}
	return nil
}

// workerFrameEcho returns a fresh worker frame to send back upstream when one
// reaches the sink, or nil if f is not a worker frame. A new instance is built
// rather than the original being reused so the two directions never share a
// frame; the reason, where there is one, is carried across.
func workerFrameEcho(f frames.Frame) frames.Frame {
	switch fr := f.(type) {
	case *frames.EndWorkerFrame:
		back := frames.NewEndWorkerFrame()
		back.Reason = fr.Reason
		return back
	case *frames.StopWorkerFrame:
		return frames.NewStopWorkerFrame()
	case *frames.CancelWorkerFrame:
		back := frames.NewCancelWorkerFrame()
		back.Reason = fr.Reason
		return back
	case *frames.InterruptionWorkerFrame:
		return frames.NewInterruptionWorkerFrame()
	}
	return nil
}

func isPipelineEnd(f frames.Frame) bool {
	switch f.(type) {
	case *frames.EndFrame, *frames.CancelFrame, *frames.StopFrame:
		return true
	}
	return false
}

// frameQueue is an unbounded, concurrency-safe FIFO of frames with a single
// consumer, used for frames the user queues for the task to push.
type frameQueue struct {
	mu     sync.Mutex
	items  []frames.Frame
	notify chan struct{}
}

func newFrameQueue() *frameQueue {
	return &frameQueue{notify: make(chan struct{}, 1)}
}

func (q *frameQueue) push(f frames.Frame) {
	q.mu.Lock()
	q.items = append(q.items, f)
	q.mu.Unlock()
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// tryGet pops the next frame without blocking, reporting false when the queue is
// empty.
func (q *frameQueue) tryGet() (frames.Frame, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil, false
	}
	f := q.items[0]
	q.items = q.items[1:]
	return f, true
}

func (q *frameQueue) get(ctx context.Context) (frames.Frame, bool) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			f := q.items[0]
			q.items = q.items[1:]
			q.mu.Unlock()
			return f, true
		}
		q.mu.Unlock()

		select {
		case <-ctx.Done():
			return nil, false
		case <-q.notify:
		}
	}
}
