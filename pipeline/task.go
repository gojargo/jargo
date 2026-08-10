package pipeline

import (
	"context"
	"log/slog"
	"sync"
	"time"

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

// defaultCancelTimeout bounds how long the task waits for a CancelFrame to
// reach the end of the pipeline before giving up on it.
const defaultCancelTimeout = 20 * time.Second

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
	// CancelTimeout bounds the wait for a CancelFrame to travel the pipeline;
	// zero defaults to 20 seconds. Canceling is what a caller reaches for when
	// something has already gone wrong, so it must not be the thing that hangs:
	// a processor wedged somewhere would otherwise hold the shutdown open for
	// good. Reaching the timeout gives up on the frame and finishes the run. An
	// EndFrame is not bounded this way, since a graceful shutdown is expected to
	// flush whatever is in flight however long that takes.
	CancelTimeout time.Duration
	// OnPipelineStarted, if set, is called once the StartFrame has traveled the
	// whole pipeline, meaning every processor is up and the pipeline is ready.
	OnPipelineStarted func(*frames.StartFrame)
	// OnPipelineFinished, if set, is called once the pipeline has reached a
	// terminal state, with the frame that ended it: an EndFrame, a StopFrame or
	// a CancelFrame. It is called for a CancelFrame even when the frame never
	// arrived and the wait timed out, so a caller always hears the run end.
	OnPipelineFinished func(frames.Frame)
	// OnPipelineError, if set, is called for every error frame reaching the
	// start of the pipeline, fatal or not. A fatal one also cancels the task,
	// after the handler has run.
	OnPipelineError func(*frames.ErrorFrame)
	// OnReachedDownstream, if set, is called for the frames reaching the end of
	// the pipeline that ReachedDownstreamFilter selects.
	OnReachedDownstream func(frames.Frame)
	// OnReachedUpstream, if set, is called for the frames reaching the start of
	// the pipeline that ReachedUpstreamFilter selects.
	OnReachedUpstream func(frames.Frame)
	// ReachedDownstreamFilter selects which frames reaching the end of the
	// pipeline are reported. Nil selects nothing, so a handler without a filter
	// is never called; that is the default because a handler on every frame sits
	// on the path of everything the pipeline does. Use AnyFrame to watch the
	// whole stream anyway.
	ReachedDownstreamFilter FrameFilter
	// ReachedUpstreamFilter selects which frames reaching the start of the
	// pipeline are reported. Nil selects nothing. See ReachedDownstreamFilter.
	ReachedUpstreamFilter FrameFilter
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

	// reachedDownstream are the handlers registered after construction, called
	// for the frames reaching the end of the pipeline alongside the one
	// TaskParams carries. See OnReachedDownstream. The filters alongside them
	// select which frames are reported, and start out as the ones TaskParams
	// carries.
	reachedMu         sync.Mutex
	reachedDownstream []func(frames.Frame)
	reachedDownFilter FrameFilter
	reachedUpFilter   FrameFilter

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
	if params.CancelTimeout == 0 {
		params.CancelTimeout = defaultCancelTimeout
	}
	t := &Task{
		params:            params,
		clk:               params.Clock,
		pushQueue:         newFrameQueue(),
		startSig:          make(chan struct{}),
		endSig:            make(chan struct{}),
		reachedDownFilter: params.ReachedDownstreamFilter,
		reachedUpFilter:   params.ReachedUpstreamFilter,
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

// OnReachedDownstream registers a handler called for every frame that reaches
// the end of the pipeline, in addition to the one TaskParams carries.
//
// It exists because the things that watch the end of the pipeline are usually
// built after the task is: something that queues frames needs the task to queue
// them on, so it cannot also have been passed to NewTask. Handlers are called in
// the order they were registered, on the goroutine the frame arrived on, so a
// handler that does real work should hand it off rather than block the sink.
func (t *Task) OnReachedDownstream(fn func(frames.Frame)) {
	if fn == nil {
		return
	}
	t.reachedMu.Lock()
	t.reachedDownstream = append(t.reachedDownstream, fn)
	t.reachedMu.Unlock()
}

// SetReachedDownstreamFilter replaces the filter selecting which frames reaching
// the end of the pipeline are reported.
func (t *Task) SetReachedDownstreamFilter(f FrameFilter) {
	t.reachedMu.Lock()
	t.reachedDownFilter = f
	t.reachedMu.Unlock()
}

// SetReachedUpstreamFilter replaces the filter selecting which frames reaching
// the start of the pipeline are reported.
func (t *Task) SetReachedUpstreamFilter(f FrameFilter) {
	t.reachedMu.Lock()
	t.reachedUpFilter = f
	t.reachedMu.Unlock()
}

// AddReachedDownstreamFilter widens the downstream filter to select what f
// selects as well, so two consumers of the same task can each ask for their own
// frames without knowing about each other.
func (t *Task) AddReachedDownstreamFilter(f FrameFilter) {
	t.reachedMu.Lock()
	t.reachedDownFilter = Or(t.reachedDownFilter, f)
	t.reachedMu.Unlock()
}

// AddReachedUpstreamFilter widens the upstream filter to select what f selects
// as well. See AddReachedDownstreamFilter.
func (t *Task) AddReachedUpstreamFilter(f FrameFilter) {
	t.reachedMu.Lock()
	t.reachedUpFilter = Or(t.reachedUpFilter, f)
	t.reachedMu.Unlock()
}

// notifyReachedDownstream reports a frame reaching the end of the pipeline to
// the handlers, if the filter selects it. Handlers run in the order they were
// registered, on the goroutine the frame arrived on.
func (t *Task) notifyReachedDownstream(f frames.Frame) {
	t.reachedMu.Lock()
	filter := t.reachedDownFilter
	handlers := t.reachedDownstream
	t.reachedMu.Unlock()

	if !filter.selects(f) {
		return
	}
	if t.params.OnReachedDownstream != nil {
		t.params.OnReachedDownstream(f)
	}
	for _, h := range handlers {
		h(f)
	}
}

// notifyReachedUpstream reports a frame reaching the start of the pipeline, if
// the filter selects it.
func (t *Task) notifyReachedUpstream(f frames.Frame) {
	t.reachedMu.Lock()
	filter := t.reachedUpFilter
	t.reachedMu.Unlock()

	if !filter.selects(f) {
		return
	}
	if t.params.OnReachedUpstream != nil {
		t.params.OnReachedUpstream(f)
	}
}

// StopWhenDone schedules the pipeline to stop once all queued frames have been
// processed, by queueing an EndFrame.
func (t *Task) StopWhenDone() { t.QueueFrame(frames.NewEndFrame()) }

// Flush waits for the pipeline to drain: it sends a PipelineFlushFrame probe and
// blocks until the probe has traveled down to the sink and back up to the
// source, meaning everything already in the pipeline ahead of it has been
// processed. Use it to let the pipeline settle, after an interruption say,
// before injecting new work. It returns ctx.Err() if ctx is done first.
//
// The probe is injected straight into the pipeline rather than queued with
// QueueFrame. The queue stops being drained the moment a pipeline-ending frame
// goes into it, since the task then waits for that frame to travel through, so a
// probe queued behind one would never enter the pipeline at all and the caller
// would wait out its whole timeout. A tool handler that ends the session leaves
// exactly that behind.
//
// What it covers follows from that: the frames in the pipeline, not the ones
// still waiting in the task's own queue. A frame handed to QueueFrame just
// before this call may still be in that queue, and the probe will not wait for
// it. That is what a caller wants of it, since what has to settle is the work
// the pipeline is in the middle of.
func (t *Task) Flush(ctx context.Context) error {
	// Nothing goes into the pipeline before it is running. Injecting straight
	// into it, rather than through the queue the task drains once it is up,
	// means arriving during setup is possible, and a processor being set up on
	// the task's goroutine must not be read from this one. Waiting here also
	// means the probe is never handed to a processor that has yet to see its
	// StartFrame, which would drop it and leave the caller waiting out the whole
	// timeout for nothing.
	select {
	case <-t.startSig:
	case <-ctx.Done():
		return ctx.Err()
	}

	probe := frames.NewPipelineFlushFrame()
	if err := t.pipeline.QueueFrame(ctx, probe, processor.Downstream); err != nil {
		return err
	}
	select {
	case <-probe.Done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Cancel stops the pipeline immediately by queueing a CancelFrame. It does
// nothing once the task has finished.
func (t *Task) Cancel() { t.cancelWithReason("") }

// cancelWithReason queues a CancelFrame carrying reason, at most once.
func (t *Task) cancelWithReason(reason string) {
	t.mu.Lock()
	if t.canceling || t.finished {
		t.mu.Unlock()
		return
	}
	t.canceling = true
	t.mu.Unlock()

	// Release the run loop if it is still waiting for the StartFrame to reach
	// the sink. Canceling before the pipeline is up is exactly the case where
	// something is wedged during startup, and the run loop only starts draining
	// the queue this frame goes into once it has stopped waiting. Without this
	// the CancelFrame would sit in the queue and the run would never end.
	t.startOnce.Do(func() { close(t.startSig) })

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

	endFrame, runErr := t.runLoop(runCtx)

	// A StopFrame ends the run but leaves the processors up, connections and
	// all, ready for another one. Cleaning up here would shut down the very
	// thing it asked to keep. Every other way of ending tears the pipeline down.
	//
	// Clean up with a fresh context so a canceled runCtx does not abort the
	// goroutine shutdown.
	if _, stopped := endFrame.(*frames.StopFrame); !stopped {
		_ = t.pipeline.Cleanup(context.Background())
	}

	// The conversation span closes last, once the processors have stopped and
	// the spans raised beneath it have ended.
	t.turnTrace.EndConversation()

	t.mu.Lock()
	t.finished = true
	t.mu.Unlock()
	return runErr
}

// runLoop sends the StartFrame, waits for the pipeline to be ready, then pushes
// queued frames until a pipeline-ending frame has traveled through. It returns
// that frame, so the caller can tell a StopFrame from the ways of ending that
// shut the processors down, and nil when the context ended the run instead.
func (t *Task) runLoop(ctx context.Context) (frames.Frame, error) {
	t.clk.Start()

	start := frames.NewStartFrame()
	start.AudioInSampleRate = t.params.AudioInSampleRate
	start.AudioOutSampleRate = t.params.AudioOutSampleRate
	start.EnableMetrics = t.params.EnableMetrics
	start.EnableUsageMetrics = t.params.EnableUsageMetrics
	start.ReportOnlyInitialTTFB = t.params.ReportOnlyInitialTTFB
	if err := t.pipeline.QueueFrame(ctx, start, processor.Downstream); err != nil {
		return nil, err
	}

	select {
	case <-t.startSig:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	if t.params.EnableMetrics && boolValue(t.params.SendInitialEmptyMetrics, true) {
		if err := t.pipeline.QueueFrame(ctx, t.initialMetricsFrame(), processor.Downstream); err != nil {
			return nil, err
		}
	}

	for {
		f, ok := t.pushQueue.get(ctx)
		if !ok {
			return nil, ctx.Err()
		}
		if err := t.pipeline.QueueFrame(ctx, f, processor.Downstream); err != nil {
			return nil, err
		}
		if isPipelineEnd(f) {
			return f, t.waitPipelineEnd(ctx, f)
		}
	}
}

// waitPipelineEnd blocks until f has traveled all the way through the pipeline.
//
// The wait for a CancelFrame is bounded by CancelTimeout, and the finished
// handler runs whether or not the frame arrived: canceling is the path taken
// when something is already wrong, so it has to complete even when a processor
// is wedged and the frame never comes back. An EndFrame is waited out in full,
// since a graceful shutdown is meant to flush what is in flight.
func (t *Task) waitPipelineEnd(ctx context.Context, f frames.Frame) error {
	if _, isCancel := f.(*frames.CancelFrame); !isCancel {
		select {
		case <-t.endSig:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	// The sink does not report a CancelFrame as finished, so that this runs
	// exactly once either way: here, on arrival or on timeout.
	defer func() {
		if t.params.OnPipelineFinished != nil {
			t.params.OnPipelineFinished(f)
		}
	}()

	timeout := time.NewTimer(t.params.CancelTimeout)
	defer timeout.Stop()
	select {
	case <-t.endSig:
	case <-timeout.C:
		slog.Warn("timed out waiting for the cancel frame to reach the end of the pipeline",
			"timeout", t.params.CancelTimeout)
	case <-ctx.Done():
		return ctx.Err()
	}
	return nil
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
	// Reported first, so a caller watching the end of the pipeline sees every
	// frame that gets here, including the ones handled and consumed below.
	t.notifyReachedDownstream(f)

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

	switch fr := f.(type) {
	case *frames.StartFrame:
		if t.params.OnPipelineStarted != nil {
			t.params.OnPipelineStarted(fr)
		}
		t.startOnce.Do(func() { close(t.startSig) })
	case *frames.EndFrame, *frames.StopFrame:
		if t.params.OnPipelineFinished != nil {
			t.params.OnPipelineFinished(f)
		}
		t.endOnce.Do(func() { close(t.endSig) })
	case *frames.CancelFrame:
		// The finished handler is not called here. A CancelFrame may never
		// arrive, so the run loop reports it instead, on arrival or on timeout.
		t.endOnce.Do(func() { close(t.endSig) })
	}
	return nil
}

// sourcePush observes frames reaching the start of the pipeline and converts the
// worker frames a processor pushed upstream into the corresponding pipeline-wide
// frame. A fatal error cancels the pipeline.
func (t *Task) sourcePush(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	// Reported first, so a caller watching the start of the pipeline sees every
	// frame that gets here, including the ones handled and consumed below.
	t.notifyReachedUpstream(f)

	// The flush probe completed its round trip: everything queued ahead of it has
	// been processed, so release whoever is waiting on it.
	if probe, ok := f.(*frames.PipelineFlushFrame); ok {
		probe.CloseDone()
		return nil
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
	case frames.ErrorReport:
		// Matched by interface, so a frame reporting an unrecoverable failure by
		// type is caught alongside a plain error frame carrying the flag.
		ef := fr.ErrorInfo()
		if t.params.OnPipelineError != nil {
			t.params.OnPipelineError(ef)
		}
		if ef.Fatal {
			slog.Error("fatal error reached the start of the pipeline, canceling", "err", ef.Error)
			t.Cancel()
		} else {
			slog.Warn("error reached the start of the pipeline", "err", ef.Error)
		}
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
