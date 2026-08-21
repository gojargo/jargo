package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"sync"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/telemetry/tracing"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
	"go.opentelemetry.io/otel/attribute"
)

// Default sample rates used for the StartFrame when not overridden.
const (
	defaultAudioInSampleRate  = 16000
	defaultAudioOutSampleRate = 24000
)

// Timing defaults for the task's own lifecycle and monitoring.
const (
	// defaultSetupTimeout bounds how long the worker waits for every processor
	// to finish being set up.
	defaultSetupTimeout = 20 * time.Second
	// defaultStartTimeout bounds how long the worker waits for the StartFrame to
	// reach the end of the pipeline.
	defaultStartTimeout = 20 * time.Second
	// defaultCancelTimeout bounds how long the task waits for a CancelFrame to
	// reach the end of the pipeline before giving up on it.
	defaultCancelTimeout = 20 * time.Second
	// defaultHeartbeatPeriod is how often a heartbeat is sent when heartbeats
	// are enabled.
	defaultHeartbeatPeriod = time.Second
	// defaultHeartbeatMonitorTimeout is how long the monitor goes without a
	// heartbeat before reporting the silence.
	defaultHeartbeatMonitorTimeout = 10 * time.Second
	// defaultIdleTimeout is how long the pipeline may go without the frames that
	// count as activity before it is considered idle.
	defaultIdleTimeout = 5 * time.Minute
)

// Params is what the pipeline itself is told when it starts: the values that
// ride on the StartFrame and reach every processor at setup. Anything about the
// worker rather than the pipeline belongs in WorkerConfig.
type Params struct {
	// AudioInSampleRate is the StartFrame input sample rate; default 16000.
	AudioInSampleRate int
	// AudioOutSampleRate is the StartFrame output sample rate; default 24000.
	AudioOutSampleRate int
	// EnableMetrics enables performance-metrics collection across the pipeline.
	EnableMetrics bool
	// EnableUsageMetrics enables usage-metrics collection (e.g. LLM token usage)
	// across the pipeline.
	EnableUsageMetrics bool
	// StartMetadata is carried on the StartFrame, where every processor sees it
	// at setup. It is for the values a whole session is stamped with, a call id
	// or a tenant say, that a processor or an observer needs and the framework
	// knows nothing about.
	StartMetadata map[string]any
	// ReportOnlyInitialTTFB reports each service's first time-to-first-byte and
	// no more, for a consumer who wants the figure the call opened with rather
	// than one per turn. It only applies when EnableMetrics is set.
	ReportOnlyInitialTTFB bool
	// SendInitialEmptyMetrics sends a zeroed MetricsFrame for every processor
	// that reports metrics once the pipeline is ready, so a consumer knows which
	// processors to expect metrics from before any have been measured; nil
	// defaults to true. It only applies when EnableMetrics is set.
	SendInitialEmptyMetrics *bool
	// EnableHeartbeats sends a heartbeat through the pipeline at a fixed
	// interval and reports when one stops coming out the far end, which is how a
	// caller tells a pipeline that is quiet from one that is stuck.
	EnableHeartbeats bool
	// HeartbeatPeriod is how often a heartbeat is sent; zero defaults to one
	// second. It only applies when EnableHeartbeats is set.
	HeartbeatPeriod time.Duration
	// HeartbeatMonitorTimeout is how long to go without a heartbeat reaching the
	// end of the pipeline before saying so; zero defaults to ten seconds. It only
	// applies when EnableHeartbeats is set.
	HeartbeatMonitorTimeout time.Duration
}

// WorkerConfig configures a Worker: how it drives its pipeline, and how it
// takes part in a session alongside the other workers.
type WorkerConfig struct {
	// Name is what other workers address this one by. Empty names it after its
	// type, which suits a worker taking no part in worker-to-worker messaging.
	Name string
	// Active reports whether the worker starts active; nil defaults to true.
	Active *bool
	// Params is what the pipeline is told when it starts.
	Params Params
	// Clock is the pipeline clock; a system clock is used when nil.
	Clock clock.Clock
	// AppResources is whatever the application wants to share across a session,
	// a database handle or an HTTP client say. It is passed by reference and
	// never read by the framework; a processor reaches it through its setup.
	AppResources any
	// CancelTimeout bounds the wait for a CancelFrame to travel the pipeline;
	// zero defaults to 20 seconds. Canceling is what a caller reaches for when
	// something has already gone wrong, so it must not be the thing that hangs:
	// a processor wedged somewhere would otherwise hold the shutdown open for
	// good. Reaching the timeout gives up on the frame and finishes the run. An
	// EndFrame is not bounded this way, since a graceful shutdown is expected to
	// flush whatever is in flight however long that takes.
	CancelTimeout time.Duration
	// SetupTimeout bounds the wait for every processor to finish being set up;
	// zero defaults to 20 seconds. A processor connects while it is set up, so
	// one that never connects would otherwise leave the run waiting on it with
	// nothing to time it out. Reaching the timeout abandons setting up, raises
	// EventSetupTimeout and tears down whatever came up, so Run returns rather
	// than hanging.
	SetupTimeout time.Duration
	// StartTimeout bounds the wait for the StartFrame to reach the end of the
	// pipeline; zero defaults to 20 seconds. A processor that blocks while
	// handling it leaves the pipeline half-started, and nothing pushed in
	// afterwards would be processed, so reaching the timeout raises
	// EventPipelineTimeout and tears the pipeline down.
	StartTimeout time.Duration
	// IdleTimeout is how long the pipeline may go without any of the frames
	// IdleTimeoutFrames selects before it counts as idle; zero defaults to five
	// minutes and a negative value turns idle detection off.
	IdleTimeout time.Duration
	// IdleTimeoutFrames selects the frames that count as the pipeline being
	// busy; nil counts the assistant or the user speaking. The StartFrame always
	// counts, so the first stretch is measured from the pipeline coming up.
	IdleTimeoutFrames FrameFilter
	// CancelOnIdleTimeout cancels the worker once the pipeline has gone idle;
	// nil defaults to true. Setting it false reports the idle pipeline and
	// leaves it running, and goes on reporting each further stretch of quiet.
	CancelOnIdleTimeout *bool
	// CancelRunnerOnIdleTimeout also cancels the runner, and with it every other
	// root worker, when the pipeline goes idle; nil defaults to true. It only
	// applies when CancelOnIdleTimeout is set: opting out of canceling this
	// worker opts out of canceling the rest. Set it false for a worker running
	// beside others that should see itself out without taking them with it.
	CancelRunnerOnIdleTimeout *bool
	// ProcessorUnusablePolicy says what to do when a processor reports an error
	// that leaves it unable to do its job, a service whose API key was rejected
	// being the case in point. The zero value continues, leaving the decision to
	// the EventPipelineError handlers.
	ProcessorUnusablePolicy ProcessorUnusablePolicy
	// ReachedDownstreamFilter selects which frames reaching the end of the
	// pipeline are reported. Nil selects nothing, so a handler without a filter
	// is never called; that is the default because a handler on every frame sits
	// on the path of everything the pipeline does. Use AnyFrame to watch the
	// whole stream anyway.
	ReachedDownstreamFilter FrameFilter
	// ReachedUpstreamFilter selects which frames reaching the start of the
	// pipeline are reported. Nil selects nothing. See ReachedDownstreamFilter.
	ReachedUpstreamFilter FrameFilter
	// Observers watch every frame reaching either end of the pipeline.
	Observers []Observer
	// EnableTurnTracking follows the conversation's turns; nil defaults to true.
	// It applies whether or not the worker traces, since where the turns fell is
	// worth knowing either way; see Worker.TurnTracking. Tracing hangs off it, so
	// turning it off turns tracing off too: there would be nothing to hang the
	// turn spans from.
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
	// ones the worker sets itself. They are where the keys a trace backend reads
	// from the root span belong: a session id, a user id, tags.
	AdditionalSpanAttributes []attribute.KeyValue
	// Bridged puts the pipeline on the bus: what comes out of either end is
	// copied across, and what arrives from another worker is injected back in.
	//
	// Nil leaves the pipeline to itself. An empty but non-nil slice takes frames
	// from every bridge, and naming bridges takes them only from those. A
	// bridged worker is how one worker's pipeline feeds another's.
	Bridged []string
	// ExcludeFrames are frame types that never cross the bus, on top of the
	// lifecycle frames, which never do. It only applies to a bridged worker.
	ExcludeFrames []frames.Frame
	// EnableRTVI puts an RTVI processor at the head of the pipeline, and its
	// observer alongside the others, so a client is told what the session is
	// doing without the pipeline having to be built for it; nil defaults to
	// true.
	//
	// A pipeline that already contains one keeps its own, and the observer for
	// it must be given in Observers. Set it false for a pipeline with no client
	// on the other end.
	EnableRTVI *bool
	// RTVIProcessor is the processor to add when RTVI is enabled and the
	// pipeline does not already contain one; nil builds a default one.
	RTVIProcessor *rtvi.Processor
	// RTVIObserverParams configures the observer added alongside it.
	RTVIObserverParams *rtvi.ObserverParams
}

// ProcessorUnusablePolicy is what a pipeline worker does when a processor can no
// longer do its job.
//
// An unusable processor keeps failing for as long as the pipeline keeps using
// it, so the pipeline has to decide whether it is still worth running.
type ProcessorUnusablePolicy string

// The policies a worker can take on an unusable processor.
const (
	// UnusableContinue reports the error and keeps running. The application
	// decides what to do, by failing over to another provider for instance. It
	// is the zero value, and so the default.
	UnusableContinue ProcessorUnusablePolicy = ""
	// UnusableEnd ends the pipeline gracefully, letting queued frames drain
	// first.
	UnusableEnd ProcessorUnusablePolicy = "end"
	// UnusableCancel cancels the pipeline immediately, abandoning queued frames.
	UnusableCancel ProcessorUnusablePolicy = "cancel"
)

// The events a pipeline worker raises, on top of the ones every worker raises.
const (
	// EventFrameReachedUpstream fires for a frame reaching the start of the
	// pipeline that ReachedUpstreamFilter selects, carrying that frame.
	EventFrameReachedUpstream = "on_frame_reached_upstream"
	// EventFrameReachedDownstream fires for a frame reaching the end of the
	// pipeline that ReachedDownstreamFilter selects, carrying that frame.
	EventFrameReachedDownstream = "on_frame_reached_downstream"
	// EventHeartbeatTimeout fires when no heartbeat has reached the end of the
	// pipeline within the monitor timeout, and again every interval for as long
	// as the silence lasts. It carries nothing.
	EventHeartbeatTimeout = "on_heartbeat_timeout"
	// EventIdleTimeout fires when the pipeline has gone idle, before it is
	// canceled. It carries nothing.
	EventIdleTimeout = "on_idle_timeout"
	// EventPipelineStarted fires once the StartFrame has traveled the whole
	// pipeline, meaning every processor is up, carrying that frame.
	EventPipelineStarted = "on_pipeline_started"
	// EventPipelineFinished fires once the pipeline has reached a terminal
	// state, carrying the frame that ended it: an EndFrame, a StopFrame or a
	// CancelFrame. It fires for a CancelFrame even when the frame never arrived
	// and the wait timed out, so a caller always hears the run end.
	EventPipelineFinished = "on_pipeline_finished"
	// EventPipelineTimeout fires when a frame the worker was waiting on never
	// reached the end of the pipeline: the StartFrame within StartTimeout, or a
	// CancelFrame within CancelTimeout. Handlers receive the worker and that
	// frame. Nothing else tells an application its pipeline never came up.
	EventPipelineTimeout = "on_pipeline_timeout"
	// EventSetupTimeout fires when the processors did not all finish being set
	// up within SetupTimeout. Handlers receive the worker.
	EventSetupTimeout = "on_setup_timeout"
	// EventPipelineError fires for every error frame reaching the start of the
	// pipeline, fatal or not, carrying that frame. A fatal one also cancels the
	// worker, after the handlers have run.
	EventPipelineError = "on_pipeline_error"
)

// Worker runs a pipeline for a single session. It drives the lifecycle: it sends
// the StartFrame, waits for the pipeline to be ready, pushes queued frames, and
// shuts the pipeline down once an EndFrame or CancelFrame has traveled all the
// way through.
type Worker struct {
	*workers.Base

	pipeline *Pipeline
	source   processor.Processor
	sink     processor.Processor
	cfg      WorkerConfig
	params   Params
	clk      clock.Clock

	pushQueue *frameQueue

	// observers are the caller's, plus the ones the worker registers itself to
	// track and trace turns.
	observers []Observer
	// tracing is the session's tracing state, handed to the processors at setup;
	// nil when the worker is not tracing.
	tracing *tracing.TracingContext
	// turnTrace writes the conversation and turn spans; nil when not tracing.
	turnTrace *observers.TurnTrace
	// turnTracking follows the conversation's turns; nil only when turn tracking
	// has been turned off.
	turnTracking *observers.TurnTracking
	// observerProxy is the single observer the processors are given. It passes
	// each report on to the real observers, off the frame path.
	observerProxy *observerProxy
	// rtvi is the processor a client talks to, this worker's own or the one the
	// pipeline already carried; nil when the worker runs without RTVI.
	rtvi *rtvi.Processor

	// The filters select which frames reaching either end of the pipeline are
	// reported, and start out as the ones WorkerConfig carries.
	reachedMu         sync.Mutex
	reachedDownFilter FrameFilter
	reachedUpFilter   FrameFilter

	// monitors is the scope the worker's background goroutines run in, heartbeats
	// is the queue the sink hands each arriving heartbeat to for the monitor to
	// time, and idleSig carries the activity the idle observer reports.
	monitors   monitors
	heartbeats *frameQueue
	idleSig    chan struct{}

	startOnce sync.Once
	startSig  chan struct{}
	endOnce   sync.Once
	endSig    chan struct{}

	mu        sync.Mutex
	finished  bool
	canceling bool
	// cancelReason is what the cancellation was asked for, kept so the frame
	// carries it even when the run's context ends before the queue is drained.
	cancelReason string
	// cancelFrame is the CancelFrame already sent into the pipeline, so a run
	// whose context ends while that frame is still traveling waits for it out
	// rather than sending a second.
	cancelFrame *frames.CancelFrame
	// finishOnce reports the pipeline finished once. The cancel path can be
	// reached twice, by the run loop waiting for the frame and by the
	// context-end handling behind it, and a run finishes once.
	finishOnce sync.Once
	// unusable holds the ids of the processors the unusable policy has already
	// been applied to. One that can no longer do its job goes on failing for as
	// long as the pipeline uses it, so the policy answers the first error that
	// says so and not the ones behind it.
	unusable map[uint64]bool
	// runCtx is the context the run is under, kept so a frame queued from
	// outside the pipeline has one to enter with.
	runCtx context.Context
}

// NewWorker wraps pipe in a Worker. pipe is usually a *Pipeline but may be any
// processor.
func NewWorker(pipe processor.Processor, cfg WorkerConfig) *Worker {
	if cfg.Params.AudioInSampleRate == 0 {
		cfg.Params.AudioInSampleRate = defaultAudioInSampleRate
	}
	if cfg.Params.AudioOutSampleRate == 0 {
		cfg.Params.AudioOutSampleRate = defaultAudioOutSampleRate
	}
	if cfg.CancelTimeout == 0 {
		cfg.CancelTimeout = defaultCancelTimeout
	}
	if cfg.SetupTimeout == 0 {
		cfg.SetupTimeout = defaultSetupTimeout
	}
	if cfg.StartTimeout == 0 {
		cfg.StartTimeout = defaultStartTimeout
	}
	if cfg.Params.HeartbeatPeriod == 0 {
		cfg.Params.HeartbeatPeriod = defaultHeartbeatPeriod
	}
	if cfg.Params.HeartbeatMonitorTimeout == 0 {
		cfg.Params.HeartbeatMonitorTimeout = defaultHeartbeatMonitorTimeout
	}
	if cfg.IdleTimeout == 0 {
		cfg.IdleTimeout = defaultIdleTimeout
	}
	if cfg.IdleTimeoutFrames == nil {
		cfg.IdleTimeoutFrames = FrameTypes(
			&frames.BotSpeakingFrame{},
			&frames.UserSpeakingFrame{},
		)
	}
	t := &Worker{
		cfg:               cfg,
		params:            cfg.Params,
		clk:               cfg.Clock,
		pushQueue:         newFrameQueue(),
		heartbeats:        newFrameQueue(),
		idleSig:           make(chan struct{}, 1),
		startSig:          make(chan struct{}),
		endSig:            make(chan struct{}),
		reachedDownFilter: cfg.ReachedDownstreamFilter,
		reachedUpFilter:   cfg.ReachedUpstreamFilter,
	}
	t.Base = workers.New(workers.Config{Name: cfg.Name, Active: cfg.Active}, t)
	t.registerEvents()

	pipe = t.maybeAddRTVI(pipe, &cfg)
	// Taken again: adding RTVI puts an observer among the caller's, and the
	// copy above was made before that.
	t.cfg = cfg

	// A bridged pipeline is wrapped in the two edges before the worker's own
	// source and sink go around it, so what the pipeline produces is teed onto
	// the bus and what arrives from the bus enters the pipeline proper. The
	// edges read the bus when they are set up rather than when they are built,
	// so the worker need only be attached by the time it runs.
	if cfg.Bridged != nil {
		pipe = New(
			bus.NewEdgeProcessor(bus.EdgeConfig{
				Worker:        t,
				Direction:     processor.Upstream,
				Bridges:       cfg.Bridged,
				ExcludeFrames: cfg.ExcludeFrames,
			}),
			pipe,
			bus.NewEdgeProcessor(bus.EdgeConfig{
				Worker:        t,
				Direction:     processor.Downstream,
				Bridges:       cfg.Bridged,
				ExcludeFrames: cfg.ExcludeFrames,
			}),
		)
	}
	if t.clk == nil {
		t.clk = clock.NewSystem()
	}
	t.buildObservers()
	t.observerProxy = newObserverProxy(t.observers)
	// The source observes upstream frames, the sink observes downstream frames.
	// They bracket the user pipeline so the worker can inject and observe frames.
	t.source = processor.NewSource("Worker::Source", t.sourcePush)
	t.sink = processor.NewSink("Worker::Sink", t.sinkPush)
	t.pipeline = build(t.source, t.sink, []processor.Processor{pipe})
	return t
}

// buildObservers assembles the observers the pipeline runs with: the caller's,
// and, when the worker traces, the turn tracing plus the turn tracking and
// latency measurement that feed it.
func (t *Worker) buildObservers() {
	t.observers = append([]Observer(nil), t.cfg.Observers...)
	if t.cfg.IdleTimeout > 0 {
		t.observers = append(t.observers, &idleObserver{
			match: t.cfg.IdleTimeoutFrames,
			sig:   t.idleSig,
		})
	}

	// Turn tracking runs whether or not the task traces: a caller wants to know
	// where the turns fell in an untraced session too. Tracing is what hangs off
	// it, not the other way round, so turning turn tracking off turns tracing off
	// with it: there would be nothing to hang the turn spans from.
	if !boolOrTrue(t.cfg.EnableTurnTracking) {
		return
	}

	if t.cfg.EnableTracing {
		t.tracing = tracing.NewTracingContext()
		t.turnTrace = observers.NewTurnTrace(observers.TurnTraceConfig{
			Tracing:        t.tracing,
			ConversationID: t.cfg.ConversationID,
			Attributes:     t.cfg.AdditionalSpanAttributes,
		})
	}

	// The turn tracking feeds the tracing when there is any, and stands on its
	// own when there is not.
	cfg := observers.TurnTrackingConfig{}
	if t.turnTrace != nil {
		cfg.OnTurnStarted = t.turnTrace.TurnStarted
		cfg.OnTurnEnded = t.turnTrace.TurnEnded
	}
	t.turnTracking = observers.NewTurnTracking(cfg)
	t.observers = append(t.observers, t.turnTracking)

	if t.turnTrace != nil {
		t.observers = append(t.observers,
			observers.NewUserBotLatency(observers.LatencyConfig{
				OnLatency: t.turnTrace.LatencyMeasured,
			}),
			t.turnTrace,
		)
	}
}

// AddObserver registers an observer while the pipeline runs, for something built
// after the task that wants to watch the frames going by. It watches from here
// on; what already happened is not replayed.
func (t *Worker) AddObserver(o Observer) {
	if o == nil {
		return
	}
	t.observerProxy.add(o)
}

// RemoveObserver stops reporting to an observer registered earlier. It has
// stopped by the time this returns, so whatever the observer holds may be
// released. Removing one that was never registered does nothing.
func (t *Worker) RemoveObserver(o Observer) {
	if o == nil {
		return
	}
	t.observerProxy.remove(o)
}

// TurnTracking is the observer following the conversation's turns. It is nil
// only when turn tracking has been turned off.
func (t *Worker) TurnTracking() *observers.TurnTracking { return t.turnTracking }

// TurnTrace is the observer writing the conversation and turn spans. It is nil
// unless the task traces.
func (t *Worker) TurnTrace() *observers.TurnTrace { return t.turnTrace }

// Tracing is the session's tracing state: the conversation span the trace hangs
// from and the turn being spoken. It is nil unless the task traces, and is what
// the processors are given at setup so their spans land under the right turn.
func (t *Worker) Tracing() *tracing.TracingContext { return t.tracing }

// QueueFrame queues a frame to be pushed through the pipeline. The direction is
// optional and defaults to downstream, which enters at the start of the
// pipeline; pass processor.Upstream to enter at the end instead, which is how a
// caller replies to something the pipeline sent it. Only the first direction is
// read.
func (t *Worker) QueueFrame(f frames.Frame, dir ...processor.Direction) {
	if len(dir) > 0 && dir[0] == processor.Upstream {
		// Straight into the sink, the far end of the pipeline, rather than the
		// queue the run loop drains from the near end.
		_ = t.sink.QueueFrame(t.runContext(), f, processor.Upstream)
		return
	}
	t.pushQueue.push(f)
}

// QueueFrames queues several frames, in order. See QueueFrame for the direction.
func (t *Worker) QueueFrames(fs []frames.Frame, dir ...processor.Direction) {
	for _, f := range fs {
		t.QueueFrame(f, dir...)
	}
}

// runContext is the context the run is under, or a background context before it
// has started. A frame entering the pipeline needs one, and the caller of
// QueueFrame has none to give.
func (t *Worker) runContext() context.Context {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.runCtx != nil {
		return t.runCtx
	}
	return context.Background()
}

// SetReachedDownstreamFilter replaces the filter selecting which frames reaching
// the end of the pipeline are reported.
func (t *Worker) SetReachedDownstreamFilter(f FrameFilter) {
	t.reachedMu.Lock()
	t.reachedDownFilter = f
	t.reachedMu.Unlock()
}

// SetReachedUpstreamFilter replaces the filter selecting which frames reaching
// the start of the pipeline are reported.
func (t *Worker) SetReachedUpstreamFilter(f FrameFilter) {
	t.reachedMu.Lock()
	t.reachedUpFilter = f
	t.reachedMu.Unlock()
}

// AddReachedDownstreamFilter widens the downstream filter to select what f
// selects as well, so two consumers of the same task can each ask for their own
// frames without knowing about each other.
func (t *Worker) AddReachedDownstreamFilter(f FrameFilter) {
	t.reachedMu.Lock()
	t.reachedDownFilter = Or(t.reachedDownFilter, f)
	t.reachedMu.Unlock()
}

// AddReachedUpstreamFilter widens the upstream filter to select what f selects
// as well. See AddReachedDownstreamFilter.
func (t *Worker) AddReachedUpstreamFilter(f FrameFilter) {
	t.reachedMu.Lock()
	t.reachedUpFilter = Or(t.reachedUpFilter, f)
	t.reachedMu.Unlock()
}

// notifyReachedDownstream reports a frame reaching the end of the pipeline, if
// the filter selects it.
func (t *Worker) notifyReachedDownstream(ctx context.Context, f frames.Frame) {
	t.reachedMu.Lock()
	filter := t.reachedDownFilter
	t.reachedMu.Unlock()

	if filter.selects(f) {
		t.Call(ctx, EventFrameReachedDownstream, t, f)
	}
}

// notifyReachedUpstream reports a frame reaching the start of the pipeline, if
// the filter selects it.
func (t *Worker) notifyReachedUpstream(ctx context.Context, f frames.Frame) {
	t.reachedMu.Lock()
	filter := t.reachedUpFilter
	t.reachedMu.Unlock()

	if filter.selects(f) {
		t.Call(ctx, EventFrameReachedUpstream, t, f)
	}
}

// registerEvents declares the events a pipeline worker raises on top of the
// ones every worker raises, and ties the worker's own lifecycle to the
// pipeline's: it becomes ready once the pipeline is up, and has finished once
// the pipeline has, so whoever is waiting on the worker is told at the moment
// that is true of it.
func (t *Worker) registerEvents() {
	t.Register(EventFrameReachedUpstream, false)
	t.Register(EventFrameReachedDownstream, false)
	t.Register(EventHeartbeatTimeout, false)
	t.Register(EventIdleTimeout, false)
	t.Register(EventPipelineStarted, false)
	t.Register(EventPipelineFinished, false)
	t.Register(EventPipelineError, false)
	t.Register(EventPipelineTimeout, false)
	t.Register(EventSetupTimeout, false)

	events.On(&t.Registry, EventPipelineStarted, func(ctx context.Context, _ *frames.StartFrame) {
		t.Start(ctx)
	})
	events.On(&t.Registry, EventPipelineFinished, func(ctx context.Context, _ frames.Frame) {
		t.Stop(ctx)
	})
}

// OnBusMessage handles the messages a pipeline worker acts on itself, after the
// handling every worker does.
//
// A request to speak becomes a frame in this worker's pipeline. A pipeline with
// no speech in it simply lets the frame through.
func (t *Worker) OnBusMessage(ctx context.Context, m bus.Message) {
	t.Base.OnBusMessage(ctx, m)

	// The base drops a message addressed elsewhere, but it returns before
	// reaching here, so the same check has to be made again before anything is
	// queued into this worker's pipeline.
	if m.Target() != "" && m.Target() != t.Name() {
		return
	}

	if speak, ok := m.(*bus.TTSSpeakMessage); ok {
		f := frames.NewTTSSpeakFrame(speak.Text)
		f.AppendToContext = speak.AppendToContext
		t.QueueFrame(f)
	}
}

// HandleWorkerEnd ends the pipeline once the children have ended.
//
// The end is driven through the pipeline as an EndFrame rather than stopping
// the worker where it stands, so every processor is told in turn and the worker
// is finished only once the frame has drained out the far end.
func (t *Worker) HandleWorkerEnd(ctx context.Context, m *bus.EndWorkerMessage) {
	slog.Debug("pipeline worker received end", "worker", t.Name())
	t.PropagateEndToChildren(ctx, m.Reason)
	end := frames.NewEndFrame()
	end.Reason = m.Reason
	t.QueueFrame(end)
}

// HandleWorkerCancel cancels the pipeline once the children have been told. See
// HandleWorkerEnd.
func (t *Worker) HandleWorkerCancel(ctx context.Context, m *bus.CancelWorkerMessage) {
	slog.Debug("pipeline worker received cancel", "worker", t.Name())
	t.PropagateCancelToChildren(ctx, m.Reason)
	t.Cancel(ctx, m.Reason)
}

// StopWhenDone schedules the pipeline to stop once all queued frames have been
// processed, by queueing an EndFrame.
func (t *Worker) StopWhenDone() { t.QueueFrame(frames.NewEndFrame()) }

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
func (t *Worker) Flush(ctx context.Context) error {
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

// Cancel stops this worker's pipeline immediately by queueing a CancelFrame. It
// does nothing once the worker has finished.
//
// It replaces the plain worker's cancel, which asks every worker in the session
// to stop: a pipeline worker stops by driving a frame through its own pipeline,
// so that each processor is told in turn. Call Base.Cancel for the session-wide
// one.
func (t *Worker) Cancel(ctx context.Context, reason string) {
	if t.HasFinished() {
		return
	}
	t.cancelWithReason(reason)
}

// unusableSource returns the processor an error names when that processor can no
// longer do its job, and nil otherwise. It is how the two kinds of error are
// told apart: the verdict travels with the error rather than in a flag on it.
func unusableSource(ef *frames.ErrorFrame) processor.Processor {
	p, ok := ef.Source.(processor.Processor)
	if !ok || p.Usable() {
		return nil
	}
	return p
}

// handleUnusableProcessor applies the unusable-processor policy, once per
// processor.
//
// A processor that can no longer do its job keeps failing for as long as the
// pipeline keeps using it, so the policy is applied to the first error that says
// so and not to the ones that follow.
func (t *Worker) handleUnusableProcessor(ctx context.Context, p processor.Processor) {
	t.mu.Lock()
	if t.unusable == nil {
		t.unusable = make(map[uint64]bool)
	}
	if t.unusable[p.ID()] {
		t.mu.Unlock()
		return
	}
	t.unusable[p.ID()] = true
	t.mu.Unlock()

	slog.ErrorContext(ctx, "processor can no longer do its job",
		"worker", t.Name(), "processor", p.Name())

	switch t.cfg.ProcessorUnusablePolicy {
	case UnusableEnd:
		t.StopWhenDone()
	case UnusableCancel:
		t.cancelWithReason(fmt.Sprintf("%s can no longer do its job", p.Name()))
	case UnusableContinue:
	}
}

// callPipelineFinished raises EventPipelineFinished for the frame that ended
// the run, at most once.
//
// A run ends once, but the cancel path can be reached twice: the run loop waits
// for the frame, and the handling of a run whose context ended waits behind it.
// Upstream is spared this because the queue there is drained by a task of its
// own that outlives the wait; here the run loop is the drain, so the two meet.
func (t *Worker) callPipelineFinished(ctx context.Context, f frames.Frame) {
	t.finishOnce.Do(func() { t.Call(ctx, EventPipelineFinished, t, f) })
}

// cancelWithReason queues a CancelFrame carrying reason, at most once.
func (t *Worker) cancelWithReason(reason string) {
	t.mu.Lock()
	if t.canceling || t.finished {
		t.mu.Unlock()
		return
	}
	t.canceling = true
	t.cancelReason = reason
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

// isCanceling reports whether a CancelFrame has already been queued.
func (t *Worker) isCanceling() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.canceling
}

// cancelOnContextEnd shuts the pipeline down after the caller's context ended
// the run, and returns the frame that did it.
//
// The run loop has stopped by now, so nothing is draining the task's queue and
// the frame goes straight into the pipeline. Everything else about the shutdown
// is what any other cancellation does: the frame travels the whole pipeline, so
// each processor stops in its turn and gets to close what it had open, and the
// wait for it is bounded so a wedged one cannot hold the run open.
func (t *Worker) cancelOnContextEnd(ctx context.Context) frames.Frame {
	t.mu.Lock()
	sent := t.cancelFrame
	reason := t.cancelReason
	t.canceling = true
	t.mu.Unlock()

	if sent != nil {
		// The run loop already sent one and was still waiting for it when the
		// context ended. Waiting it out here is what the run loop would have
		// done, and a second frame would tell the pipeline something it has
		// already been told.
		_ = t.waitPipelineEnd(ctx, sent)
		return sent
	}

	slog.Debug("the run's context ended, canceling the pipeline")

	cancelFrame := frames.NewCancelFrame()
	cancelFrame.Reason = reason
	if cancelFrame.Reason == "" {
		cancelFrame.Reason = "context canceled"
	}
	if err := t.pipeline.QueueFrame(ctx, cancelFrame, processor.Downstream); err != nil {
		// The pipeline will not carry the frame, so nothing else is going to
		// report the run over. Say so here rather than leave the caller waiting
		// on an event that cannot come.
		t.callPipelineFinished(ctx, cancelFrame)
		return nil
	}
	_ = t.waitPipelineEnd(ctx, cancelFrame)
	return cancelFrame
}

// setupWithinTimeout sets the pipeline up, bounded by SetupTimeout.
//
// A processor connects while it is set up, so one that never connects would
// otherwise leave the run waiting on it with nothing to time it out. Reaching
// the timeout gives up on the wait and leaves the blocked processor where it is,
// to be released by the cleanup that follows: the context the processors were
// handed is the one their own goroutines run on, so bounding it here would tear
// down the very processors that did come up. It reports whether everything was
// set up.
func (t *Worker) setupWithinTimeout(ctx context.Context, setup processor.Setup) (bool, error) {
	done := make(chan error, 1)
	go func() { done <- t.pipeline.Setup(ctx, setup) }()

	timeout := time.NewTimer(t.cfg.SetupTimeout)
	defer timeout.Stop()
	select {
	case err := <-done:
		return err == nil, err
	case <-timeout.C:
		slog.Error("timed out setting the pipeline up, stopping it "+
			"(a processor blocked while connecting?)", "timeout", t.cfg.SetupTimeout)
		t.Call(ctx, EventSetupTimeout, t)
		return false, nil
	}
}

// HasFinished reports whether the task has finished running.
func (t *Worker) HasFinished() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.finished
}

// Run sets up the pipeline and drives it until an EndFrame or CancelFrame
// completes its journey through the pipeline, or ctx is canceled. It then
// cleans up the pipeline. Run blocks until the pipeline has finished.
func (t *Worker) Run(ctx context.Context) error {
	t.mu.Lock()
	if t.finished {
		t.mu.Unlock()
		return nil
	}
	t.mu.Unlock()

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// The pipeline is set up detached from the caller's context, so canceling
	// that context does not kill the processors where they stand. Shutting a
	// pipeline down means sending a CancelFrame through it, and a pipeline whose
	// processors have already stopped cannot carry one: the services would never
	// hear that the call ended, and would be left to their own timeouts to close
	// what they had open. The processors are stopped by the cleanup below
	// instead, once the frame has been through them.
	pipeCtx := context.WithoutCancel(ctx)

	t.mu.Lock()
	t.runCtx = pipeCtx
	t.mu.Unlock()

	// The conversation span opens before anything can be traced under it. The
	// turn-trace observer opens it on the StartFrame too, which is the only
	// chance an observer wired up by hand gets, but that report reaches it off
	// the frame path: a service raising its span for the first frame of the
	// conversation can win the race and leave its span rooted in a trace of its
	// own. Opening it here settles the order, and the observer then finds it
	// already open.
	t.turnTrace.StartConversation(t.cfg.ConversationID)

	// The processors are handed one observer, the proxy, which passes each
	// report on to the real ones off the frame path.
	t.observerProxy.start()

	// Processors connect while they are set up and push frames as they do, so
	// the clock runs from here rather than from the StartFrame, which would
	// leave those frames timestamped zero.
	t.clk.Start()
	t.observerProxy.setupStarted(time.Now())

	setup := processor.Setup{
		Clock:          t.clk,
		Observers:      []processor.Observer{t.observerProxy},
		Tracing:        t.tracing,
		TracingEnabled: t.cfg.EnableTracing,
		Running:        t,
	}
	setUp, err := t.setupWithinTimeout(pipeCtx, setup)
	if err != nil {
		return err
	}
	if !setUp {
		// Nothing was pushed into the pipeline, so there is nothing to drain:
		// release whatever came up and give up. The teardown is the same as the
		// one at the end of a run, minus the monitors and the frame path, which
		// never started.
		_ = t.pipeline.Cleanup(context.Background())
		t.Registry.Cleanup(context.Background())
		t.observerProxy.stop()
		t.turnTrace.EndConversation()
		t.mu.Lock()
		t.finished = true
		t.mu.Unlock()
		return nil
	}

	monCtx := t.monitors.start(runCtx)
	endFrame, runErr := t.runLoop(monCtx)

	// The caller's context ended the run rather than a frame doing it, so the
	// pipeline has not been told. Tell it now, on the still-living processors.
	if runErr != nil && ctx.Err() != nil {
		endFrame = t.cancelOnContextEnd(pipeCtx)
	}

	// Stop the monitors before the pipeline is torn down, so none of them is
	// still pushing frames into processors that are shutting down.
	t.monitors.stop()

	// A StopFrame ends the run but leaves the processors up, connections and
	// all, ready for another one. Cleaning up here would shut down the very
	// thing it asked to keep. Every other way of ending tears the pipeline down.
	//
	// Clean up with a fresh context so a canceled runCtx does not abort the
	// goroutine shutdown.
	if _, stopped := endFrame.(*frames.StopFrame); !stopped {
		_ = t.pipeline.Cleanup(context.Background())
	}

	// The event handlers are waited for before the observers stop, so the
	// handlers for what the pipeline raised on its way down have run by the
	// time the run returns. A caller reading what a handler collected finds it
	// there rather than racing it.
	t.Registry.Cleanup(context.Background())

	// The observers stop once the pipeline has, so the reports raised as it shut
	// down are delivered rather than lost.
	t.observerProxy.stop()

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
func (t *Worker) runLoop(ctx context.Context) (frames.Frame, error) {
	// Watching starts before the StartFrame goes out, so a pipeline that never
	// comes to life is noticed too.
	if t.cfg.IdleTimeout > 0 {
		t.monitors.spawn(t.idleMonitorLoop)
	}

	start := frames.NewStartFrame()
	start.AudioInSampleRate = t.params.AudioInSampleRate
	start.AudioOutSampleRate = t.params.AudioOutSampleRate
	start.EnableMetrics = t.params.EnableMetrics
	start.EnableUsageMetrics = t.params.EnableUsageMetrics
	start.ReportOnlyInitialTTFB = t.params.ReportOnlyInitialTTFB
	if len(t.params.StartMetadata) > 0 {
		maps.Copy(start.Metadata(), t.params.StartMetadata)
	}
	if err := t.pipeline.QueueFrame(ctx, start, processor.Downstream); err != nil {
		return nil, err
	}

	startTimeout := time.NewTimer(t.cfg.StartTimeout)
	defer startTimeout.Stop()
	select {
	case <-t.startSig:
	case <-startTimeout.C:
		// A pipeline that never started cannot process anything pushed into it,
		// so the run gives up here and goes straight to tearing it down.
		slog.Error("timed out waiting for the start frame to reach the end of the pipeline, "+
			"stopping it (being blocked somewhere?)", "timeout", t.cfg.StartTimeout)
		t.Call(ctx, EventPipelineTimeout, t, start)
		return start, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	if t.params.EnableMetrics && boolOrTrue(t.params.SendInitialEmptyMetrics) {
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
func (t *Worker) waitPipelineEnd(ctx context.Context, f frames.Frame) error {
	if _, isCancel := f.(*frames.CancelFrame); !isCancel {
		select {
		case <-t.endSig:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	// The caller has just put this frame into the pipeline, so a run whose
	// context ends while it is still traveling knows not to send another.
	if cf, ok := f.(*frames.CancelFrame); ok {
		t.mu.Lock()
		t.cancelFrame = cf
		t.mu.Unlock()
	}

	// The sink does not report a CancelFrame as finished, so that this runs
	// exactly once either way: here, on arrival or on timeout.
	defer t.callPipelineFinished(ctx, f)

	timeout := time.NewTimer(t.cfg.CancelTimeout)
	defer timeout.Stop()
	select {
	case <-t.endSig:
	case <-timeout.C:
		slog.Warn("timed out waiting for the cancel frame to reach the end of the pipeline",
			"timeout", t.cfg.CancelTimeout)
		t.Call(ctx, EventPipelineTimeout, t, f)
	case <-ctx.Done():
		return ctx.Err()
	}
	return nil
}

// initialMetricsFrame builds one MetricsFrame carrying a zeroed time to first
// byte and processing time for every processor in the pipeline that reports
// metrics, so a consumer knows the full set before any have been measured.
func (t *Worker) initialMetricsFrame() *frames.MetricsFrame {
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

// boolOrTrue returns *p, or true when p is nil, which is the default every
// optional flag in the configuration takes.
func boolOrTrue(p *bool) bool {
	if p == nil {
		return true
	}
	return *p
}

// sinkPush observes frames reaching the end of the pipeline and signals the
// lifecycle events the run loop waits on.
func (t *Worker) sinkPush(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	// Reported first, so a caller watching the end of the pipeline sees every
	// frame that gets here, including the ones handled and consumed below.
	t.notifyReachedDownstream(ctx, f)

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
		t.Call(ctx, EventPipelineStarted, t, fr)
		t.observerProxy.pipelineStarted()
		t.startHeartbeats()
		t.startOnce.Do(func() { close(t.startSig) })
	case *frames.HeartbeatFrame:
		// Hand it to the monitor, which times how long it took to get here.
		t.heartbeats.push(fr)
	case *frames.EndFrame, *frames.StopFrame:
		t.callPipelineFinished(ctx, f)
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
func (t *Worker) sourcePush(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	// Reported first, so a caller watching the start of the pipeline sees every
	// frame that gets here, including the ones handled and consumed below.
	t.notifyReachedUpstream(ctx, f)

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
		t.Call(ctx, EventPipelineError, t, ef)
		switch {
		case ef.Fatal:
			slog.Error("fatal error reached the start of the pipeline, canceling", "err", ef.Error)
			t.cancelWithReason("")
		case unusableSource(ef) != nil:
			t.handleUnusableProcessor(ctx, unusableSource(ef))
		default:
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

// A pipeline worker is a worker, so a runner can drive it and it can be asked
// to finish what it is doing.
var (
	_ workers.Worker            = (*Worker)(nil)
	_ workers.StoppableWhenDone = (*Worker)(nil)
)

// Bridged reports whether this worker's pipeline is on the bus, which is
// announced when the worker becomes ready so the others know it can be fed.
func (t *Worker) Bridged() bool { return t.cfg.Bridged != nil }

// maybeAddRTVI puts an RTVI processor at the head of the pipeline, and its
// observer alongside the others, unless the pipeline already carries one.
//
// A client talks to the session over RTVI, so a pipeline usually wants one and
// gets it without having to be built for it. A pipeline that already contains
// one keeps it: adding a second would put it in the frame path twice.
func (t *Worker) maybeAddRTVI(pipe processor.Processor, cfg *WorkerConfig) processor.Processor {
	existing := findRTVI(pipe)
	hasObserver := false
	for _, o := range cfg.Observers {
		if _, ok := o.(*rtvi.Observer); ok {
			hasObserver = true
			break
		}
	}

	switch {
	case existing != nil && !hasObserver:
		// The two go together: the processor sends to the client and the
		// observer is what feeds it.
		slog.Error("the pipeline has an RTVI processor but no RTVI observer was given; add both")
		t.rtvi = existing
		return pipe
	case existing == nil && hasObserver:
		slog.Error("an RTVI observer was given but the pipeline has no RTVI processor; add both")
		return pipe
	case existing != nil && hasObserver:
		slog.Warn("the pipeline already has an RTVI processor and observer, keeping them; " +
			"a worker adds both itself, so they need not be added by hand")
		t.rtvi = existing
		return pipe
	case !boolOrTrue(cfg.EnableRTVI):
		return pipe
	}

	t.rtvi = cfg.RTVIProcessor
	if t.rtvi == nil {
		t.rtvi = rtvi.NewProcessor()
	}
	observer := rtvi.NewObserver(t.rtvi)
	if cfg.RTVIObserverParams != nil {
		observer = rtvi.NewObserverWithParams(t.rtvi, *cfg.RTVIObserverParams)
	}
	cfg.Observers = append(append([]Observer(nil), cfg.Observers...), observer)

	return New(t.rtvi, pipe)
}

// RTVI is the processor a client talks to, whether this worker added it or the
// pipeline already carried one. It is nil when the worker runs without RTVI.
func (t *Worker) RTVI() *rtvi.Processor { return t.rtvi }

// findRTVI looks for an RTVI processor anywhere in a pipeline, including inside
// a nested one.
func findRTVI(p processor.Processor) *rtvi.Processor {
	if r, ok := p.(*rtvi.Processor); ok {
		return r
	}
	for _, child := range p.Processors() {
		if r := findRTVI(child); r != nil {
			return r
		}
	}
	return nil
}
