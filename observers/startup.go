package observers

import (
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// ProcessorStartupTiming is what one processor cost to start.
type ProcessorStartupTiming struct {
	// ProcessorName is the name of the processor.
	ProcessorName string
	// StartOffset is how long after the StartFrame entered the pipeline this
	// processor began starting.
	StartOffset time.Duration
	// Duration is how long the processor held the StartFrame, which is what its
	// own start cost: opening a socket, authenticating, loading a model.
	Duration time.Duration
}

// StartupTimingReport is what every measured processor cost to start.
type StartupTimingReport struct {
	// StartTime is the wall-clock time at which the first processor began
	// starting.
	StartTime time.Time
	// TotalDuration is the sum of the per-processor durations. Processors on
	// separate branches of a parallel pipeline start at the same time, so it is
	// a total of the work rather than of the wall clock.
	TotalDuration time.Duration
	// ProcessorTimings is what each processor cost, in the order the StartFrame
	// left them.
	ProcessorTimings []ProcessorStartupTiming
}

// TransportTimingReport is how long the transport took to reach the points at
// which a conversation can actually happen.
type TransportTimingReport struct {
	// StartTime is the wall-clock time at which the pipeline started.
	StartTime time.Time
	// BotConnected is how long after the pipeline started the bot itself joined
	// the session. It is nil on a transport that never reports the bot joining,
	// which is every transport that is not an SFU.
	BotConnected *time.Duration
	// ClientConnected is how long after the pipeline started the first remote
	// participant connected.
	ClientConnected time.Duration
}

// StartupTimingConfig configures a StartupTiming observer.
type StartupTimingConfig struct {
	// Track selects the processors to measure. When nil every processor is
	// measured except the pipeline plumbing: the pipelines themselves, and the
	// sources they wrap the head of their chains in.
	Track func(processor.Processor) bool
	// OnStartupTimingReport is called once, when the pipeline has started, with
	// what each processor cost. It is not called when nothing was measured.
	OnStartupTimingReport func(r StartupTimingReport)
	// OnTransportTimingReport is called once, when the first client connects,
	// with how long the transport took to get there.
	OnTransportTimingReport func(r TransportTimingReport)
}

// startFrameInfo is what the observer locks onto: the first StartFrame it sees,
// and when it entered the pipeline. A pipeline is started once, so a second
// StartFrame belongs to something else and is ignored.
type startFrameInfo struct {
	frameID   uint64
	arrival   time.Duration
	wallClock time.Time
}

// arrivalInfo records a processor having been handed the StartFrame, waiting for
// the push that says it has finished starting.
type arrivalInfo struct {
	proc    processor.Processor
	arrival time.Duration
}

// StartupTiming measures what starting a pipeline costs, processor by processor.
//
// A processor does its startup work while handling the StartFrame: connecting to
// a provider, authenticating, loading a model. The observer times the gap
// between that frame reaching the processor and the processor passing it on,
// which is exactly that work, and reports the lot once the pipeline is up.
//
// It reports separately on the transport, which is the other half of how long a
// call takes to become answerable: how long after the pipeline started the bot
// joined the session, and how long until the first client did.
type StartupTiming struct {
	cfg StartupTimingConfig

	mu         sync.Mutex
	startFrame *startFrameInfo
	arrivals   map[uint64]arrivalInfo
	timings    []ProcessorStartupTiming
	// startupReported and transportReported make each report happen once.
	startupReported   bool
	transportReported bool
	// botConnected is kept until the client connects, which is what carries it
	// into the transport report.
	botConnected *time.Duration
}

// NewStartupTiming builds a StartupTiming observer.
func NewStartupTiming(cfg StartupTimingConfig) *StartupTiming {
	return &StartupTiming{cfg: cfg, arrivals: map[uint64]arrivalInfo{}}
}

// shouldTrack reports whether p is measured. The caller holds o.mu.
func (o *StartupTiming) shouldTrack(p processor.Processor) bool {
	if o.cfg.Track != nil {
		return o.cfg.Track(p)
	}
	// A pipeline hands its StartFrame to the chain it contains, so timing it
	// would count the whole chain again under the pipeline's own name; a source
	// is plumbing that starts nothing.
	return !processor.IsSource(p) && len(p.Processors()) == 0
}

// OnProcessFrame implements processor.ProcessObserver. It records the StartFrame
// reaching a processor, which is where that processor's startup begins.
func (o *StartupTiming) OnProcessFrame(data processor.FrameProcessed) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.startupReported {
		return
	}
	if _, ok := data.Frame.(*frames.StartFrame); !ok {
		return
	}
	if o.startFrame == nil {
		o.startFrame = &startFrameInfo{
			frameID:   data.Frame.ID(),
			arrival:   data.Timestamp,
			wallClock: time.Now(),
		}
	} else if data.Frame.ID() != o.startFrame.frameID {
		return
	}
	if o.shouldTrack(data.Processor) {
		o.arrivals[data.Processor.ID()] = arrivalInfo{proc: data.Processor, arrival: data.Timestamp}
	}
}

// OnPushFrame implements processor.Observer. It closes the measurement a
// processor opened, and watches for the transport's connection milestones.
func (o *StartupTiming) OnPushFrame(data processor.FramePushed) {
	o.mu.Lock()
	defer o.mu.Unlock()

	switch data.Frame.(type) {
	case *frames.BotConnectedFrame:
		o.handleBotConnected(data)
		return
	case *frames.ClientConnectedFrame:
		o.handleClientConnected(data)
		return
	}

	if o.startupReported {
		return
	}
	if _, ok := data.Frame.(*frames.StartFrame); !ok {
		return
	}
	if o.startFrame != nil && data.Frame.ID() != o.startFrame.frameID {
		return
	}
	arrival, ok := o.arrivals[data.Source.ID()]
	if !ok || o.startFrame == nil {
		return
	}
	delete(o.arrivals, data.Source.ID())

	o.timings = append(o.timings, ProcessorStartupTiming{
		ProcessorName: arrival.proc.Name(),
		StartOffset:   arrival.arrival - o.startFrame.arrival,
		Duration:      data.Timestamp - arrival.arrival,
	})
}

// OnPipelineStarted implements processor.PipelineStartedObserver. The pipeline
// being up is what says every processor has started, so it is where the report
// is made.
func (o *StartupTiming) OnPipelineStarted() {
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.timings) == 0 {
		return
	}
	o.emitReport()
}

// handleBotConnected records when the bot joined. The caller holds o.mu.
func (o *StartupTiming) handleBotConnected(data processor.FramePushed) {
	if o.botConnected != nil || o.startFrame == nil {
		return
	}
	d := data.Timestamp - o.startFrame.arrival
	o.botConnected = &d
}

// handleClientConnected reports the transport timing on the first client to
// connect. The caller holds o.mu.
func (o *StartupTiming) handleClientConnected(data processor.FramePushed) {
	if o.transportReported || o.startFrame == nil {
		return
	}
	o.transportReported = true
	if o.cfg.OnTransportTimingReport == nil {
		return
	}
	o.cfg.OnTransportTimingReport(TransportTimingReport{
		StartTime:       o.startFrame.wallClock,
		BotConnected:    o.botConnected,
		ClientConnected: data.Timestamp - o.startFrame.arrival,
	})
}

// emitReport builds and reports the startup timings. The caller holds o.mu.
func (o *StartupTiming) emitReport() {
	if o.startupReported {
		return
	}
	o.startupReported = true

	var total time.Duration
	for _, t := range o.timings {
		total += t.Duration
	}
	var startedAt time.Time
	if o.startFrame != nil {
		startedAt = o.startFrame.wallClock
	}
	if o.cfg.OnStartupTimingReport == nil {
		return
	}
	o.cfg.OnStartupTimingReport(StartupTimingReport{
		StartTime:        startedAt,
		TotalDuration:    total,
		ProcessorTimings: append([]ProcessorStartupTiming(nil), o.timings...),
	})
}

// Compile-time interface checks.
var (
	_ processor.Observer                = (*StartupTiming)(nil)
	_ processor.ProcessObserver         = (*StartupTiming)(nil)
	_ processor.PipelineStartedObserver = (*StartupTiming)(nil)
)
