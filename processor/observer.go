package processor

import (
	"time"

	"github.com/gojargo/jargo/frames"
)

// FramePushed is one frame moving from one processor to the next.
type FramePushed struct {
	// Source is the processor sending the frame.
	Source Processor
	// Destination is the processor receiving it.
	Destination Processor
	// Frame is the frame being handed over.
	Frame frames.Frame
	// Direction is which way it is going.
	Direction Direction
	// Timestamp is when it was pushed, on the pipeline clock.
	Timestamp time.Duration
}

// FrameProcessed is one frame reaching a processor.
type FrameProcessed struct {
	// Processor is the processor handling the frame.
	Processor Processor
	// Frame is the frame being handled.
	Frame frames.Frame
	// Direction is which way it is going.
	Direction Direction
	// Timestamp is when it was handed over, on the pipeline clock.
	Timestamp time.Duration
}

// ProcessorSetUp is one processor having been set up.
//
// Processors are set up concurrently and before any frame flows, so this is what
// a timing observer measures the work a processor does to get ready by. The
// times are wall-clock readings carrying a monotonic reading, since the pipeline
// clock is not what these are offsets from.
type ProcessorSetUp struct {
	// Processor is the processor that was set up.
	Processor Processor
	// StartedAt is when the processor's Setup began.
	StartedAt time.Time
	// FinishedAt is when it returned.
	FinishedAt time.Time
}

// Duration is what the processor's Setup cost.
func (d ProcessorSetUp) Duration() time.Duration { return d.FinishedAt.Sub(d.StartedAt) }

// Observer watches frames flowing through a pipeline without modifying them, to
// derive turn, latency or startup metrics, to log the stream, or to report
// events to a client.
//
// Every handover between processors is reported, not only what reaches the ends
// of the pipeline, so an observer sees where each frame came from. That is what
// lets it tell a frame that has been through the output transport, and so
// carries real playback timing, from the same frame earlier in the pipeline.
//
// Observers must be safe for concurrent use: a pipeline's processors each run on
// their own goroutine, so the methods may be called from any of them.
type Observer interface {
	// OnPushFrame reports one frame handed from one processor to the next.
	OnPushFrame(data FramePushed)
}

// ProcessObserver is an optional interface an Observer implements to also see a
// frame as it reaches a processor, before that processor has handled it.
type ProcessObserver interface {
	Observer
	// OnProcessFrame reports one frame reaching a processor.
	OnProcessFrame(data FrameProcessed)
}

// SetupObserver is an optional interface an Observer implements to also hear
// that a processor has been set up.
//
// A processor connects and does its other slow start-up work there, so this is
// where that cost can be measured. Processors are set up concurrently, so these
// arrive in the order they finish rather than in pipeline order.
type SetupObserver interface {
	Observer
	// OnProcessorSetup reports one processor having been set up.
	OnProcessorSetup(data ProcessorSetUp)
}

// NotifyProcessorSetup reports a processor having been set up to every observer
// listening for it. A pipeline calls it as each of its processors is set up.
func NotifyProcessorSetup(observers []Observer, data ProcessorSetUp) {
	for _, o := range observers {
		if so, ok := o.(SetupObserver); ok {
			so.OnProcessorSetup(data)
		}
	}
}

// SetupStartedObserver is an optional interface an Observer implements to hear
// that the pipeline has begun setting its processors up.
//
// It arrives before any processor has been set up, so an observer timing the
// start of a session measures from here. Processors connect while they are set
// up, so this is earlier than the StartFrame and is what the pipeline clock runs
// from.
type SetupStartedObserver interface {
	Observer
	// OnPipelineSetupStarted reports the instant the pipeline began setting its
	// processors up.
	OnPipelineSetupStarted(at time.Time)
}

// PipelineStartedObserver is an optional interface an Observer implements to
// hear that the pipeline has fully started, which is the StartFrame having been
// handled by every processor, including the branches of a parallel pipeline.
//
// It is reported in order with the frames, so an observer that sets itself up
// here has done so before the first frame of the conversation reaches it.
type PipelineStartedObserver interface {
	Observer
	// OnPipelineStarted reports that the pipeline has started.
	OnPipelineStarted()
}

// notifyPush reports a handover to every observer.
func (b *Base) notifyPush(f frames.Frame, dir Direction, dst Processor) {
	observers := b.setupState().observers
	if len(observers) == 0 {
		return
	}
	data := FramePushed{
		Source:      b.self,
		Destination: dst,
		Frame:       f,
		Direction:   dir,
		Timestamp:   b.now(),
	}
	for _, o := range observers {
		o.OnPushFrame(data)
	}
}

// notifyProcess reports a frame reaching this processor.
func (b *Base) notifyProcess(f frames.Frame, dir Direction) {
	observers := b.setupState().observers
	if len(observers) == 0 {
		return
	}
	data := FrameProcessed{
		Processor: b.self,
		Frame:     f,
		Direction: dir,
		Timestamp: b.now(),
	}
	for _, o := range observers {
		if po, ok := o.(ProcessObserver); ok {
			po.OnProcessFrame(data)
		}
	}
}

// now reads the pipeline clock, or zero before the pipeline has one.
func (b *Base) now() time.Duration {
	c := b.setupState().clock
	if c == nil {
		return 0
	}
	return c.Time()
}
