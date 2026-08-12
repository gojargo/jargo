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
	if len(b.observers) == 0 {
		return
	}
	data := FramePushed{
		Source:      b.self,
		Destination: dst,
		Frame:       f,
		Direction:   dir,
		Timestamp:   b.now(),
	}
	for _, o := range b.observers {
		o.OnPushFrame(data)
	}
}

// notifyProcess reports a frame reaching this processor.
func (b *Base) notifyProcess(f frames.Frame, dir Direction) {
	if len(b.observers) == 0 {
		return
	}
	data := FrameProcessed{
		Processor: b.self,
		Frame:     f,
		Direction: dir,
		Timestamp: b.now(),
	}
	for _, o := range b.observers {
		if po, ok := o.(ProcessObserver); ok {
			po.OnProcessFrame(data)
		}
	}
}

// now reads the pipeline clock, or zero before the pipeline has one.
func (b *Base) now() time.Duration {
	if b.clock == nil {
		return 0
	}
	return b.clock.Time()
}
