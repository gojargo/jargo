package frames

import "fmt"

// The worker frames ask the pipeline worker — the Task — to change the run's
// lifecycle. A processor deep in the pipeline has no handle on the Task, so it
// pushes one of these instead and the Task converts it into the corresponding
// pipeline-wide frame: EndWorkerFrame becomes an EndFrame, CancelWorkerFrame a
// CancelFrame, and so on. Push them downstream (the usual direction) so frames
// queued ahead of them are processed first; the Task's sink returns them
// upstream. Pushing upstream directly is also supported.
//
// The two graceful requests are control frames, so they are ordered behind the
// work already queued, and uninterruptible, so a barge-in cannot drop the
// request. The two immediate requests are system frames, so they reach the Task
// even while the pipeline is busy.

// EndWorkerFrame requests a graceful shutdown of the pipeline worker (the Task).
// On reaching the Task it queues an EndFrame downstream, so frames already
// queued are flushed and the bot finishes speaking before the pipeline ends.
type EndWorkerFrame struct {
	BaseControlFrame
	UninterruptibleMixin
	// Reason describes why the run is ending; "" when unset.
	Reason string
}

// NewEndWorkerFrame builds an EndWorkerFrame.
func NewEndWorkerFrame() *EndWorkerFrame {
	return &EndWorkerFrame{BaseControlFrame: NewBaseControlFrame("EndWorkerFrame")}
}

// String implements fmt.Stringer.
func (f *EndWorkerFrame) String() string {
	return fmt.Sprintf("%s(reason: %s)", f.Name(), f.Reason)
}

// StopWorkerFrame requests that the Task stop once queued frames are flushed,
// while leaving the processors running and ready for another run. On reaching the
// Task it queues a StopFrame. It is the counterpart of EndWorkerFrame for a run
// that should stop without shutting processors down.
type StopWorkerFrame struct {
	BaseControlFrame
	UninterruptibleMixin
}

// NewStopWorkerFrame builds a StopWorkerFrame.
func NewStopWorkerFrame() *StopWorkerFrame {
	return &StopWorkerFrame{BaseControlFrame: NewBaseControlFrame("StopWorkerFrame")}
}

// CancelWorkerFrame requests immediate cancellation of the Task. On reaching it
// the Task queues a CancelFrame downstream, so the pipeline stops at once without
// flushing queued frames. It is the deliberate counterpart to a fatal ErrorFrame:
// use it to stop a run that is not failing (the caller hung up, a supervisor
// asked to stop).
type CancelWorkerFrame struct {
	BaseSystemFrame
	// Reason describes why the run is being canceled; "" when unset.
	Reason string
}

// NewCancelWorkerFrame builds a CancelWorkerFrame.
func NewCancelWorkerFrame() *CancelWorkerFrame {
	return &CancelWorkerFrame{BaseSystemFrame: NewBaseSystemFrame("CancelWorkerFrame")}
}

// String implements fmt.Stringer.
func (f *CancelWorkerFrame) String() string {
	return fmt.Sprintf("%s(reason: %s)", f.Name(), f.Reason)
}

// InterruptionWorkerFrame asks the Task to interrupt the pipeline. On reaching
// the Task it queues an InterruptionFrame downstream. A processor that can
// broadcast an InterruptionFrame itself should do so; this frame is for one that
// only has a path to the Task.
type InterruptionWorkerFrame struct {
	BaseSystemFrame
}

// NewInterruptionWorkerFrame builds an InterruptionWorkerFrame.
func NewInterruptionWorkerFrame() *InterruptionWorkerFrame {
	return &InterruptionWorkerFrame{BaseSystemFrame: NewBaseSystemFrame("InterruptionWorkerFrame")}
}

// Compile-time interface checks.
var (
	_ ControlFrame    = (*EndWorkerFrame)(nil)
	_ Uninterruptible = (*EndWorkerFrame)(nil)
	_ ControlFrame    = (*StopWorkerFrame)(nil)
	_ Uninterruptible = (*StopWorkerFrame)(nil)
	_ SystemFrame     = (*CancelWorkerFrame)(nil)
	_ SystemFrame     = (*InterruptionWorkerFrame)(nil)
)
