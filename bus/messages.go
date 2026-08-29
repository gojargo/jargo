package bus

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/registry"
)

// Every message carries who sent it and, optionally, who it is for. A message
// with no target is a broadcast: every subscriber receives it and decides for
// itself whether it applies.
//
// A message falls in one of two bands. A system message is delivered ahead of
// the data messages already queued, because it is how a worker is told to stop
// and that must not wait behind the work it is stopping. Everything else is a
// data message, delivered in the order it was sent.
//
// A message may also be local, which means it never leaves the process: it
// carries something a remote bus could not act on, an object rather than data.

// Message is one message carried by the bus.
type Message interface {
	// Source is the name of the worker or component that sent it.
	Source() string
	// Target is the name of the worker it is for, or "" to broadcast.
	Target() string

	base() *BaseMessage
}

// SystemMessage is a message delivered ahead of the queued data messages.
type SystemMessage interface {
	Message
	isSystemMessage()
}

// LocalMessage is a message that stays on this bus and is never forwarded to a
// remote one.
type LocalMessage interface {
	Message
	isLocalMessage()
}

// BaseMessage carries what every message has. Embed BaseDataMessage or
// BaseSystemMessage rather than this directly.
type BaseMessage struct {
	// From is the name of the sender.
	From string
	// To is the name of the intended recipient, or "" to broadcast.
	To string
}

// Source implements Message.
func (m *BaseMessage) Source() string { return m.From }

// Target implements Message.
func (m *BaseMessage) Target() string { return m.To }

func (m *BaseMessage) base() *BaseMessage { return m }

// BaseDataMessage is embedded by every message delivered in send order.
type BaseDataMessage struct{ BaseMessage }

// BaseSystemMessage is embedded by every message delivered ahead of the queued
// data messages.
type BaseSystemMessage struct{ BaseMessage }

func (m *BaseSystemMessage) isSystemMessage() {}

// localMixin is embedded by a message that never leaves this process.
type localMixin struct{}

func (localMixin) isLocalMessage() {}

// FrameMessage carries a pipeline frame between workers, which is how a worker
// bridged into another's pipeline exchanges work with it.
type FrameMessage struct {
	BaseDataMessage
	// Frame is the frame being carried.
	Frame frames.Frame
	// Direction is the direction it should travel in on arrival.
	Direction processor.Direction
	// Bridge names the bridge it came through, or "" when it did not come
	// through one.
	Bridge string
}

// TTSSpeakMessage asks a worker to say something.
type TTSSpeakMessage struct {
	BaseDataMessage
	// Text is what to say.
	Text string
	// AppendToContext reports whether what is said joins the conversation.
	AppendToContext bool
}

// FlushProgressMessage reports that a flush probe from another worker is still
// making progress. A probe that crosses into another pipeline is answered there,
// so the worker waiting on it cannot see whether anything is happening. The
// pipeline holding the probe says so, and the wait stays alive for as long as it
// keeps saying it.
type FlushProgressMessage struct {
	BaseDataMessage
	// FlushID is the id of the probe being reported on.
	FlushID uint64
}

// ActivateWorkerMessage asks a worker to become active.
type ActivateWorkerMessage struct {
	BaseDataMessage
	// Args are the activation arguments, and may be nil.
	Args map[string]any
}

// DeactivateWorkerMessage asks a worker to become inactive.
type DeactivateWorkerMessage struct{ BaseDataMessage }

// EndMessage asks everything to shut down gracefully.
type EndMessage struct {
	BaseDataMessage
	// Reason describes why, and may be empty.
	Reason string
}

// EndWorkerMessage asks one worker to shut down gracefully.
type EndWorkerMessage struct {
	BaseDataMessage
	// Reason describes why, and may be empty.
	Reason string
}

// CancelMessage stops everything at once, without flushing queued work.
type CancelMessage struct {
	BaseSystemMessage
	// Reason describes why, and may be empty.
	Reason string
}

// CancelWorkerMessage stops one worker at once.
type CancelWorkerMessage struct {
	BaseSystemMessage
	// Reason describes why, and may be empty.
	Reason string
}

// AddWorkerMessage hands a worker to the runner to manage. It never leaves the
// process, because it carries the worker itself rather than a description of
// one.
//
// Worker is typed as any because the workers are built on this package and so
// cannot be named here. Set it to a worker; anything else is ignored by the
// runner that receives it.
type AddWorkerMessage struct {
	BaseSystemMessage
	localMixin
	// Worker is the worker to manage.
	Worker any
}

// WorkerRegistryMessage announces which workers a runner knows about, so the
// runners on a bus can learn of each other's workers.
type WorkerRegistryMessage struct {
	BaseSystemMessage
	// Runner is the runner the snapshot belongs to.
	Runner string
	// Workers is the snapshot.
	Workers []registry.WorkerRegistryEntry
}

// WorkerReadyMessage announces that a worker has started and can be addressed.
type WorkerReadyMessage struct {
	BaseDataMessage
	// Runner is the runner managing the worker.
	Runner string
	// Parent is the worker's parent, or "" for a root worker.
	Parent string
	// Active reports whether it is currently active.
	Active bool
	// Bridged reports whether it is bridged.
	Bridged bool
	// StartedAt is when it became ready, as a Unix timestamp, and zero when
	// unset.
	StartedAt float64
}

// WorkerErrorMessage reports a worker that failed, to everyone.
type WorkerErrorMessage struct {
	BaseSystemMessage
	// Error describes the failure.
	Error string
}

// WorkerLocalErrorMessage reports a worker that failed, to this process only.
type WorkerLocalErrorMessage struct {
	BaseSystemMessage
	localMixin
	// Error describes the failure.
	Error string
}

// JobRequestMessage asks a worker to carry out a job.
type JobRequestMessage struct {
	BaseDataMessage
	// JobID identifies the job, and is what every later message about it
	// carries.
	JobID string
	// JobName names the kind of work, and may be empty.
	JobName string
	// Payload is the job's input, and may be nil.
	Payload map[string]any
}

// A job's progress and its outcome each travel in either band: ordinarily in
// send order, or urgently, ahead of the work already queued. The two versions
// of each carry the same body, so a handler takes whichever arrived through the
// JobResponse or JobUpdate interface and reads it the same way.

// JobResult is the body of a message reporting how a job ended.
type JobResult struct {
	// JobID identifies the job.
	JobID string
	// Status is how it ended.
	Status jobcontext.JobStatus
	// Response is the job's output, and may be nil.
	Response map[string]any
}

// Result is the body, and is what makes a message a JobResponse.
func (r *JobResult) Result() *JobResult { return r }

// JobResponse reports how a job ended, whichever band it travels in.
type JobResponse interface {
	Message
	// Result is what the message reports.
	Result() *JobResult
}

// JobProgress is the body of a message reporting progress on a running job.
type JobProgress struct {
	// JobID identifies the job.
	JobID string
	// Update is the progress being reported, and may be nil.
	Update map[string]any
}

// Progress is the body, and is what makes a message a JobUpdate.
func (p *JobProgress) Progress() *JobProgress { return p }

// JobUpdate reports progress on a running job, whichever band it travels in.
type JobUpdate interface {
	Message
	// Progress is what the message reports.
	Progress() *JobProgress
}

// JobResponseMessage reports how a job ended.
type JobResponseMessage struct {
	BaseDataMessage
	JobResult
}

// JobResponseUrgentMessage reports how a job ended, ahead of the queued work.
type JobResponseUrgentMessage struct {
	BaseSystemMessage
	JobResult
}

// JobUpdateMessage reports progress on a job still running.
type JobUpdateMessage struct {
	BaseDataMessage
	JobProgress
}

// JobUpdateUrgentMessage reports progress ahead of the queued work.
type JobUpdateUrgentMessage struct {
	BaseSystemMessage
	JobProgress
}

// JobUpdateRequestMessage asks for the current progress of a job.
type JobUpdateRequestMessage struct {
	BaseDataMessage
	// JobID identifies the job.
	JobID string
}

// JobCancelMessage calls a job off. It is a system message, so it reaches the
// worker ahead of the work it is calling off.
type JobCancelMessage struct {
	BaseSystemMessage
	// JobID identifies the job.
	JobID string
	// Reason describes why, and may be empty.
	Reason string
}

// JobStreamStartMessage opens a stream of results for a job.
type JobStreamStartMessage struct {
	BaseDataMessage
	// JobID identifies the job.
	JobID string
	// Data is what opens the stream, and may be nil.
	Data map[string]any
}

// JobStreamDataMessage carries one item of a job's result stream.
type JobStreamDataMessage struct {
	BaseDataMessage
	// JobID identifies the job.
	JobID string
	// Data is the item, and may be nil.
	Data map[string]any
}

// JobStreamEndMessage closes a job's result stream.
type JobStreamEndMessage struct {
	BaseDataMessage
	// JobID identifies the job.
	JobID string
	// Data is what closes the stream, and may be nil.
	Data map[string]any
}

// Compile-time checks that each message is in the band it belongs to.
var (
	_ Message       = (*FrameMessage)(nil)
	_ Message       = (*TTSSpeakMessage)(nil)
	_ Message       = (*ActivateWorkerMessage)(nil)
	_ Message       = (*DeactivateWorkerMessage)(nil)
	_ Message       = (*EndMessage)(nil)
	_ Message       = (*EndWorkerMessage)(nil)
	_ SystemMessage = (*CancelMessage)(nil)
	_ SystemMessage = (*CancelWorkerMessage)(nil)
	_ SystemMessage = (*AddWorkerMessage)(nil)
	_ LocalMessage  = (*AddWorkerMessage)(nil)
	_ SystemMessage = (*WorkerRegistryMessage)(nil)
	_ Message       = (*WorkerReadyMessage)(nil)
	_ SystemMessage = (*WorkerErrorMessage)(nil)
	_ SystemMessage = (*WorkerLocalErrorMessage)(nil)
	_ LocalMessage  = (*WorkerLocalErrorMessage)(nil)
	_ Message       = (*JobRequestMessage)(nil)
	_ JobResponse   = (*JobResponseMessage)(nil)
	_ SystemMessage = (*JobResponseUrgentMessage)(nil)
	_ JobResponse   = (*JobResponseUrgentMessage)(nil)
	_ JobUpdate     = (*JobUpdateMessage)(nil)
	_ SystemMessage = (*JobUpdateUrgentMessage)(nil)
	_ JobUpdate     = (*JobUpdateUrgentMessage)(nil)
	_ Message       = (*JobUpdateRequestMessage)(nil)
	_ SystemMessage = (*JobCancelMessage)(nil)
	_ Message       = (*JobStreamStartMessage)(nil)
	_ Message       = (*JobStreamDataMessage)(nil)
	_ Message       = (*JobStreamEndMessage)(nil)
)
