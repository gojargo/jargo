package frames

import (
	"fmt"
	"sync"
	"time"
)

// EndFrame indicates the pipeline has ended and processors should shut down. As
// a control frame it is received in order, after preceding frames are flushed.
// It is uninterruptible so it survives an interruption and the pipeline always
// shuts down cleanly.
type EndFrame struct {
	BaseControlFrame
	UninterruptibleMixin
	// Reason describes why the pipeline is ending; "" when unset.
	Reason string
}

// NewEndFrame builds an EndFrame.
func NewEndFrame() *EndFrame {
	return &EndFrame{BaseControlFrame: NewBaseControlFrame("EndFrame")}
}

// String implements fmt.Stringer.
func (f *EndFrame) String() string {
	return fmt.Sprintf("%s(reason: %s)", f.Name(), f.Reason)
}

// StopFrame indicates the pipeline should stop but that its processors are to be
// kept running, ready for another run. Unlike EndFrame it does not shut
// processors down. It is normally queued by the Task. It is uninterruptible so a
// barge-in cannot drop it.
type StopFrame struct {
	BaseControlFrame
	UninterruptibleMixin
}

// NewStopFrame builds a StopFrame.
func NewStopFrame() *StopFrame {
	return &StopFrame{BaseControlFrame: NewBaseControlFrame("StopFrame")}
}

// PipelineFlushFrame is a probe that reports when the pipeline has drained. It is
// pushed downstream; the Task's sink bounces it back upstream, and when it
// reaches the source the Task closes Done. Waiting on Done therefore means every
// frame queued ahead of the probe has completed its round trip — useful to let
// the pipeline settle after an interruption before injecting new work.
//
// It is uninterruptible so the probe survives an InterruptionFrame and still
// completes its round trip; otherwise a waiter would block forever. Done is
// carried on the frame so concurrent flushes stay isolated, each awaiting its
// own probe.
type PipelineFlushFrame struct {
	BaseControlFrame
	UninterruptibleMixin
	// Done is closed by the Task once the probe has completed its round trip.
	Done chan struct{}

	closeOnce sync.Once
}

// NewPipelineFlushFrame builds a PipelineFlushFrame with a fresh Done channel.
// Wait on Done (against a context) to know the pipeline has drained.
func NewPipelineFlushFrame() *PipelineFlushFrame {
	return &PipelineFlushFrame{
		BaseControlFrame: NewBaseControlFrame("PipelineFlushFrame"),
		Done:             make(chan struct{}),
	}
}

// CloseDone closes Done, releasing whoever is waiting on the probe. The Task
// calls it when the probe completes its round trip. Unlike the rest of a frame's
// state this is safe to call from any goroutine, and more than once: the probe is
// a deliberate handoff between the waiter and the pipeline.
func (f *PipelineFlushFrame) CloseDone() {
	f.closeOnce.Do(func() { close(f.Done) })
}

// ProposedUserStoppedSpeakingFrame proposes that the user's turn has ended. Like
// its counterpart it is a proposal an external turn-stop strategy resolves into
// a UserStoppedSpeakingFrame.
//
// It is a control frame so that it stays ordered against the final
// TranscriptionFrame. A service with turn detection of its own pushes that
// transcript and then proposes the stop, and the strategy needs the text in hand
// to close the turn on it.
type ProposedUserStoppedSpeakingFrame struct {
	BaseControlFrame
}

// NewProposedUserStoppedSpeakingFrame builds a ProposedUserStoppedSpeakingFrame.
func NewProposedUserStoppedSpeakingFrame() *ProposedUserStoppedSpeakingFrame {
	return &ProposedUserStoppedSpeakingFrame{
		BaseControlFrame: NewBaseControlFrame("ProposedUserStoppedSpeakingFrame"),
	}
}

// LLMFullResponseStartFrame marks the beginning of an LLM response, followed by
// one or more TextFrames and a final LLMFullResponseEndFrame. It is a control
// frame.
type LLMFullResponseStartFrame struct {
	BaseControlFrame
	// SkipTTS, when set, reports whether the response should skip TTS. A nil
	// value means "unset".
	SkipTTS *bool
}

// NewLLMFullResponseStartFrame builds an LLMFullResponseStartFrame.
func NewLLMFullResponseStartFrame() *LLMFullResponseStartFrame {
	return &LLMFullResponseStartFrame{BaseControlFrame: NewBaseControlFrame("LLMFullResponseStartFrame")}
}

// LLMFullResponseEndFrame marks the end of an LLM response. It is a control
// frame.
type LLMFullResponseEndFrame struct {
	BaseControlFrame
	// SkipTTS, when set, reports whether the response should skip TTS. A nil
	// value means "unset".
	SkipTTS *bool
}

// NewLLMFullResponseEndFrame builds an LLMFullResponseEndFrame.
func NewLLMFullResponseEndFrame() *LLMFullResponseEndFrame {
	return &LLMFullResponseEndFrame{BaseControlFrame: NewBaseControlFrame("LLMFullResponseEndFrame")}
}

// LLMAssistantPushAggregationFrame makes the assistant aggregator commit what it
// has gathered to the conversation as an assistant message, without waiting for
// an LLMFullResponseEndFrame. It closes out an utterance the service spoke on
// its own, which has no LLM response around it to end. It is a control frame.
type LLMAssistantPushAggregationFrame struct {
	BaseControlFrame
}

// NewLLMAssistantPushAggregationFrame builds an LLMAssistantPushAggregationFrame.
func NewLLMAssistantPushAggregationFrame() *LLMAssistantPushAggregationFrame {
	return &LLMAssistantPushAggregationFrame{
		BaseControlFrame: NewBaseControlFrame("LLMAssistantPushAggregationFrame"),
	}
}

// TTSStartedFrame marks the beginning of a TTS response. The following
// TTSAudioRawFrames are part of the response until a TTSStoppedFrame. It is a
// control frame.
type TTSStartedFrame struct {
	BaseControlFrame
	// ContextID identifies this TTS context; "" when unset.
	ContextID string
	// AppendToContext reports whether the spoken text for this response will be
	// appended to the LLM context. Defaults to true.
	AppendToContext bool
}

// NewTTSStartedFrame builds a TTSStartedFrame.
func NewTTSStartedFrame() *TTSStartedFrame {
	return &TTSStartedFrame{BaseControlFrame: NewBaseControlFrame("TTSStartedFrame"), AppendToContext: true}
}

// TTSStoppedFrame marks the end of a TTS response. It is a control frame.
type TTSStoppedFrame struct {
	BaseControlFrame
	// ContextID identifies this TTS context; "" when unset.
	ContextID string
}

// NewTTSStoppedFrame builds a TTSStoppedFrame.
func NewTTSStoppedFrame() *TTSStoppedFrame {
	return &TTSStoppedFrame{BaseControlFrame: NewBaseControlFrame("TTSStoppedFrame")}
}

// FrameProcessorPauseFrame asks a processor to pause its handling of data and
// control frames. Paused frames stay in the processor's queue and are handled
// once processing resumes with a FrameProcessorResumeFrame. It is a control
// frame, so it is received in order, after the frames queued ahead of it; use
// FrameProcessorPauseUrgentFrame to pause immediately instead.
type FrameProcessorPauseFrame struct {
	BaseControlFrame
	// Processor is the processor to pause.
	Processor ProcessorTarget
}

// NewFrameProcessorPauseFrame builds a FrameProcessorPauseFrame addressed to p.
func NewFrameProcessorPauseFrame(p ProcessorTarget) *FrameProcessorPauseFrame {
	return &FrameProcessorPauseFrame{
		BaseControlFrame: NewBaseControlFrame("FrameProcessorPauseFrame"),
		Processor:        p,
	}
}

// FrameProcessorResumeFrame asks a processor to resume the handling of data and
// control frames it paused. Queued frames are then handled in the order they
// were received. It is a control frame, so it is received in order, after the
// frames queued ahead of it; use FrameProcessorResumeUrgentFrame to resume
// immediately instead.
type FrameProcessorResumeFrame struct {
	BaseControlFrame
	// Processor is the processor to resume.
	Processor ProcessorTarget
}

// NewFrameProcessorResumeFrame builds a FrameProcessorResumeFrame addressed to p.
func NewFrameProcessorResumeFrame(p ProcessorTarget) *FrameProcessorResumeFrame {
	return &FrameProcessorResumeFrame{
		BaseControlFrame: NewBaseControlFrame("FrameProcessorResumeFrame"),
		Processor:        p,
	}
}

// HeartbeatFrame is pushed through the pipeline at a fixed interval so the Task
// can tell frames are still moving. It travels in order like any control frame,
// which is what makes it a measure of the path real work takes. It is
// interruptible on purpose: a barge-in drops the ones in flight, and the next
// interval sends another. It is a control frame.
type HeartbeatFrame struct {
	BaseControlFrame
	// Timestamp is the pipeline clock reading when the heartbeat was created.
	// Comparing it with the clock on arrival gives the time the frame spent
	// crossing the pipeline.
	Timestamp time.Duration
}

// NewHeartbeatFrame builds a HeartbeatFrame stamped with ts.
func NewHeartbeatFrame(ts time.Duration) *HeartbeatFrame {
	return &HeartbeatFrame{
		BaseControlFrame: NewBaseControlFrame("HeartbeatFrame"),
		Timestamp:        ts,
	}
}

// String implements fmt.Stringer.
func (f *HeartbeatFrame) String() string {
	return fmt.Sprintf("%s(timestamp: %s)", f.Name(), f.Timestamp)
}

// Compile-time interface checks.
var (
	_ ControlFrame    = (*HeartbeatFrame)(nil)
	_ ControlFrame    = (*FrameProcessorPauseFrame)(nil)
	_ ControlFrame    = (*FrameProcessorResumeFrame)(nil)
	_ ControlFrame    = (*EndFrame)(nil)
	_ Uninterruptible = (*EndFrame)(nil)
	_ ControlFrame    = (*StopFrame)(nil)
	_ Uninterruptible = (*StopFrame)(nil)
	_ ControlFrame    = (*PipelineFlushFrame)(nil)
	_ Uninterruptible = (*PipelineFlushFrame)(nil)
	_ ControlFrame    = (*ProposedUserStoppedSpeakingFrame)(nil)
	_ ControlFrame    = (*LLMFullResponseStartFrame)(nil)
	_ ControlFrame    = (*LLMFullResponseEndFrame)(nil)
	_ ControlFrame    = (*LLMAssistantPushAggregationFrame)(nil)
	_ ControlFrame    = (*TTSStartedFrame)(nil)
	_ ControlFrame    = (*TTSStoppedFrame)(nil)
)

// LLMThoughtStartFrame marks the beginning of a thought from a reasoning model,
// followed by one or more LLMThoughtTextFrames and a final LLMThoughtEndFrame.
// It is a control frame.
type LLMThoughtStartFrame struct {
	BaseControlFrame
	// AppendToContext reports whether the thought should be written to the
	// conversation. A thought that is written goes in as a message of the
	// provider's own, so LLM must name that provider when this is set.
	AppendToContext bool
	// LLM identifies the provider whose native message the thought is written
	// as. It is only read when AppendToContext is set.
	LLM string
}

// NewLLMThoughtStartFrame builds an LLMThoughtStartFrame.
func NewLLMThoughtStartFrame() *LLMThoughtStartFrame {
	return &LLMThoughtStartFrame{BaseControlFrame: NewBaseControlFrame("LLMThoughtStartFrame")}
}

// String implements fmt.Stringer.
func (f *LLMThoughtStartFrame) String() string {
	return fmt.Sprintf("%s(append_to_context: %t, llm: %s)", f.Name(), f.AppendToContext, f.LLM)
}

// LLMThoughtTextFrame carries the text of a thought, or a chunk of one.
//
// Despite carrying text it is a data frame rather than a TextFrame, which is
// what keeps it out of the ordinary text handling: a thought is the model
// reasoning with itself and must not be spoken.
type LLMThoughtTextFrame struct {
	BaseDataFrame
	// Text is the thought, or a chunk of it.
	Text string
	// IncludesInterFrameSpaces reports whether the spacing between chunks is
	// already part of Text. A thought's chunks always carry their own spacing.
	IncludesInterFrameSpaces bool
}

// NewLLMThoughtTextFrame builds an LLMThoughtTextFrame.
func NewLLMThoughtTextFrame(text string) *LLMThoughtTextFrame {
	return &LLMThoughtTextFrame{
		BaseDataFrame: NewBaseDataFrame("LLMThoughtTextFrame"),
		Text:          text,
		// The chunks a provider streams already include whatever spacing runs
		// between them.
		IncludesInterFrameSpaces: true,
	}
}

// String implements fmt.Stringer.
func (f *LLMThoughtTextFrame) String() string {
	return fmt.Sprintf("%s(thought text: %s)", f.Name(), f.Text)
}

// LLMThoughtEndFrame marks the end of a thought. It is a control frame.
type LLMThoughtEndFrame struct {
	BaseControlFrame
	// Signature is what the provider signs the thought with, where it signs one.
	// It is carried back unread, since only that provider can make sense of it.
	Signature any
}

// NewLLMThoughtEndFrame builds an LLMThoughtEndFrame.
func NewLLMThoughtEndFrame() *LLMThoughtEndFrame {
	return &LLMThoughtEndFrame{BaseControlFrame: NewBaseControlFrame("LLMThoughtEndFrame")}
}

// String implements fmt.Stringer.
func (f *LLMThoughtEndFrame) String() string {
	return fmt.Sprintf("%s(signature: %v)", f.Name(), f.Signature)
}
