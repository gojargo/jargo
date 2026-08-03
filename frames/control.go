package frames

import (
	"fmt"
	"sync"
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

// Compile-time interface checks.
var (
	_ ControlFrame    = (*FrameProcessorPauseFrame)(nil)
	_ ControlFrame    = (*FrameProcessorResumeFrame)(nil)
	_ ControlFrame    = (*EndFrame)(nil)
	_ Uninterruptible = (*EndFrame)(nil)
	_ ControlFrame    = (*StopFrame)(nil)
	_ Uninterruptible = (*StopFrame)(nil)
	_ ControlFrame    = (*PipelineFlushFrame)(nil)
	_ Uninterruptible = (*PipelineFlushFrame)(nil)
	_ ControlFrame    = (*LLMFullResponseStartFrame)(nil)
	_ ControlFrame    = (*LLMFullResponseEndFrame)(nil)
	_ ControlFrame    = (*TTSStartedFrame)(nil)
	_ ControlFrame    = (*TTSStoppedFrame)(nil)
)
