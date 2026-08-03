package frames

import "fmt"

// The sample rates NewStartFrame applies when the application does not override
// them: input is sized for speech recognition, output for synthesis quality.
const (
	// DefaultAudioInSampleRate is the default input audio sample rate in Hz.
	DefaultAudioInSampleRate = 16000
	// DefaultAudioOutSampleRate is the default output audio sample rate in Hz.
	DefaultAudioOutSampleRate = 24000
)

// StartFrame is the first frame pushed down a pipeline. It initializes every
// processor with the pipeline-wide configuration. It is a system frame.
type StartFrame struct {
	BaseSystemFrame
	// AudioInSampleRate is the input audio sample rate in Hz.
	AudioInSampleRate int
	// AudioOutSampleRate is the output audio sample rate in Hz.
	AudioOutSampleRate int
	// EnableMetrics enables performance metrics collection.
	EnableMetrics bool
	// EnableUsageMetrics enables usage metrics collection.
	EnableUsageMetrics bool
	// ReportOnlyInitialTTFB reports only the initial time-to-first-byte.
	ReportOnlyInitialTTFB bool
}

// NewStartFrame builds a StartFrame with the default sample rates (16 kHz in,
// 24 kHz out). Override any field before pushing it.
func NewStartFrame() *StartFrame {
	return &StartFrame{
		BaseSystemFrame:    NewBaseSystemFrame("StartFrame"),
		AudioInSampleRate:  DefaultAudioInSampleRate,
		AudioOutSampleRate: DefaultAudioOutSampleRate,
	}
}

// CancelFrame indicates the pipeline must stop immediately, without processing
// any remaining queued frames. It is a system frame.
type CancelFrame struct {
	BaseSystemFrame
	// Reason describes why the pipeline was canceled; "" when unset.
	Reason string
}

// NewCancelFrame builds a CancelFrame.
func NewCancelFrame() *CancelFrame {
	return &CancelFrame{BaseSystemFrame: NewBaseSystemFrame("CancelFrame")}
}

// String implements fmt.Stringer.
func (f *CancelFrame) String() string {
	return fmt.Sprintf("%s(reason: %s)", f.Name(), f.Reason)
}

// ErrorSource identifies the component that raised an error — in practice the
// frame processor that produced it. It is declared here, rather than imported
// from the processor package, so the frames package keeps no dependency on it;
// a frame processor satisfies this interface by exposing its name.
type ErrorSource interface {
	Name() string
}

// ErrorFrame notifies upstream that an error occurred downstream. A fatal error
// is unrecoverable and the bot should exit. It is a system frame.
type ErrorFrame struct {
	BaseSystemFrame
	// Error describes the error that occurred.
	Error string
	// Fatal reports whether the error is unrecoverable and requires shutdown.
	Fatal bool
	// Source is the processor that raised the error, if known.
	Source ErrorSource
	// Err is the underlying error, if any.
	Err error
}

// NewErrorFrame builds a non-fatal ErrorFrame describing message.
func NewErrorFrame(message string) *ErrorFrame {
	return &ErrorFrame{BaseSystemFrame: NewBaseSystemFrame("ErrorFrame"), Error: message}
}

// String implements fmt.Stringer.
func (f *ErrorFrame) String() string {
	return fmt.Sprintf("%s(error: %s, fatal: %t)", f.Name(), f.Error, f.Fatal)
}

// FatalErrorFrame notifies upstream that an unrecoverable error occurred and the
// bot should exit immediately. It is an ErrorFrame whose Fatal is always set, so
// a processor can report an unrecoverable failure by type rather than by
// remembering to set the flag.
type FatalErrorFrame struct {
	ErrorFrame
}

// NewFatalErrorFrame builds a FatalErrorFrame describing message.
func NewFatalErrorFrame(message string) *FatalErrorFrame {
	return &FatalErrorFrame{
		ErrorFrame: ErrorFrame{
			BaseSystemFrame: NewBaseSystemFrame("FatalErrorFrame"),
			Error:           message,
			Fatal:           true,
		},
	}
}

// InterruptionFrame interrupts the pipeline — for example when the user starts
// speaking, to cancel in-progress bot output. It can be pushed by any
// processor. It is a system frame.
type InterruptionFrame struct {
	BaseSystemFrame
}

// NewInterruptionFrame builds an InterruptionFrame.
func NewInterruptionFrame() *InterruptionFrame {
	return &InterruptionFrame{BaseSystemFrame: NewBaseSystemFrame("InterruptionFrame")}
}

// UserStartedSpeakingFrame indicates the user turn has started. It is a system
// frame.
type UserStartedSpeakingFrame struct {
	BaseSystemFrame
}

// NewUserStartedSpeakingFrame builds a UserStartedSpeakingFrame.
func NewUserStartedSpeakingFrame() *UserStartedSpeakingFrame {
	return &UserStartedSpeakingFrame{BaseSystemFrame: NewBaseSystemFrame("UserStartedSpeakingFrame")}
}

// UserStoppedSpeakingFrame indicates the user turn has ended. It is a system
// frame.
type UserStoppedSpeakingFrame struct {
	BaseSystemFrame
}

// NewUserStoppedSpeakingFrame builds a UserStoppedSpeakingFrame.
func NewUserStoppedSpeakingFrame() *UserStoppedSpeakingFrame {
	return &UserStoppedSpeakingFrame{BaseSystemFrame: NewBaseSystemFrame("UserStoppedSpeakingFrame")}
}

// BotStartedSpeakingFrame indicates the bot started speaking. It is a system
// frame.
type BotStartedSpeakingFrame struct {
	BaseSystemFrame
}

// NewBotStartedSpeakingFrame builds a BotStartedSpeakingFrame.
func NewBotStartedSpeakingFrame() *BotStartedSpeakingFrame {
	return &BotStartedSpeakingFrame{BaseSystemFrame: NewBaseSystemFrame("BotStartedSpeakingFrame")}
}

// BotStoppedSpeakingFrame indicates the bot stopped speaking. It is a system
// frame.
type BotStoppedSpeakingFrame struct {
	BaseSystemFrame
}

// NewBotStoppedSpeakingFrame builds a BotStoppedSpeakingFrame.
func NewBotStoppedSpeakingFrame() *BotStoppedSpeakingFrame {
	return &BotStoppedSpeakingFrame{BaseSystemFrame: NewBaseSystemFrame("BotStoppedSpeakingFrame")}
}

// ProcessorTarget identifies the processor a frame is addressed to. Like
// ErrorSource it is declared here rather than imported from the processor
// package, so the frames package keeps no dependency on it; a frame processor
// satisfies it by exposing its name.
type ProcessorTarget interface {
	Name() string
}

// FrameProcessorPauseUrgentFrame asks a processor to pause its handling of data
// and control frames as fast as possible. Paused frames stay in the processor's
// queue and are handled once processing resumes. It is a system frame, so it
// overtakes the frames queued ahead of it; use FrameProcessorPauseFrame to pause
// in order instead.
type FrameProcessorPauseUrgentFrame struct {
	BaseSystemFrame
	// Processor is the processor to pause.
	Processor ProcessorTarget
}

// NewFrameProcessorPauseUrgentFrame builds a FrameProcessorPauseUrgentFrame
// addressed to p.
func NewFrameProcessorPauseUrgentFrame(p ProcessorTarget) *FrameProcessorPauseUrgentFrame {
	return &FrameProcessorPauseUrgentFrame{
		BaseSystemFrame: NewBaseSystemFrame("FrameProcessorPauseUrgentFrame"),
		Processor:       p,
	}
}

// FrameProcessorResumeUrgentFrame asks a processor to resume the handling of
// data and control frames it paused, as fast as possible. Queued frames are then
// handled in the order they were received. It is a system frame, so it overtakes
// the frames queued ahead of it; use FrameProcessorResumeFrame to resume in
// order instead.
type FrameProcessorResumeUrgentFrame struct {
	BaseSystemFrame
	// Processor is the processor to resume.
	Processor ProcessorTarget
}

// NewFrameProcessorResumeUrgentFrame builds a FrameProcessorResumeUrgentFrame
// addressed to p.
func NewFrameProcessorResumeUrgentFrame(p ProcessorTarget) *FrameProcessorResumeUrgentFrame {
	return &FrameProcessorResumeUrgentFrame{
		BaseSystemFrame: NewBaseSystemFrame("FrameProcessorResumeUrgentFrame"),
		Processor:       p,
	}
}

// Compile-time interface checks.
var (
	_ SystemFrame = (*StartFrame)(nil)
	_ SystemFrame = (*CancelFrame)(nil)
	_ SystemFrame = (*ErrorFrame)(nil)
	_ SystemFrame = (*FatalErrorFrame)(nil)
	_ SystemFrame = (*InterruptionFrame)(nil)
	_ SystemFrame = (*FrameProcessorPauseUrgentFrame)(nil)
	_ SystemFrame = (*FrameProcessorResumeUrgentFrame)(nil)
	_ SystemFrame = (*UserStartedSpeakingFrame)(nil)
	_ SystemFrame = (*UserStoppedSpeakingFrame)(nil)
	_ SystemFrame = (*BotStartedSpeakingFrame)(nil)
	_ SystemFrame = (*BotStoppedSpeakingFrame)(nil)
)
