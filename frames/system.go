package frames

import (
	"fmt"

	"github.com/gojargo/jargo/utils/errors"
)

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

	// The fields below carry the pipeline's configuration, which a processor now
	// reads from its [processor.Setup] instead: it knows its configuration from
	// the moment it is set up, rather than when this frame reaches it, which is
	// what lets it connect while it is being set up. The worker still fills them
	// in so a processor that reads one gets the configured value rather than the
	// field's default.

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
	// Category is what kind of failure this was: rejected credentials, an
	// unreachable provider, a malformed request and so on. The zero value means
	// nobody has said yet, which invites the category to be worked out from Err;
	// set it to errors.Unknown to report a failure whose cause cannot be
	// attributed. It is always settled by the time the frame travels.
	Category errors.Category
}

// ErrorInfo implements [ErrorReport].
func (f *ErrorFrame) ErrorInfo() *ErrorFrame { return f }

// ErrorReport is implemented by every frame that reports an error: [ErrorFrame]
// and the frames embedding it, such as [FatalErrorFrame]. Match on it rather
// than on [ErrorFrame] itself, which a frame that embeds it does not satisfy, so
// an error reported by type is not missed.
type ErrorReport interface {
	Frame
	// ErrorInfo returns the error the frame carries.
	ErrorInfo() *ErrorFrame
}

// NewErrorFrame builds a non-fatal ErrorFrame describing message.
func NewErrorFrame(message string) *ErrorFrame {
	return &ErrorFrame{BaseSystemFrame: NewBaseSystemFrame("ErrorFrame"), Error: message}
}

// String implements fmt.Stringer. An unset or unknown category is left out:
// neither says anything about the failure that the message does not.
func (f *ErrorFrame) String() string {
	category := ""
	if f.Category != errors.Unset && f.Category != errors.Unknown {
		category = fmt.Sprintf(", category: %s", f.Category)
	}
	return fmt.Sprintf("%s(error: %s, fatal: %t%s)", f.Name(), f.Error, f.Fatal, category)
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
	_ ErrorReport = (*ErrorFrame)(nil)
	_ SystemFrame = (*FatalErrorFrame)(nil)
	_ ErrorReport = (*FatalErrorFrame)(nil)
	_ SystemFrame = (*InterruptionFrame)(nil)
	_ SystemFrame = (*FrameProcessorPauseUrgentFrame)(nil)
	_ SystemFrame = (*FrameProcessorResumeUrgentFrame)(nil)
	_ SystemFrame = (*UserStartedSpeakingFrame)(nil)
	_ SystemFrame = (*UserStoppedSpeakingFrame)(nil)
	_ SystemFrame = (*BotStartedSpeakingFrame)(nil)
	_ SystemFrame = (*BotStoppedSpeakingFrame)(nil)
)

// ProposedUserStartedSpeakingFrame proposes that the user's turn has started.
//
// It is emitted by a component with turn detection of its own, typically an STT
// or realtime LLM service whose provider reports speech boundaries. It is a
// proposal rather than a decision: an external turn-start strategy resolves it
// into a UserStartedSpeakingFrame and broadcasts the interruption. It is a
// system frame.
type ProposedUserStartedSpeakingFrame struct {
	BaseSystemFrame
}

// NewProposedUserStartedSpeakingFrame builds a ProposedUserStartedSpeakingFrame.
func NewProposedUserStartedSpeakingFrame() *ProposedUserStartedSpeakingFrame {
	return &ProposedUserStartedSpeakingFrame{
		BaseSystemFrame: NewBaseSystemFrame("ProposedUserStartedSpeakingFrame"),
	}
}

// ProposedUserStoppedSpeakingFrame proposes that the user's turn has ended. Like
// its counterpart it is a proposal an external turn-stop strategy resolves into
// a UserStoppedSpeakingFrame. It is a system frame.
type ProposedUserStoppedSpeakingFrame struct {
	BaseSystemFrame
}

// NewProposedUserStoppedSpeakingFrame builds a ProposedUserStoppedSpeakingFrame.
func NewProposedUserStoppedSpeakingFrame() *ProposedUserStoppedSpeakingFrame {
	return &ProposedUserStoppedSpeakingFrame{
		BaseSystemFrame: NewBaseSystemFrame("ProposedUserStoppedSpeakingFrame"),
	}
}

// STTMuteFrame mutes or unmutes the transcription service, so audio reaching it
// is left untranscribed for as long as the mute stands. It is a system frame, so
// it takes effect ahead of the audio queued behind it.
type STTMuteFrame struct {
	BaseSystemFrame
	// Mute reports whether the service should be muted.
	Mute bool
}

// NewSTTMuteFrame builds an STTMuteFrame.
func NewSTTMuteFrame(mute bool) *STTMuteFrame {
	return &STTMuteFrame{
		BaseSystemFrame: NewBaseSystemFrame("STTMuteFrame"),
		Mute:            mute,
	}
}

// String implements fmt.Stringer.
func (f *STTMuteFrame) String() string {
	return fmt.Sprintf("%s(mute: %t)", f.Name(), f.Mute)
}

// ClientConnectedFrame reports that a client has connected to the transport. The
// input transport pushes it downstream when a participant connects, and an
// observer measuring how long the transport took to be ready reads it. It is a
// system frame.
type ClientConnectedFrame struct {
	BaseSystemFrame
}

// NewClientConnectedFrame builds a ClientConnectedFrame.
func NewClientConnectedFrame() *ClientConnectedFrame {
	return &ClientConnectedFrame{BaseSystemFrame: NewBaseSystemFrame("ClientConnectedFrame")}
}

// BotConnectedFrame reports that the bot has connected to the transport service.
// A transport that joins a room of its own pushes it downstream once the join
// succeeds; a transport that is dialed rather than joining emits nothing. It is
// a system frame.
type BotConnectedFrame struct {
	BaseSystemFrame
}

// NewBotConnectedFrame builds a BotConnectedFrame.
func NewBotConnectedFrame() *BotConnectedFrame {
	return &BotConnectedFrame{BaseSystemFrame: NewBaseSystemFrame("BotConnectedFrame")}
}
