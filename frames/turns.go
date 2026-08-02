package frames

import (
	"fmt"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
)

// VADUserStartedSpeakingFrame reports that a voice-activity detector heard the
// user start speaking. It is the raw VAD signal the turn subsystem consumes to
// decide a user turn; it is distinct from UserStartedSpeakingFrame, which is the
// turn decision. It is a system frame.
type VADUserStartedSpeakingFrame struct {
	BaseSystemFrame
	// StartSecs is the VAD's confirmation delay (how long speech persisted
	// before onset was confirmed), in seconds.
	StartSecs float64
}

// NewVADUserStartedSpeakingFrame builds a VADUserStartedSpeakingFrame.
func NewVADUserStartedSpeakingFrame(startSecs float64) *VADUserStartedSpeakingFrame {
	return &VADUserStartedSpeakingFrame{
		BaseSystemFrame: NewBaseSystemFrame("VADUserStartedSpeakingFrame"),
		StartSecs:       startSecs,
	}
}

// String implements fmt.Stringer.
func (f *VADUserStartedSpeakingFrame) String() string {
	return fmt.Sprintf("%s(start_secs: %.3f)", f.Name(), f.StartSecs)
}

// VADUserStoppedSpeakingFrame reports that the VAD heard the user stop speaking.
// It is a system frame.
type VADUserStoppedSpeakingFrame struct {
	BaseSystemFrame
	// StopSecs is the silence duration the VAD required before confirming the
	// stop, in seconds.
	StopSecs float64
	// Timestamp is when the stop was detected, as an RFC3339 string; "" when
	// unset.
	Timestamp string
}

// NewVADUserStoppedSpeakingFrame builds a VADUserStoppedSpeakingFrame.
func NewVADUserStoppedSpeakingFrame(stopSecs float64, timestamp string) *VADUserStoppedSpeakingFrame {
	return &VADUserStoppedSpeakingFrame{
		BaseSystemFrame: NewBaseSystemFrame("VADUserStoppedSpeakingFrame"),
		StopSecs:        stopSecs,
		Timestamp:       timestamp,
	}
}

// String implements fmt.Stringer.
func (f *VADUserStoppedSpeakingFrame) String() string {
	return fmt.Sprintf("%s(stop_secs: %.3f)", f.Name(), f.StopSecs)
}

// UserSpeakingFrame is emitted periodically while the user is speaking, a
// keepalive that lets strategies and idle logic know audio is still arriving. It
// is a system frame.
type UserSpeakingFrame struct {
	BaseSystemFrame
}

// NewUserSpeakingFrame builds a UserSpeakingFrame.
func NewUserSpeakingFrame() *UserSpeakingFrame {
	return &UserSpeakingFrame{BaseSystemFrame: NewBaseSystemFrame("UserSpeakingFrame")}
}

// BotSpeakingFrame is emitted periodically while the bot is speaking. It is a
// system frame.
type BotSpeakingFrame struct {
	BaseSystemFrame
}

// NewBotSpeakingFrame builds a BotSpeakingFrame.
func NewBotSpeakingFrame() *BotSpeakingFrame {
	return &BotSpeakingFrame{BaseSystemFrame: NewBaseSystemFrame("BotSpeakingFrame")}
}

// UserMuteStartedFrame reports that user input is now being suppressed (a mute
// strategy engaged). It is a system frame.
type UserMuteStartedFrame struct {
	BaseSystemFrame
}

// NewUserMuteStartedFrame builds a UserMuteStartedFrame.
func NewUserMuteStartedFrame() *UserMuteStartedFrame {
	return &UserMuteStartedFrame{BaseSystemFrame: NewBaseSystemFrame("UserMuteStartedFrame")}
}

// UserMuteStoppedFrame reports that user input is no longer suppressed. It is a
// system frame.
type UserMuteStoppedFrame struct {
	BaseSystemFrame
}

// NewUserMuteStoppedFrame builds a UserMuteStoppedFrame.
func NewUserMuteStoppedFrame() *UserMuteStoppedFrame {
	return &UserMuteStoppedFrame{BaseSystemFrame: NewBaseSystemFrame("UserMuteStoppedFrame")}
}

// UserIdleTimeoutUpdateFrame updates the user-idle timeout at runtime. A value
// <= 0 disables idle detection. It is a system frame.
type UserIdleTimeoutUpdateFrame struct {
	BaseSystemFrame
	// Timeout is the new idle timeout.
	Timeout time.Duration
}

// NewUserIdleTimeoutUpdateFrame builds a UserIdleTimeoutUpdateFrame.
func NewUserIdleTimeoutUpdateFrame(timeout time.Duration) *UserIdleTimeoutUpdateFrame {
	return &UserIdleTimeoutUpdateFrame{
		BaseSystemFrame: NewBaseSystemFrame("UserIdleTimeoutUpdateFrame"),
		Timeout:         timeout,
	}
}

// String implements fmt.Stringer.
func (f *UserIdleTimeoutUpdateFrame) String() string {
	return fmt.Sprintf("%s(timeout: %s)", f.Name(), f.Timeout)
}

// UserTurnInferenceCompletedFrame signals that an external judge (an LLM
// completion gate, an EOT classifier) decided the user's turn is semantically
// complete. A turn-stop strategy waits for it to finalize the turn. It is a
// control frame.
type UserTurnInferenceCompletedFrame struct {
	BaseControlFrame
}

// NewUserTurnInferenceCompletedFrame builds a UserTurnInferenceCompletedFrame.
func NewUserTurnInferenceCompletedFrame() *UserTurnInferenceCompletedFrame {
	return &UserTurnInferenceCompletedFrame{
		BaseControlFrame: NewBaseControlFrame("UserTurnInferenceCompletedFrame"),
	}
}

// SpeechControlParamsFrame reports the parameters governing speech detection and
// end-of-turn analysis, so a processor downstream can size its own behavior to
// them (speech recognition matching its endpointing to the detector's, say) and
// clients and observers can mirror them. Either set is nil when only the other
// is being reported. It is a system frame.
type SpeechControlParamsFrame struct {
	BaseSystemFrame
	// VADParams are the voice-activity parameters in force, or nil.
	VADParams *vad.Params
	// TurnParams are the end-of-turn parameters in force, or nil.
	TurnParams *turn.Params
}

// NewSpeechControlParamsFrame builds a SpeechControlParamsFrame. Either set may
// be nil.
func NewSpeechControlParamsFrame(vadParams *vad.Params, turnParams *turn.Params) *SpeechControlParamsFrame {
	return &SpeechControlParamsFrame{
		BaseSystemFrame: NewBaseSystemFrame("SpeechControlParamsFrame"),
		VADParams:       vadParams,
		TurnParams:      turnParams,
	}
}

// VADParamsUpdateFrame asks the voice-activity detector to adopt new parameters.
// It is pushed upstream (by the RTVI processor acting on a client request, say)
// and takes effect from the next chunk analyzed. It is a control frame.
type VADParamsUpdateFrame struct {
	BaseControlFrame
	// Params are the detection parameters to adopt.
	Params vad.Params
}

// NewVADParamsUpdateFrame builds a VADParamsUpdateFrame.
func NewVADParamsUpdateFrame(params vad.Params) *VADParamsUpdateFrame {
	return &VADParamsUpdateFrame{
		BaseControlFrame: NewBaseControlFrame("VADParamsUpdateFrame"),
		Params:           params,
	}
}

// LLMMarkerFrame carries a turn-completion marker the LLM emitted (e.g. "✓"). It
// is informational — downstream TTS ignores it — and lets observers see the
// model's completeness verdict. It is a data frame.
type LLMMarkerFrame struct {
	BaseDataFrame
	// Marker is the marker text.
	Marker string
}

// NewLLMMarkerFrame builds an LLMMarkerFrame.
func NewLLMMarkerFrame(marker string) *LLMMarkerFrame {
	return &LLMMarkerFrame{BaseDataFrame: NewBaseDataFrame("LLMMarkerFrame"), Marker: marker}
}

// LLMMessagesAppendFrame asks the context aggregator to append messages to the
// LLM context (used by the turn-completion re-prompt). It is a control frame.
type LLMMessagesAppendFrame struct {
	BaseControlFrame
	// Messages are the messages to append.
	Messages []Message
}

// NewLLMMessagesAppendFrame builds an LLMMessagesAppendFrame.
func NewLLMMessagesAppendFrame(messages []Message) *LLMMessagesAppendFrame {
	return &LLMMessagesAppendFrame{
		BaseControlFrame: NewBaseControlFrame("LLMMessagesAppendFrame"),
		Messages:         messages,
	}
}

// Compile-time interface checks.
var (
	_ SystemFrame  = (*VADUserStartedSpeakingFrame)(nil)
	_ SystemFrame  = (*VADUserStoppedSpeakingFrame)(nil)
	_ SystemFrame  = (*UserSpeakingFrame)(nil)
	_ SystemFrame  = (*BotSpeakingFrame)(nil)
	_ SystemFrame  = (*UserMuteStartedFrame)(nil)
	_ SystemFrame  = (*UserMuteStoppedFrame)(nil)
	_ SystemFrame  = (*UserIdleTimeoutUpdateFrame)(nil)
	_ SystemFrame  = (*SpeechControlParamsFrame)(nil)
	_ ControlFrame = (*VADParamsUpdateFrame)(nil)
	_ ControlFrame = (*UserTurnInferenceCompletedFrame)(nil)
	_ DataFrame    = (*LLMMarkerFrame)(nil)
	_ ControlFrame = (*LLMMessagesAppendFrame)(nil)
)
