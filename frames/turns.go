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
	// Timestamp is the wall-clock time at which the VAD made its determination.
	// Subtracting StartSecs from it gives the moment the speech itself began,
	// which is what a measurement of the whole utterance is anchored to.
	Timestamp time.Time
}

// NewVADUserStartedSpeakingFrame builds a VADUserStartedSpeakingFrame.
func NewVADUserStartedSpeakingFrame(
	startSecs float64, timestamp time.Time,
) *VADUserStartedSpeakingFrame {
	return &VADUserStartedSpeakingFrame{
		BaseSystemFrame: NewBaseSystemFrame("VADUserStartedSpeakingFrame"),
		StartSecs:       startSecs,
		Timestamp:       timestamp,
	}
}

// SpeechStart is the moment the speech this frame reports actually began, which
// is earlier than the determination by the confirmation delay.
func (f *VADUserStartedSpeakingFrame) SpeechStart() time.Time {
	return f.Timestamp.Add(-time.Duration(f.StartSecs * float64(time.Second)))
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
	// Timestamp is the wall-clock time at which the VAD made its determination.
	// Subtracting StopSecs from it gives the moment the speech itself ended,
	// which is what a deadline measured from the end of speech is anchored to.
	// The zero value means unset.
	Timestamp time.Time
}

// NewVADUserStoppedSpeakingFrame builds a VADUserStoppedSpeakingFrame.
func NewVADUserStoppedSpeakingFrame(stopSecs float64, timestamp time.Time) *VADUserStoppedSpeakingFrame {
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

// LLMConfigureOutputFrame configures how an LLM service produces output. It
// tells the service to stamp the tokens it emits so a TTS service downstream
// passes them through instead of speaking them: the reply is added to the
// conversation but never said out loud. It is a data frame.
type LLMConfigureOutputFrame struct {
	BaseDataFrame
	// SkipTTS reports whether the tokens the LLM emits should skip the TTS
	// service, if the pipeline has one.
	SkipTTS bool
}

// NewLLMConfigureOutputFrame builds an LLMConfigureOutputFrame.
func NewLLMConfigureOutputFrame(skipTTS bool) *LLMConfigureOutputFrame {
	return &LLMConfigureOutputFrame{
		BaseDataFrame: NewBaseDataFrame("LLMConfigureOutputFrame"),
		SkipTTS:       skipTTS,
	}
}

// LLMMessagesAppendFrame asks the context aggregator to append messages to the
// LLM context, in contrast to LLMMessagesUpdateFrame which replaces them. The
// turn-completion re-prompt and a conversation flow entering a node both use it.
// It is a data frame, so the messages are ordered against the surrounding
// conversation.
type LLMMessagesAppendFrame struct {
	BaseDataFrame
	// Messages are the messages to append.
	Messages []Message
	// RunLLM reports whether the LLM should run on the updated context.
	RunLLM bool
}

// NewLLMMessagesAppendFrame builds an LLMMessagesAppendFrame.
func NewLLMMessagesAppendFrame(messages []Message) *LLMMessagesAppendFrame {
	return &LLMMessagesAppendFrame{
		BaseDataFrame: NewBaseDataFrame("LLMMessagesAppendFrame"),
		Messages:      messages,
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
	_ DataFrame    = (*LLMConfigureOutputFrame)(nil)
	_ DataFrame    = (*LLMMessagesAppendFrame)(nil)
)
