package rtvi

import (
	"encoding/json"
	"fmt"

	"github.com/gojargo/jargo/frames"
)

// ConfigureObserverFrame reconfigures a running Observer. It lets a trusted,
// server-side source adjust what the observer exposes at runtime without baking
// the setting into the bot, where it would apply to every client. Only the
// fields that are set are applied; a nil field leaves the current configuration
// unchanged.
//
// The eval harness pushes this, through the eval-only serializer, to raise the
// function-call report level for the calls a scenario asserts on, so production
// bots can keep the secure default.
type ConfigureObserverFrame struct {
	frames.BaseSystemFrame
	// FunctionCallReportLevel is the per-function report-level map to apply, or
	// nil to leave the observer's current map unchanged.
	FunctionCallReportLevel map[string]FunctionCallReportLevel
	// VADUserSpeakingEnabled turns the raw VAD speaking events on or off, or is
	// nil to leave the observer's current setting unchanged.
	VADUserSpeakingEnabled *bool
}

// NewConfigureObserverFrame builds a ConfigureObserverFrame.
func NewConfigureObserverFrame(
	level map[string]FunctionCallReportLevel, vadUserSpeaking *bool,
) *ConfigureObserverFrame {
	return &ConfigureObserverFrame{
		BaseSystemFrame:         frames.NewBaseSystemFrame("RTVIConfigureObserverFrame"),
		FunctionCallReportLevel: level,
		VADUserSpeakingEnabled:  vadUserSpeaking,
	}
}

// String implements fmt.Stringer.
func (f *ConfigureObserverFrame) String() string {
	return fmt.Sprintf("%s(function_call_report_level: %v, vad_user_speaking: %v)",
		f.Name(), f.FunctionCallReportLevel, f.VADUserSpeakingEnabled)
}

// ClientMessageFrame carries a client-message: something the client asked the
// bot that the protocol has no message of its own for. It travels downstream
// from the RTVI processor, so a processor placed anywhere in the pipeline can
// answer it by pushing a ServerResponseFrame back.
//
// A client message expects an answer. Answer it with a ServerResponseFrame
// naming this frame, so the client can pair the two.
type ClientMessageFrame struct {
	frames.BaseSystemFrame
	// MsgID is the client's id for the request, echoed in the answer.
	MsgID string
	// Type is the client's own message type, opaque to the protocol.
	Type string
	// Data is whatever the client sent with it, left as raw JSON so the
	// processor answering it decodes the shape it expects.
	Data json.RawMessage
}

// NewClientMessageFrame builds a ClientMessageFrame.
func NewClientMessageFrame(msgID, msgType string, data json.RawMessage) *ClientMessageFrame {
	return &ClientMessageFrame{
		BaseSystemFrame: frames.NewBaseSystemFrame("RTVIClientMessageFrame"),
		MsgID:           msgID,
		Type:            msgType,
		Data:            data,
	}
}

// String implements fmt.Stringer.
func (f *ClientMessageFrame) String() string {
	return fmt.Sprintf("%s(id: %s, type: %s)", f.Name(), f.MsgID, f.Type)
}

// ServerMessageFrame carries an unprompted message to the client: whatever the
// bot wants to tell it that the protocol has no message for. Push one from
// anywhere in the pipeline and the Observer sends it.
type ServerMessageFrame struct {
	frames.BaseSystemFrame
	// Data is the message, serialized as the message's data.
	Data any
}

// NewServerMessageFrame builds a ServerMessageFrame.
func NewServerMessageFrame(data any) *ServerMessageFrame {
	return &ServerMessageFrame{
		BaseSystemFrame: frames.NewBaseSystemFrame("RTVIServerMessageFrame"),
		Data:            data,
	}
}

// String implements fmt.Stringer.
func (f *ServerMessageFrame) String() string {
	return fmt.Sprintf("%s(data: %v)", f.Name(), f.Data)
}

// ServerResponseFrame answers one ClientMessageFrame. It names the frame it
// answers, so the client can pair the answer with the request it made.
//
// Setting Error refuses the request instead: the client receives an
// error-response rather than a server-response, so a request it made never
// simply goes unanswered.
type ServerResponseFrame struct {
	frames.BaseSystemFrame
	// ClientMsg is the request being answered.
	ClientMsg *ClientMessageFrame
	// Data is the answer, serialized as the response's data.
	Data any
	// Error refuses the request, and is the reason given to the client. When it
	// is set, Data is not sent.
	Error string
}

// NewServerResponseFrame builds the answer to msg.
func NewServerResponseFrame(msg *ClientMessageFrame, data any) *ServerResponseFrame {
	return &ServerResponseFrame{
		BaseSystemFrame: frames.NewBaseSystemFrame("RTVIServerResponseFrame"),
		ClientMsg:       msg,
		Data:            data,
	}
}

// NewServerErrorResponseFrame builds the refusal of msg, giving reason.
func NewServerErrorResponseFrame(msg *ClientMessageFrame, reason string) *ServerResponseFrame {
	return &ServerResponseFrame{
		BaseSystemFrame: frames.NewBaseSystemFrame("RTVIServerResponseFrame"),
		ClientMsg:       msg,
		Error:           reason,
	}
}

// String implements fmt.Stringer.
func (f *ServerResponseFrame) String() string {
	if f.Error != "" {
		return fmt.Sprintf("%s(error: %s)", f.Name(), f.Error)
	}
	return fmt.Sprintf("%s(data: %v)", f.Name(), f.Data)
}
