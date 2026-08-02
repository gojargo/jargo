package frames

import "fmt"

// InputTransportMessageFrame carries an application message received by a
// transport from the client — for example an RTVI message off a WebRTC data
// channel. It is a system frame so it is handled with priority and in order.
type InputTransportMessageFrame struct {
	BaseSystemFrame
	// Message is the raw message payload as received (typically JSON).
	Message []byte
}

// NewInputTransportMessageFrame builds an InputTransportMessageFrame.
func NewInputTransportMessageFrame(message []byte) *InputTransportMessageFrame {
	return &InputTransportMessageFrame{
		BaseSystemFrame: NewBaseSystemFrame("InputTransportMessageFrame"),
		Message:         message,
	}
}

// String implements fmt.Stringer.
func (f *InputTransportMessageFrame) String() string {
	return fmt.Sprintf("%s(size: %d)", f.Name(), len(f.Message))
}

// OutputTransportReadyFrame reports that the output transport has opened its
// media path and can receive frames. It is pushed upstream once the transport is
// ready, so a producer that must not speak into a connection that is not up yet
// (an avatar or video service, say) can wait for it. It is a control frame.
type OutputTransportReadyFrame struct {
	BaseControlFrame
}

// NewOutputTransportReadyFrame builds an OutputTransportReadyFrame.
func NewOutputTransportReadyFrame() *OutputTransportReadyFrame {
	return &OutputTransportReadyFrame{
		BaseControlFrame: NewBaseControlFrame("OutputTransportReadyFrame"),
	}
}

// InputTransportStartAudioStreamingFrame asks the input transport to start
// streaming audio from its source. It is pushed downstream (by the RTVI
// processor once the client is ready, say) so that starting the stream stays
// frame-based rather than a direct call across processors. It is a control
// frame.
type InputTransportStartAudioStreamingFrame struct {
	BaseControlFrame
}

// NewInputTransportStartAudioStreamingFrame builds an
// InputTransportStartAudioStreamingFrame.
func NewInputTransportStartAudioStreamingFrame() *InputTransportStartAudioStreamingFrame {
	return &InputTransportStartAudioStreamingFrame{
		BaseControlFrame: NewBaseControlFrame("InputTransportStartAudioStreamingFrame"),
	}
}

// OutputTransportMessageFrame carries an application message to send to the
// client over the transport — for example an RTVI message onto a WebRTC data
// channel. Message is serialized by the output transport. It is a data frame, so
// it is delivered in order with the surrounding audio: use it for a message that
// must land in step with what the bot is saying. For a message that must go out
// immediately, ahead of any queued audio, use OutputTransportMessageUrgentFrame.
type OutputTransportMessageFrame struct {
	BaseDataFrame
	// Message is the message payload to send; the transport serializes it.
	Message any
}

// NewOutputTransportMessageFrame builds an OutputTransportMessageFrame.
func NewOutputTransportMessageFrame(message any) *OutputTransportMessageFrame {
	return &OutputTransportMessageFrame{
		BaseDataFrame: NewBaseDataFrame("OutputTransportMessageFrame"),
		Message:       message,
	}
}

// String implements fmt.Stringer.
func (f *OutputTransportMessageFrame) String() string {
	return fmt.Sprintf("%s(message: %v)", f.Name(), f.Message)
}

// OutputTransportMessageUrgentFrame carries an application message that must be
// sent to the client immediately, ahead of any queued audio. It is a system
// frame; prefer OutputTransportMessageFrame when the message should stay ordered
// with the bot's speech.
type OutputTransportMessageUrgentFrame struct {
	BaseSystemFrame
	// Message is the message payload to send; the transport serializes it.
	Message any
}

// NewOutputTransportMessageUrgentFrame builds an OutputTransportMessageUrgentFrame.
func NewOutputTransportMessageUrgentFrame(message any) *OutputTransportMessageUrgentFrame {
	return &OutputTransportMessageUrgentFrame{
		BaseSystemFrame: NewBaseSystemFrame("OutputTransportMessageUrgentFrame"),
		Message:         message,
	}
}

// String implements fmt.Stringer.
func (f *OutputTransportMessageUrgentFrame) String() string {
	return fmt.Sprintf("%s(message: %v)", f.Name(), f.Message)
}

// Compile-time interface checks.
var (
	_ SystemFrame = (*InputTransportMessageFrame)(nil)
	_ DataFrame   = (*OutputTransportMessageFrame)(nil)
	_ SystemFrame = (*OutputTransportMessageUrgentFrame)(nil)
)
