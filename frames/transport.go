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

// OutputTransportMessageFrame carries an application message to send to the
// client over the transport — for example an RTVI message onto a WebRTC data
// channel. Message is serialized by the output transport. It is a system frame.
type OutputTransportMessageFrame struct {
	BaseSystemFrame
	// Message is the message payload to send; the transport serializes it.
	Message any
}

// NewOutputTransportMessageFrame builds an OutputTransportMessageFrame.
func NewOutputTransportMessageFrame(message any) *OutputTransportMessageFrame {
	return &OutputTransportMessageFrame{
		BaseSystemFrame: NewBaseSystemFrame("OutputTransportMessageFrame"),
		Message:         message,
	}
}

// Compile-time interface checks.
var (
	_ SystemFrame = (*InputTransportMessageFrame)(nil)
	_ SystemFrame = (*OutputTransportMessageFrame)(nil)
)
