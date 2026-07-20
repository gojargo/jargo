// Package rtviws is the wsserver.Serializer that carries the RTVI protocol over
// a plain WebSocket, so an RTVI client can drive a jargo bot without WebRTC.
// Inbound RTVI messages are handed to the pipeline's RTVI processor; outbound
// RTVI server messages reach the socket through the transport's own message
// path (an OutputTransportMessageFrame), so the serializer only bridges the
// inbound direction.
//
// This carries the RTVI control, event and text channel only — bot audio is not
// streamed over the socket, so a client on this transport is text-first. Pair it
// with a pipeline that includes an rtvi.Processor.
package rtviws

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

// Serializer bridges RTVI JSON messages and pipeline frames over a WebSocket.
// It holds no per-session state, so a single value may serve any session.
type Serializer struct{}

// New builds an RTVI WebSocket serializer.
func New() *Serializer { return &Serializer{} }

// Setup is a no-op: the RTVI channel carries no audio, so there is nothing to
// configure from the StartFrame.
func (*Serializer) Setup(*frames.StartFrame) error { return nil }

// Serialize drops outbound frames. RTVI server messages reach the socket through
// the transport's own OutputTransportMessageFrame path rather than the
// serializer, and bot audio is not streamed over this channel.
func (*Serializer) Serialize(frames.Frame) ([]byte, error) { return nil, nil }

// Deserialize wraps an inbound RTVI message in an InputTransportMessageFrame so
// the pipeline's RTVI processor parses and routes it (the handshake, send-text,
// and so on). Payloads that are not RTVI messages are ignored.
func (*Serializer) Deserialize(data []byte) (frames.Frame, error) {
	in, err := rtvi.ParseIncoming(data)
	if err != nil || in.Label != rtvi.MessageLabel {
		// Malformed or non-RTVI payloads carry no frame and are dropped, per the
		// wsserver.Serializer contract — not an error.
		return nil, nil //nolint:nilnil,nilerr // dropping a non-RTVI message is intentional
	}
	// The transport may reuse its read buffer, and the frame outlives this call,
	// so hand the RTVI processor its own copy.
	msg := make([]byte, len(data))
	copy(msg, data)
	return frames.NewInputTransportMessageFrame(msg), nil
}
