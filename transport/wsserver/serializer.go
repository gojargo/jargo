package wsserver

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
)

// Message is one wire message a Serializer produced, and how the WebSocket
// frames it.
//
// Which framing to use is the serializer's to decide rather than the
// transport's, because one wire can carry both: a provider that takes raw PCM
// as binary still takes its control messages as JSON text.
type Message struct {
	// Data is the encoded message. Empty means the serializer had nothing to
	// send for the frame, which is not an error.
	Data []byte
	// Binary frames the message as binary rather than text.
	Binary bool
}

// Empty reports that there is nothing to write.
func (m Message) Empty() bool { return len(m.Data) == 0 }

// TextMessage builds a text wire message. It takes an error alongside the data
// so that it can wrap an encoder call whole:
//
//	return wsserver.TextMessage(json.Marshal(answer))
func TextMessage(data []byte, err error) (Message, error) {
	if err != nil {
		return Message{}, err
	}
	return Message{Data: data}, nil
}

// BinaryMessage builds a binary wire message. See TextMessage.
func BinaryMessage(data []byte, err error) (Message, error) {
	if err != nil {
		return Message{}, err
	}
	return Message{Data: data, Binary: true}, nil
}

// BaseSerializer holds the behavior every Serializer shares. Embed it to get
// it, and pass its fields when the defaults are not what the wire wants.
type BaseSerializer struct {
	// KeepRTVIMessages writes RTVI protocol messages to the wire instead of
	// dropping them.
	//
	// It is off by default, because the usual wire here is a telephony
	// provider's media stream: RTVI means nothing to a provider expecting its
	// own control messages, and a pipeline with an RTVI processor in it would
	// otherwise write the whole protocol onto the call. A serializer for a wire
	// that does carry the protocol, which is what a browser client connects
	// over, turns it on.
	KeepRTVIMessages bool
}

// ShouldIgnoreFrame reports whether f carries an application message this wire
// does not carry. A serializer calls it before encoding a message frame.
func (b *BaseSerializer) ShouldIgnoreFrame(f frames.Frame) bool {
	if b.KeepRTVIMessages {
		return false
	}
	m, ok := f.(frames.OutputTransportMessage)
	return ok && IsRTVIMessage(m.TransportMessage())
}

// IsRTVIMessage reports whether the message is one of the RTVI protocol's, which
// every one of them says of itself in its label.
func IsRTVIMessage(message any) bool {
	switch m := message.(type) {
	case rtvi.Message:
		return m.Label == rtvi.MessageLabel
	case *rtvi.Message:
		return m != nil && m.Label == rtvi.MessageLabel
	case map[string]any:
		// Built by hand rather than by the rtvi package, which a caller speaking
		// the protocol itself would produce.
		label, _ := m["label"].(string)
		return label == rtvi.MessageLabel
	default:
		return false
	}
}
