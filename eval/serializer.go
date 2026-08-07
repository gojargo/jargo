package eval

import (
	"encoding/json"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/rtviws"
)

// evalConfigureMessage is the client-message `t` the harness uses to reconfigure
// the bot's RTVI observer for the duration of one eval. It is intercepted here
// and turned into a ConfigureObserverFrame rather than forwarded to the RTVI
// processor.
//
// This interception is the trust boundary for raising the function-call report
// level: only the eval serializer understands the message, so a production RTVI
// transport cannot be elevated by a remote client.
const evalConfigureMessage = "eval-configure"

// typeClientMessage is the RTVI message type carrying an application-defined
// payload, the envelope the eval control messages travel in.
const typeClientMessage = "client-message"

// messageID identifies the harness on every message it sends.
const messageID = "eval"

// clientMessageData is the payload of a client-message: a type tag and its own
// data.
type clientMessageData struct {
	T string          `json:"t"`
	D json.RawMessage `json:"d"`
}

// configurePayload is the `d` of an eval-configure message.
type configurePayload struct {
	FunctionCallReportLevel map[string]rtvi.FunctionCallReportLevel `json:"function_call_report_level"`
}

// serializer is the RTVI WebSocket serializer the eval Handler uses. It behaves
// exactly like the plain RTVI serializer except that it understands the eval's
// own control messages, which keeps eval-specific behavior out of the bot.
type serializer struct {
	*rtviws.Serializer
}

// newSerializer builds the eval serializer.
func newSerializer() *serializer { return &serializer{Serializer: rtviws.New()} }

// Deserialize turns an inbound RTVI message into a frame. An eval-configure
// client-message becomes a ConfigureObserverFrame; everything else is handed to
// the plain RTVI serializer.
func (s *serializer) Deserialize(data []byte) (frames.Frame, error) {
	if f := configureFrame(data); f != nil {
		return f, nil
	}
	return s.Serializer.Deserialize(data)
}

// configureFrame returns the observer reconfiguration an eval-configure message
// asks for, or nil when data is not one.
func configureFrame(data []byte) frames.Frame {
	in, err := rtvi.ParseIncoming(data)
	if err != nil || in.Label != rtvi.MessageLabel || in.Type != typeClientMessage {
		return nil
	}
	var msg clientMessageData
	if json.Unmarshal(in.Data, &msg) != nil || msg.T != evalConfigureMessage {
		return nil
	}
	var payload configurePayload
	if json.Unmarshal(msg.D, &payload) != nil {
		return nil
	}
	return rtvi.NewConfigureObserverFrame(payload.FunctionCallReportLevel)
}

var _ wsserver.Serializer = (*serializer)(nil)

// configureMessage builds the eval-configure client-message that raises the
// bot's function-call report level to level for every function, for the
// duration of this eval only.
func configureMessage(level rtvi.FunctionCallReportLevel) rtvi.Message {
	return rtvi.Message{
		Label: rtvi.MessageLabel, Type: typeClientMessage, ID: messageID,
		Data: map[string]any{
			"t": evalConfigureMessage,
			"d": configurePayload{
				FunctionCallReportLevel: map[string]rtvi.FunctionCallReportLevel{"*": level},
			},
		},
	}
}
