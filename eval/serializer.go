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

// evalContextMessage is the client-message `t` carrying the messages a
// scenario's context starts from. It is intercepted here and turned into a
// context update rather than forwarded, which keeps per-eval context seeding out
// of the bot.
const evalContextMessage = "eval-context"

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
	FunctionCallReportLevel map[string]rtvi.FunctionCallReportLevel `json:"function_call_report_level,omitempty"`
	VADUserSpeaking         *bool                                   `json:"vad_user_speaking,omitempty"`
}

// contextPayload is the `d` of an eval-context message: the messages the bot's
// context should start from.
type contextPayload struct {
	Messages []frames.Message `json:"messages"`
}

// serializer is the RTVI WebSocket serializer the eval Handler uses. It behaves
// exactly like the plain RTVI serializer except that it understands the eval's
// own control messages, which keeps eval-specific behavior out of the bot.
type serializer struct {
	*rtviws.Serializer
}

// newSerializer builds the eval serializer.
func newSerializer() *serializer { return &serializer{Serializer: rtviws.New()} }

// Deserialize turns an inbound RTVI message into a frame. The eval's own
// control messages become the frame each asks for; everything else is handed to
// the plain RTVI serializer.
func (s *serializer) Deserialize(data []byte) (frames.Frame, error) {
	if f := evalControlFrame(data); f != nil {
		return f, nil
	}
	return s.Serializer.Deserialize(data)
}

// evalControlFrame returns the frame an eval control message asks for, or nil
// when data is not one of them.
func evalControlFrame(data []byte) frames.Frame {
	in, err := rtvi.ParseIncoming(data)
	if err != nil || in.Label != rtvi.MessageLabel || in.Type != typeClientMessage {
		return nil
	}
	var msg clientMessageData
	if json.Unmarshal(in.Data, &msg) != nil {
		return nil
	}
	switch msg.T {
	case evalConfigureMessage:
		var payload configurePayload
		if json.Unmarshal(msg.D, &payload) != nil {
			return nil
		}
		return rtvi.NewConfigureObserverFrame(payload.FunctionCallReportLevel, payload.VADUserSpeaking)
	case evalContextMessage:
		var payload contextPayload
		if json.Unmarshal(msg.D, &payload) != nil {
			return nil
		}
		// The context is seeded, not run: the scenario's own turns drive the bot.
		return frames.NewLLMMessagesUpdateFrame(payload.Messages)
	default:
		return nil
	}
}

var _ wsserver.Serializer = (*serializer)(nil)

// clientMessage builds the client-message envelope carrying payload under t.
func clientMessage(t string, payload any) rtvi.Message {
	return rtvi.Message{
		Label: rtvi.MessageLabel, Type: typeClientMessage, ID: messageID,
		Data: map[string]any{"t": t, "d": payload},
	}
}

// configureMessage builds the eval-configure client-message that exposes what
// this scenario asserts on, for the duration of this eval only. A nil level
// leaves the bot's own report level alone.
func configureMessage(level *rtvi.FunctionCallReportLevel, vadUserSpeaking *bool) rtvi.Message {
	var levels map[string]rtvi.FunctionCallReportLevel
	if level != nil {
		levels = map[string]rtvi.FunctionCallReportLevel{"*": *level}
	}
	return clientMessage(evalConfigureMessage, configurePayload{
		FunctionCallReportLevel: levels,
		VADUserSpeaking:         vadUserSpeaking,
	})
}

// contextMessage builds the eval-context client-message that seeds the bot's
// context with the messages the scenario starts from.
func contextMessage(messages []frames.Message) rtvi.Message {
	return clientMessage(evalContextMessage, contextPayload{Messages: messages})
}
