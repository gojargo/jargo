package eval

import (
	"encoding/json"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// The bot's end of the eval wire, the serializer that understands the eval's
// own messages, lives in transport/wsserver/evalws. This file builds the
// harness's end of those messages.

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
	BotLLMMarker            *bool                                   `json:"llm_markers,omitempty"`
}

// contextPayload is the `d` of an eval-context message: the messages the bot's
// context should start from.
type contextPayload struct {
	Messages []frames.Message `json:"messages"`
}

// clientMessage builds the client-message envelope carrying payload under t.
func clientMessage(t string, payload any) rtvi.Message {
	return rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientMessage, ID: messageID,
		Data: map[string]any{"t": t, "d": payload},
	}
}

// configureMessage builds the eval-configure client-message that exposes what
// this scenario asserts on, for the duration of this eval only. A nil level
// leaves the bot's own report level alone.
func configureMessage(
	level *rtvi.FunctionCallReportLevel, vadUserSpeaking, botLLMMarker *bool,
) rtvi.Message {
	var levels map[string]rtvi.FunctionCallReportLevel
	if level != nil {
		levels = map[string]rtvi.FunctionCallReportLevel{"*": *level}
	}
	return clientMessage(evalws.ConfigureMessageType, configurePayload{
		FunctionCallReportLevel: levels,
		VADUserSpeaking:         vadUserSpeaking,
		BotLLMMarker:            botLLMMarker,
	})
}

// contextMessage builds the eval-context client-message that seeds the bot's
// context with the messages the scenario starts from.
func contextMessage(messages []frames.Message) rtvi.Message {
	return clientMessage(evalws.ContextMessageType, contextPayload{Messages: messages})
}
