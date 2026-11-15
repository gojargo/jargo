// Package rtviclient is the frame serializer for a jargo pipeline acting as an
// RTVI client.
//
// It is the client-side mirror of how a bot emits RTVI. It deserializes the RTVI
// server messages a bot sends (bot-llm-text, user-transcription, and so on) into
// pipeline frames, and serializes a client's outgoing frames (audio, and RTVI
// client messages the caller builds) onto the wire. Pair it with the WebSocket
// client transport (transport/wsclient) to let a jargo pipeline talk to a bot
// that runs an RTVI server transport, such as the eval transport.
//
// It is the inverse of the rtvi.Observer, which turns pipeline frames into
// server messages on the bot side.
//
// It covers the messages a conversation needs: the LLM response lifecycle, the
// TTS's spoken text, transcriptions, speaking and interruption signals, and
// function calls. Metrics and inbound bot audio are not mapped yet.
package rtviclient

import (
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"strconv"
	"sync/atomic"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

// Serializer bridges pipeline frames and RTVI wire messages from the client
// side. It numbers the messages it builds, so one Serializer serves one
// session.
type Serializer struct {
	wsserver.BaseSerializer

	nextID atomic.Uint64
}

// New builds an RTVI client serializer.
func New() *Serializer {
	// RTVI messages must not be ignored: the caller builds RTVI client messages
	// (client-ready, send-text, ...) as transport-message frames for this
	// serializer to emit.
	return &Serializer{KeepRTVIMessages: true}
}

// Setup is a no-op: every audio message carries its own sample rate, so there is
// nothing to take from the pipeline's configuration.
func (*Serializer) Setup(processor.Setup) error { return nil }

func (s *Serializer) messageID() string {
	return strconv.FormatUint(s.nextID.Add(1), 10)
}

// Serialize turns an outgoing client frame into an RTVI wire message, or an
// empty Message to drop the frame.
func (s *Serializer) Serialize(f frames.Frame) (wsserver.Message, error) {
	switch fr := f.(type) {
	case frames.OutputTransportMessage:
		// An RTVI client message the caller built directly (client-ready,
		// send-text, dtmf, eval-*): emit its envelope as it is.
		message := fr.TransportMessage()
		if wsserver.IsRTVIMessage(message) {
			return wsserver.TextMessage(json.Marshal(message))
		}
		return wsserver.Message{}, nil
	case frames.OutputAudioFrame:
		// The pipeline's audio output (the user-side TTS, say) goes to the bot as
		// raw-audio.
		a := fr.AudioData()
		return wsserver.TextMessage(json.Marshal(rtvi.Message{
			Label: rtvi.MessageLabel,
			Type:  rtvi.TypeRawAudio,
			ID:    s.messageID(),
			Data: rtvi.RawAudioData{
				Base64Audio: base64.StdEncoding.EncodeToString(a.Audio),
				SampleRate:  a.SampleRate,
				NumChannels: a.NumChannels,
			},
		}))
	default:
		return wsserver.Message{}, nil
	}
}

// envelope is a received message as far as routing it goes.
type envelope struct {
	Label string          `json:"label"`
	Type  string          `json:"type"`
	Data  json.RawMessage `json:"data"`
}

// payload is the data of a received message. A field the bot left out, or sent
// as something other than what the message defines, reads as its zero value.
type payload map[string]any

func (p payload) str(key string) string {
	s, _ := p[key].(string)
	return s
}

func (p payload) truthy(key string) bool {
	switch v := p[key].(type) {
	case nil:
		return false
	case bool:
		return v
	case string:
		return v != ""
	case float64:
		return v != 0
	case []any:
		return len(v) > 0
	case map[string]any:
		return len(v) > 0
	default:
		return true
	}
}

func (p payload) strings(key string) []string {
	list, _ := p[key].([]any)
	out := make([]string, 0, len(list))
	for _, v := range list {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// arguments are the call's arguments as JSON, an empty object when the bot
// reported none.
func (p payload) arguments(key string) json.RawMessage {
	if !p.truthy(key) {
		return json.RawMessage("{}")
	}
	raw, err := json.Marshal(p[key])
	if err != nil {
		return json.RawMessage("{}")
	}
	return raw
}

// result is a function call's result as the conversation records it: a string
// is the string itself, and anything else is its JSON text.
func (p payload) result(key string) string {
	switch v := p[key].(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		raw, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		return string(raw)
	}
}

// parse reads an RTVI message's type and data, reporting false for a message
// that is not one: not JSON, not an object, or not labeled as RTVI.
func parse(data []byte) (string, payload, bool) {
	if !json.Valid(data) {
		slog.Warn("RTVIClientSerializer: dropping non-JSON message")
		return "", nil, false
	}
	var message envelope
	if err := json.Unmarshal(data, &message); err != nil || message.Label != rtvi.MessageLabel {
		return "", nil, false
	}
	var p payload
	_ = json.Unmarshal(message.Data, &p)
	return message.Type, p, true
}

// Deserialize turns an incoming RTVI server message into a pipeline frame, or
// (nil, nil) when the message is not an RTVI message or has no frame mapping yet
// (handled elsewhere).
func (*Serializer) Deserialize(data []byte) (frames.Frame, error) {
	msgType, p, ok := parse(data)
	if !ok {
		return nil, nil //nolint:nilnil // not an RTVI message
	}

	switch msgType {
	case rtvi.TypeBotLLMStarted:
		return frames.NewLLMFullResponseStartFrame(), nil
	case rtvi.TypeBotLLMText:
		return frames.NewLLMTextFrame(p.str("text")), nil
	case rtvi.TypeBotLLMMarker:
		return frames.NewLLMMarkerResponseFrame(p.str("raw"), p.str("text"), p.str("kind"), p.strings("markers")), nil
	case rtvi.TypeBotLLMStopped:
		return frames.NewLLMFullResponseEndFrame(), nil
	case rtvi.TypeBotTTSText:
		// The message carries only the text; its type (sentence, word) is not on
		// the wire.
		return frames.NewTTSTextFrame(p.str("text"), frames.TextTypeSentence), nil
	case rtvi.TypeBotStartedSpeaking:
		return frames.NewBotStartedSpeakingFrame(), nil
	case rtvi.TypeBotStoppedSpeaking:
		return frames.NewBotStoppedSpeakingFrame(), nil
	case rtvi.TypeBotInterrupted:
		return frames.NewInterruptionFrame(), nil
	case rtvi.TypeLLMFunctionCall:
		// function_name and arguments may be absent at the bot's default report
		// level; "" maps back to "no name reported" downstream.
		return frames.NewFunctionCallInProgressFrame(
			p.str("tool_call_id"), p.str("function_name"), p.arguments("arguments"), false, ""), nil
	case rtvi.TypeLLMFunctionCallStop:
		return functionCallStopped(p), nil
	case rtvi.TypeUserTranscription:
		return userTranscription(p), nil
	default:
		// bot-ready (the handshake) and messages not mapped yet (metrics, bot
		// audio) are handled elsewhere.
		return nil, nil //nolint:nilnil // no frame for this message
	}
}

// functionCallStopped maps an llm-function-call-stopped message. A call ends
// either by being canceled or by returning; the two map to the frames the bot
// itself would have pushed.
func functionCallStopped(p payload) frames.Frame {
	if p.truthy("cancelled") { //nolint:misspell // the protocol spells "cancelled" this way
		return frames.NewFunctionCallCancelFrame(p.str("tool_call_id"), p.str("function_name"))
	}
	return frames.NewFunctionCallResultFrame(
		p.str("tool_call_id"), p.str("function_name"), json.RawMessage("{}"), p.result("result"))
}

// userTranscription maps a user-transcription message, which is final unless
// it says otherwise.
func userTranscription(p payload) frames.Frame {
	text, userID, timestamp := p.str("text"), p.str("user_id"), p.str("timestamp")
	if _, set := p["final"]; !set || p.truthy("final") {
		return frames.NewTranscriptionFrame(text, userID, timestamp)
	}
	return frames.NewInterimTranscriptionFrame(text, userID, timestamp)
}
