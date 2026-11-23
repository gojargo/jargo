// Package rtviclient is the frame serializer for a jargo pipeline acting as an
// RTVI client.
//
// It is the client-side mirror of how a bot emits RTVI. It hands the RTVI
// server messages a bot sends to the pipeline, and serializes a client's
// outgoing frames (audio, and RTVI client messages the caller builds) onto the
// wire. Pair it with the WebSocket client transport (transport/wsclient) to let
// a jargo pipeline talk to a bot that runs an RTVI server transport, such as
// the eval transport.
//
// It is the inverse of the rtvi.Observer, which turns pipeline frames into
// server messages on the bot side.
//
// Every message from the bot stays a raw message, a
// frames.InputTransportMessageFrame. The bot's messages are about the bot (its
// speaking, its LLM's text, its function calls and so on) or about this
// pipeline as the bot sees it (its user-transcription), and this pipeline's own
// frames would pass them off as its own: a BotStartedSpeakingFrame here means
// this pipeline's output started playing. Kept as one kind of frame, they also
// reach the pipeline in the order the bot sent them.
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

// Deserialize hands an incoming RTVI server message on as the raw message it
// is: a frames.InputTransportMessageFrame carrying data. It returns (nil, nil)
// when the message is not an RTVI message.
func (*Serializer) Deserialize(data []byte) (frames.Frame, error) {
	if !json.Valid(data) {
		slog.Warn("RTVIClientSerializer: dropping non-JSON message")
		return nil, nil //nolint:nilnil // not an RTVI message
	}
	var message struct {
		Label string `json:"label"`
	}
	// JSON that is not an object leaves the label empty.
	_ = json.Unmarshal(data, &message)
	if message.Label != rtvi.MessageLabel {
		return nil, nil //nolint:nilnil // not an RTVI message
	}
	return frames.NewInputTransportMessageFrame(data), nil
}
