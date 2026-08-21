// Package rtviws is the wsserver.Serializer that carries the RTVI protocol over
// a plain WebSocket, so an RTVI client can drive a jargo bot without WebRTC.
// Inbound RTVI messages are handed to the pipeline's RTVI processor; a client's
// microphone audio arrives as raw-audio messages and enters the pipeline as
// InputAudioRawFrames (VAD, turn detection and STT then see it as a live mic).
// Outbound RTVI server messages reach the socket through the transport's own
// message path (an OutputTransportMessageUrgentFrame).
//
// Bot audio is not yet streamed back over the socket, so a client on this
// transport hears events and text, not synthesized speech. Pair it with a
// pipeline that includes an rtvi.Processor.
package rtviws

import (
	"encoding/base64"
	"encoding/json"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
)

// TypeRawAudio is the RTVI message type carrying a chunk of microphone PCM from
// the client to the bot.
const TypeRawAudio = "raw-audio"

// RawAudioData is the payload of a raw-audio message: base64 16-bit mono PCM
// with its sample rate.
type RawAudioData struct {
	Audio       string `json:"audio"`
	SampleRate  int    `json:"sample_rate"`
	NumChannels int    `json:"num_channels"`
}

// RawAudio builds a raw-audio message carrying pcm (16-bit PCM) at sampleRate.
func RawAudio(pcm []byte, sampleRate, numChannels int) rtvi.Message {
	return rtvi.Message{
		Label: rtvi.MessageLabel,
		Type:  TypeRawAudio,
		Data: RawAudioData{
			Audio:       base64.StdEncoding.EncodeToString(pcm),
			SampleRate:  sampleRate,
			NumChannels: numChannels,
		},
	}
}

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

// Serializer bridges RTVI JSON messages and pipeline frames over a WebSocket.
// It holds no per-session state, so a single value may serve any session.
type Serializer struct{}

// New builds an RTVI WebSocket serializer.
func New() *Serializer { return &Serializer{} }

// Setup is a no-op: the RTVI channel carries no audio, so there is nothing to
// configure from the StartFrame.
func (*Serializer) Setup(processor.Setup) error { return nil }

// Serialize drops outbound frames. RTVI server messages reach the socket through
// the transport's own OutputTransportMessageUrgentFrame path rather than the
// serializer, and bot audio is not streamed over this channel.
func (*Serializer) Serialize(frames.Frame) ([]byte, error) { return nil, nil }

// Deserialize turns an inbound RTVI message into a frame: raw-audio becomes an
// InputAudioRawFrame (played into the pipeline as mic audio), and every other
// RTVI message is wrapped in an InputTransportMessageFrame so the pipeline's RTVI
// processor parses and routes it (the handshake, send-text, and so on). Payloads
// that are not RTVI messages are ignored.
func (*Serializer) Deserialize(data []byte) (frames.Frame, error) {
	in, err := rtvi.ParseIncoming(data)
	if err != nil || in.Label != rtvi.MessageLabel {
		// Malformed or non-RTVI payloads carry no frame and are dropped, per the
		// wsserver.Serializer contract — not an error.
		return nil, nil //nolint:nilnil,nilerr // dropping a non-RTVI message is intentional
	}
	if in.Type == TypeRawAudio {
		return decodeRawAudio(in.Data)
	}
	// The transport may reuse its read buffer, and the frame outlives this call,
	// so hand the RTVI processor its own copy.
	msg := make([]byte, len(data))
	copy(msg, data)
	return frames.NewInputTransportMessageFrame(msg), nil
}

// decodeRawAudio turns a raw-audio payload into an InputAudioRawFrame, or drops
// it (nil, nil) if the payload is malformed.
func decodeRawAudio(data json.RawMessage) (frames.Frame, error) {
	var d RawAudioData
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, nil //nolint:nilnil,nilerr // a malformed audio chunk is dropped
	}
	pcm, err := base64.StdEncoding.DecodeString(d.Audio)
	if err != nil || len(pcm) == 0 {
		return nil, nil //nolint:nilnil,nilerr // undecodable or empty audio is dropped
	}
	channels := d.NumChannels
	if channels == 0 {
		channels = 1
	}
	return frames.NewInputAudioRawFrame(pcm, d.SampleRate, channels), nil
}
