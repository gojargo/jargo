// Package exotel is the wsserver.Serializer for Exotel Media Streaming (the
// Voicebot applet). Unlike the μ-law telephony providers, Exotel streams raw
// 16-bit signed little-endian PCM (base64-encoded) at 8 kHz mono; this
// serializer base64-codes it to and from jargo audio frames and emits a "clear"
// message on barge-in. Exotel provides no media-stream hang-up API, so ending
// the call is left to the applet flow.
package exotel

import (
	"encoding/base64"
	"encoding/json"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

// eventMedia is the Exotel message event carrying base64 PCM audio.
const eventMedia = "media"

// Config configures the Exotel serializer.
type Config struct {
	// SampleRate is the PCM rate of the Exotel stream in Hz; 0 defaults to
	// 8000. Exotel also supports 16 kHz. It is the rate on the wire, not the
	// rate the pipeline has to run at: the serializer converts between the two.
	SampleRate int
	// Audio configures that conversion. Its WireSampleRate is taken from
	// SampleRate above, so leave it unset.
	Audio wsserver.AudioConfig
}

// Serializer implements wsserver.Serializer for Exotel. The stream SID is
// learned from the inbound "start" message.
type Serializer struct {
	wsserver.BaseSerializer

	codec *wsserver.Codec

	mu        sync.Mutex
	streamSID string
}

// New builds an Exotel serializer.
func New(cfg Config) *Serializer {
	audio := cfg.Audio
	audio.WireSampleRate = cfg.SampleRate
	return &Serializer{codec: wsserver.NewCodec(audio)}
}

// Setup learns the pipeline's sample rate, so the stream's PCM can be converted
// to it and back. Exotel streams linear PCM rather than companded audio, so only
// the rate changes.
func (s *Serializer) Setup(st processor.Setup) error { return s.codec.Setup(st) }

// Close releases the resamplers.
func (s *Serializer) Close() { s.codec.Close() }

// Serialize converts an outbound frame to an Exotel message.
func (s *Serializer) Serialize(f frames.Frame) (wsserver.Message, error) {
	switch fr := f.(type) {
	// Every kind of output audio is sent the same way, so match the family
	// rather than each concrete frame.
	case frames.OutputAudioFrame:
		return s.media(fr.AudioData())
	case *frames.InterruptionFrame:
		return s.clear()
	case frames.OutputTransportMessage:
		// An application message the pipeline addressed to the client. RTVI is
		// dropped here: it is the protocol a browser client speaks, and a
		// provider expecting its own control messages has no use for it.
		if s.ShouldIgnoreFrame(f) {
			return wsserver.Message{}, nil
		}
		return wsserver.TextMessage(json.Marshal(fr.TransportMessage()))
	default:
		return wsserver.Message{}, nil
	}
}

// Deserialize converts an Exotel message to a frame.
func (s *Serializer) Deserialize(data []byte) (frames.Frame, error) {
	var m inbound
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	switch m.Event {
	case eventMedia:
		pcm, err := base64.StdEncoding.DecodeString(m.Media.Payload)
		if err != nil {
			return nil, err
		}
		converted := s.codec.Decode(pcm, wsserver.EncodingLinear)
		if len(converted) == 0 {
			// The conversion has nothing to emit yet; no audio, no frame.
			return nil, nil //nolint:nilnil // no audio to carry
		}
		return frames.NewInputAudioRawFrame(converted, s.codec.SampleRate(), 1), nil
	case "start":
		s.mu.Lock()
		s.streamSID = m.Start.StreamSID
		s.mu.Unlock()
		return nil, nil //nolint:nilnil // handshake message carries no frame
	case "dtmf":
		if m.DTMF.Digit == "" {
			return nil, nil //nolint:nilnil // empty keypress
		}
		return frames.NewInputDTMFFrame(frames.KeypadEntry(m.DTMF.Digit)), nil
	default: // connected, mark, stop
		return nil, nil //nolint:nilnil // message carries no frame
	}
}

func (s *Serializer) media(a *frames.AudioRawData) (wsserver.Message, error) {
	s.mu.Lock()
	sid := s.streamSID
	s.mu.Unlock()
	if sid == "" {
		return wsserver.Message{}, nil
	}
	pcm := s.codec.Encode(a.Audio, a.SampleRate, wsserver.EncodingLinear)
	if len(pcm) == 0 {
		// The conversion has nothing to emit yet; no audio, no message.
		return wsserver.Message{}, nil
	}
	out := mediaOut{Event: eventMedia, StreamSID: sid}
	out.Media.Payload = base64.StdEncoding.EncodeToString(pcm)
	return wsserver.TextMessage(json.Marshal(out))
}

func (s *Serializer) clear() (wsserver.Message, error) {
	s.mu.Lock()
	sid := s.streamSID
	s.mu.Unlock()
	if sid == "" {
		return wsserver.Message{}, nil
	}
	return wsserver.TextMessage(json.Marshal(clearOut{Event: "clear", StreamSID: sid}))
}

// The JSON field names below are Exotel's wire protocol, which spells the stream
// identifier the same way in both directions.

type mediaOut struct {
	Event     string `json:"event"`
	StreamSID string `json:"stream_sid"`
	Media     struct {
		Payload string `json:"payload"`
	} `json:"media"`
}

type clearOut struct {
	Event     string `json:"event"`
	StreamSID string `json:"stream_sid"`
}

type inbound struct {
	Event string `json:"event"`
	Media struct {
		Payload string `json:"payload"`
	} `json:"media"`
	Start struct {
		StreamSID string `json:"stream_sid"`
		CallSID   string `json:"call_sid"`
	} `json:"start"`
	DTMF struct {
		Digit string `json:"digit"`
	} `json:"dtmf"`
}
