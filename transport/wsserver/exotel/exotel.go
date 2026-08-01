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
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

const (
	// defaultSampleRate is Exotel's default stream rate. Exotel also supports 16 kHz;
	// set Config.SampleRate to match and run the pipeline at the same rate.
	defaultSampleRate = 8000

	// eventMedia is the Exotel message event carrying base64 PCM audio.
	eventMedia = "media"
)

// Config configures the Exotel serializer.
type Config struct {
	// SampleRate is the PCM rate of the Exotel stream in Hz; 0 defaults to 8000.
	// The pipeline must run at this rate, since the serializer does not resample.
	SampleRate int
}

// Serializer implements wsserver.Serializer for Exotel. The stream SID is
// learned from the inbound "start" message.
type Serializer struct {
	rate int

	mu        sync.Mutex
	streamSID string
}

// New builds an Exotel serializer.
func New(cfg Config) *Serializer {
	rate := cfg.SampleRate
	if rate == 0 {
		rate = defaultSampleRate
	}
	return &Serializer{rate: rate}
}

// Setup is a no-op: the stream rate is fixed by Config.
func (s *Serializer) Setup(*frames.StartFrame) error { return nil }

// Serialize converts an outbound frame to an Exotel message.
func (s *Serializer) Serialize(f frames.Frame) ([]byte, error) {
	switch fr := f.(type) {
	// Every kind of output audio is sent the same way, so match the family
	// rather than each concrete frame.
	case frames.OutputAudioFrame:
		return s.media(fr.AudioData().Audio)
	case *frames.InterruptionFrame:
		return s.clear()
	default:
		return nil, nil //nolint:nilnil // frame not sent to Exotel
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
		return frames.NewInputAudioRawFrame(pcm, s.rate, 1), nil
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

func (s *Serializer) media(pcm []byte) ([]byte, error) {
	s.mu.Lock()
	sid := s.streamSID
	s.mu.Unlock()
	if sid == "" {
		return nil, nil //nolint:nilnil // stream not started yet; drop until "start" arrives
	}
	out := mediaOut{Event: eventMedia, StreamSID: sid}
	out.Media.Payload = base64.StdEncoding.EncodeToString(pcm)
	return json.Marshal(out)
}

func (s *Serializer) clear() ([]byte, error) {
	s.mu.Lock()
	sid := s.streamSID
	s.mu.Unlock()
	if sid == "" {
		return nil, nil //nolint:nilnil // stream not started yet; nothing to clear
	}
	return json.Marshal(clearOut{Event: "clear", StreamSID: sid})
}

// The JSON field names below are Exotel's wire protocol (camelCase for the
// outbound streamSid), so the snake_case house style does not apply.

type mediaOut struct {
	Event     string `json:"event"`
	StreamSID string `json:"streamSid"` //nolint:tagliatelle // Exotel wire field
	Media     struct {
		Payload string `json:"payload"`
	} `json:"media"`
}

type clearOut struct {
	Event     string `json:"event"`
	StreamSID string `json:"streamSid"` //nolint:tagliatelle // Exotel wire field
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
