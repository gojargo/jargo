// Package plivo is the wsserver.Serializer for Plivo Audio Streaming. Plivo
// streams call audio as base64 μ-law 8 kHz mono in JSON text messages; this
// serializer converts those to and from jargo audio frames, emits a "clearAudio"
// message on barge-in, and optionally hangs the call up over Plivo's REST API
// when the pipeline ends.
package plivo

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

const hangupURL = "https://api.plivo.com/v1/Account/%s/Call/%s/"

// Config configures the Plivo serializer.
type Config struct {
	// AuthID and AuthToken authorize the REST hang-up call. They are only needed
	// when AutoHangUp is set.
	AuthID    string
	AuthToken string
	// AutoHangUp ends the Plivo call over the REST API when the pipeline sends an
	// EndFrame or CancelFrame.
	AutoHangUp bool
	// HTTPClient is used for the hang-up request; nil uses http.DefaultClient.
	HTTPClient *http.Client
	// Audio configures the conversion between Plivo's 8 kHz μ-law wire audio and
	// the rate the pipeline runs at. The zero value converts to and from
	// whatever the StartFrame carries, so the pipeline is free to run at a rate
	// its services are happier with than 8 kHz.
	Audio wsserver.AudioConfig
}

// Serializer implements wsserver.Serializer for Plivo. The stream and call IDs
// are learned from the inbound "start" message.
type Serializer struct {
	cfg   Config
	http  *http.Client
	codec *wsserver.Codec

	mu       sync.Mutex
	streamID string
	callID   string
	hungUp   bool
}

// New builds a Plivo serializer.
func New(cfg Config) *Serializer {
	h := cfg.HTTPClient
	if h == nil {
		h = http.DefaultClient
	}
	return &Serializer{cfg: cfg, http: h, codec: wsserver.NewCodec(cfg.Audio)}
}

// Setup learns the pipeline's sample rate, so the 8 kHz μ-law on the wire can be
// converted to it and back.
func (s *Serializer) Setup(f *frames.StartFrame) error { return s.codec.Setup(f) }

// Close releases the resamplers.
func (s *Serializer) Close() { s.codec.Close() }

// Serialize converts an outbound frame to a Plivo message.
func (s *Serializer) Serialize(f frames.Frame) ([]byte, error) {
	switch fr := f.(type) {
	// Every kind of output audio is sent the same way, so match the family
	// rather than each concrete frame.
	case frames.OutputAudioFrame:
		return s.media(fr.AudioData())
	case *frames.InterruptionFrame:
		return s.clear()
	case *frames.EndFrame, *frames.CancelFrame:
		s.hangup()
		return nil, nil //nolint:nilnil // hang-up is a side effect; no wire message
	default:
		return nil, nil //nolint:nilnil // frame not sent to Plivo
	}
}

// Deserialize converts a Plivo message to a frame.
func (s *Serializer) Deserialize(data []byte) (frames.Frame, error) {
	var m inbound
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	switch m.Event {
	case "media":
		ulaw, err := base64.StdEncoding.DecodeString(m.Media.Payload)
		if err != nil {
			return nil, err
		}
		pcm := s.codec.Decode(ulaw, wsserver.EncodingULaw)
		if len(pcm) == 0 {
			// The conversion has nothing to emit yet; no audio, no frame.
			return nil, nil //nolint:nilnil // no audio to carry
		}
		return frames.NewInputAudioRawFrame(pcm, s.codec.SampleRate(), 1), nil
	case "start":
		s.mu.Lock()
		s.streamID = m.Start.StreamID
		s.callID = m.Start.CallID
		s.mu.Unlock()
		return nil, nil //nolint:nilnil // handshake message carries no frame
	case "dtmf":
		if m.DTMF.Digit == "" {
			return nil, nil //nolint:nilnil // empty keypress
		}
		return frames.NewInputDTMFFrame(frames.KeypadEntry(m.DTMF.Digit)), nil
	default: // connected, checkpoint, clearedAudio
		return nil, nil //nolint:nilnil // message carries no frame
	}
}

func (s *Serializer) media(a *frames.AudioRawData) ([]byte, error) {
	s.mu.Lock()
	id := s.streamID
	s.mu.Unlock()
	if id == "" {
		return nil, nil //nolint:nilnil // stream not started yet; drop until "start" arrives
	}
	ulaw := s.codec.Encode(a.Audio, a.SampleRate, wsserver.EncodingULaw)
	if len(ulaw) == 0 {
		// The conversion has nothing to emit yet; no audio, no message.
		return nil, nil //nolint:nilnil // no audio to send
	}
	out := playAudio{Event: "playAudio", StreamID: id}
	out.Media.ContentType = "audio/x-mulaw"
	// The rate the payload is at, which is the wire's, not the pipeline's.
	out.Media.SampleRate = s.codec.WireSampleRate()
	out.Media.Payload = base64.StdEncoding.EncodeToString(ulaw)
	return json.Marshal(out)
}

func (s *Serializer) clear() ([]byte, error) {
	s.mu.Lock()
	id := s.streamID
	s.mu.Unlock()
	if id == "" {
		return nil, nil //nolint:nilnil // stream not started yet; nothing to clear
	}
	return json.Marshal(clearAudio{Event: "clearAudio", StreamID: id})
}

func (s *Serializer) hangup() {
	if !s.cfg.AutoHangUp {
		return
	}
	s.mu.Lock()
	ready := !s.hungUp && s.callID != "" && s.cfg.AuthID != "" && s.cfg.AuthToken != ""
	if ready {
		s.hungUp = true
	}
	callID := s.callID
	s.mu.Unlock()
	if !ready {
		return
	}
	go s.doHangup(callID)
}

func (s *Serializer) doHangup(callID string) {
	endpoint := fmt.Sprintf(hangupURL, s.cfg.AuthID, callID)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodDelete, endpoint, nil)
	if err != nil {
		slog.Warn("plivo: build hang-up request", "err", err)
		return
	}
	req.SetBasicAuth(s.cfg.AuthID, s.cfg.AuthToken)
	resp, err := s.http.Do(req)
	if err != nil {
		slog.Warn("plivo: hang-up call", "err", err)
		return
	}
	_ = resp.Body.Close()
}

// The JSON field names below are Plivo's wire protocol (camelCase), so the
// snake_case house style does not apply.

type playAudio struct {
	Event string `json:"event"`
	Media struct {
		ContentType string `json:"contentType"` //nolint:tagliatelle // Plivo wire field
		SampleRate  int    `json:"sampleRate"`  //nolint:tagliatelle // Plivo wire field
		Payload     string `json:"payload"`
	} `json:"media"`
	StreamID string `json:"streamId"` //nolint:tagliatelle // Plivo wire field
}

type clearAudio struct {
	Event    string `json:"event"`
	StreamID string `json:"streamId"` //nolint:tagliatelle // Plivo wire field
}

type inbound struct {
	Event string `json:"event"`
	Media struct {
		Payload string `json:"payload"`
	} `json:"media"`
	Start struct {
		StreamID string `json:"streamId"` //nolint:tagliatelle // Plivo wire field
		CallID   string `json:"callId"`   //nolint:tagliatelle // Plivo wire field
	} `json:"start"`
	DTMF struct {
		Digit string `json:"digit"`
	} `json:"dtmf"`
}
