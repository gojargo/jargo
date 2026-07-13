// Package telnyx is the wsserver.Serializer for Telnyx Media Streaming. Telnyx
// streams call audio as base64 companded 8 kHz mono (μ-law or A-law) in JSON
// text messages; this serializer converts those to and from jargo audio frames,
// emits a "clear" message on barge-in, and optionally hangs the call up over
// Telnyx's REST API when the pipeline ends.
package telnyx

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

const (
	// sampleRate is Telnyx Media Streaming's fixed rate: 8 kHz mono.
	sampleRate = 8000
	hangupURL  = "https://api.telnyx.com/v2/calls/%s/actions/hangup"

	// eventMedia is the Telnyx message event carrying base64 audio.
	eventMedia = "media"

	// EncodingPCMU selects G.711 μ-law; EncodingPCMA selects G.711 A-law. These
	// are Telnyx's media_format.encoding values.
	EncodingPCMU = "PCMU"
	EncodingPCMA = "PCMA"
)

// Config configures the Telnyx serializer.
type Config struct {
	// APIKey authorizes the REST hang-up call (Bearer token). Only needed when
	// AutoHangUp is set.
	APIKey string
	// AutoHangUp ends the Telnyx call over the REST API when the pipeline sends an
	// EndFrame or CancelFrame.
	AutoHangUp bool
	// SendEncoding is the companding used for audio sent to Telnyx (PCMU or
	// PCMA); empty defaults to PCMU. The receive encoding is learned from the
	// "start" message's media_format, falling back to this value.
	SendEncoding string
	// HTTPClient is used for the hang-up request; nil uses http.DefaultClient.
	HTTPClient *http.Client
}

// Serializer implements wsserver.Serializer for Telnyx. The stream ID, call
// control ID and receive encoding are learned from the inbound "start" message.
type Serializer struct {
	cfg  Config
	http *http.Client
	send string

	mu       sync.Mutex
	streamID string
	callCtrl string
	recv     string
	hungUp   bool
}

// New builds a Telnyx serializer.
func New(cfg Config) *Serializer {
	h := cfg.HTTPClient
	if h == nil {
		h = http.DefaultClient
	}
	send := cfg.SendEncoding
	if send == "" {
		send = EncodingPCMU
	}
	return &Serializer{cfg: cfg, http: h, send: send, recv: send}
}

// Setup is a no-op: Telnyx audio is always 8 kHz.
func (s *Serializer) Setup(*frames.StartFrame) error { return nil }

// Serialize converts an outbound frame to a Telnyx message.
func (s *Serializer) Serialize(f frames.Frame) ([]byte, error) {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		return s.media(fr.Audio)
	case *frames.OutputAudioRawFrame:
		return s.media(fr.Audio)
	case *frames.InterruptionFrame:
		return json.Marshal(event{Event: "clear"})
	case *frames.EndFrame, *frames.CancelFrame:
		s.hangup()
		return nil, nil //nolint:nilnil // hang-up is a side effect; no wire message
	default:
		return nil, nil //nolint:nilnil // frame not sent to Telnyx
	}
}

// Deserialize converts a Telnyx message to a frame.
func (s *Serializer) Deserialize(data []byte) (frames.Frame, error) {
	var m inbound
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	switch m.Event {
	case eventMedia:
		payload, err := base64.StdEncoding.DecodeString(m.Media.Payload)
		if err != nil {
			return nil, err
		}
		s.mu.Lock()
		enc := s.recv
		s.mu.Unlock()
		return frames.NewInputAudioRawFrame(decode(enc, payload), sampleRate, 1), nil
	case "start":
		s.mu.Lock()
		s.streamID = m.StreamID
		s.callCtrl = m.Start.CallControlID
		if enc := m.Start.MediaFormat.Encoding; enc != "" {
			s.recv = enc
		}
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
	out := mediaOut{Event: eventMedia}
	out.Media.Payload = base64.StdEncoding.EncodeToString(encode(s.send, pcm))
	return json.Marshal(out)
}

func (s *Serializer) hangup() {
	if !s.cfg.AutoHangUp {
		return
	}
	s.mu.Lock()
	ready := !s.hungUp && s.callCtrl != "" && s.cfg.APIKey != ""
	if ready {
		s.hungUp = true
	}
	callCtrl := s.callCtrl
	s.mu.Unlock()
	if !ready {
		return
	}
	go s.doHangup(callCtrl)
}

func (s *Serializer) doHangup(callCtrl string) {
	endpoint := fmt.Sprintf(hangupURL, callCtrl)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, endpoint, nil)
	if err != nil {
		slog.Warn("telnyx: build hang-up request", "err", err)
		return
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	resp, err := s.http.Do(req)
	if err != nil {
		slog.Warn("telnyx: hang-up call", "err", err)
		return
	}
	_ = resp.Body.Close()
}

// encode companded bytes from PCM per the given encoding, defaulting to μ-law.
func encode(enc string, pcm []byte) []byte {
	if enc == EncodingPCMA {
		return g711.EncodeALaw(pcm)
	}
	return g711.EncodeULaw(pcm)
}

// decode PCM from companded bytes per the given encoding, defaulting to μ-law.
func decode(enc string, companded []byte) []byte {
	if enc == EncodingPCMA {
		return g711.DecodeALaw(companded)
	}
	return g711.DecodeULaw(companded)
}

type event struct {
	Event string `json:"event"`
}

type mediaOut struct {
	Event string `json:"event"`
	Media struct {
		Payload string `json:"payload"`
	} `json:"media"`
}

type inbound struct {
	Event    string `json:"event"`
	StreamID string `json:"stream_id"`
	Media    struct {
		Payload string `json:"payload"`
	} `json:"media"`
	Start struct {
		CallControlID string `json:"call_control_id"`
		MediaFormat   struct {
			Encoding string `json:"encoding"`
		} `json:"media_format"`
	} `json:"start"`
	DTMF struct {
		Digit string `json:"digit"`
	} `json:"dtmf"`
}
