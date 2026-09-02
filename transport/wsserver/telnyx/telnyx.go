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
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

const (
	hangupURL = "https://api.telnyx.com/v2/calls/%s/actions/hangup"

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
	// AutoHangUp is left on.
	APIKey string
	// AutoHangUp ends the Telnyx call over the REST API when the pipeline sends an
	// EndFrame or CancelFrame.
	//
	// Nil, the default, ends it. A pipeline that has finished leaves the caller
	// listening to silence, and the leg billing, until something hangs up. Set
	// it to false for a bot that is one step of a longer call the provider goes
	// on to route elsewhere.
	AutoHangUp *bool
	// SendEncoding is the companding used for audio sent to Telnyx (PCMU or
	// PCMA); empty defaults to PCMU. The receive encoding is learned from the
	// "start" message's media_format, falling back to this value.
	SendEncoding string
	// HTTPClient is used for the hang-up request; nil uses http.DefaultClient.
	HTTPClient *http.Client
	// Audio configures the conversion between Telnyx's 8 kHz companded wire
	// audio and the rate the pipeline runs at. The zero value converts to and
	// from whatever the StartFrame carries, so the pipeline is free to run at a
	// rate its services are happier with than 8 kHz.
	Audio wsserver.AudioConfig
}

// ErrHangUpCredentials reports a configuration that is to hang the call up but
// has nothing to authorize it with.
var ErrHangUpCredentials = errors.New("telnyx: AutoHangUp is on but credentials are missing")

// autoHangUp reports whether the call is ended when the pipeline finishes, which
// it is unless the configuration says otherwise.
func (c Config) autoHangUp() bool { return c.AutoHangUp == nil || *c.AutoHangUp }

// Validate reports whether the configuration is usable. A serializer that is to
// hang the call up needs the credentials to do it, so a missing one is refused
// when the pipeline starts rather than discovered later as a call that never
// ended.
//
// The call control id is not checked: it is learned from Telnyx's own "start"
// message, so there is nothing to check until the call is under way.
func (c Config) Validate() error {
	if !c.autoHangUp() {
		return nil
	}
	var missing []string
	if c.APIKey == "" {
		missing = append(missing, "APIKey")
	}
	if len(missing) > 0 {
		return fmt.Errorf("%w: %s", ErrHangUpCredentials, strings.Join(missing, ", "))
	}
	return nil
}

// Serializer implements wsserver.Serializer for Telnyx. The stream ID, call
// control ID and receive encoding are learned from the inbound "start" message.
type Serializer struct {
	cfg   Config
	http  *http.Client
	codec *wsserver.Codec
	send  string

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
	return &Serializer{cfg: cfg, http: h, send: send, recv: send, codec: wsserver.NewCodec(cfg.Audio)}
}

// Setup learns the pipeline's sample rate, so the 8 kHz companded audio on the
// wire can be converted to it and back.
func (s *Serializer) Setup(st processor.Setup) error {
	if err := s.cfg.Validate(); err != nil {
		return err
	}
	return s.codec.Setup(st)
}

// Close releases the resamplers.
func (s *Serializer) Close() { s.codec.Close() }

// Serialize converts an outbound frame to a Telnyx message.
func (s *Serializer) Serialize(f frames.Frame) (wsserver.Message, error) {
	switch fr := f.(type) {
	// Every kind of output audio is sent the same way, so match the family
	// rather than each concrete frame.
	case frames.OutputAudioFrame:
		return s.media(fr.AudioData())
	case *frames.InterruptionFrame:
		return wsserver.TextMessage(json.Marshal(event{Event: "clear"}))
	case *frames.EndFrame, *frames.CancelFrame:
		s.hangup()
		return wsserver.Message{}, nil
	default:
		// Application messages are among the frames not sent: unlike the other
		// providers, this wire carries no pass-through JSON, so a message the
		// pipeline addresses to the client has nowhere to go on a Telnyx call.
		return wsserver.Message{}, nil
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
		pcm := s.codec.Decode(payload, encodingOf(enc))
		if len(pcm) == 0 {
			// The conversion has nothing to emit yet; no audio, no frame.
			return nil, nil //nolint:nilnil // no audio to carry
		}
		return frames.NewInputAudioRawFrame(pcm, s.codec.SampleRate(), 1), nil
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

func (s *Serializer) media(a *frames.AudioRawData) (wsserver.Message, error) {
	companded := s.codec.Encode(a.Audio, a.SampleRate, encodingOf(s.send))
	if len(companded) == 0 {
		// The conversion has nothing to emit yet; no audio, no message.
		return wsserver.Message{}, nil
	}
	out := mediaOut{Event: eventMedia}
	out.Media.Payload = base64.StdEncoding.EncodeToString(companded)
	return wsserver.TextMessage(json.Marshal(out))
}

func (s *Serializer) hangup() {
	if !s.cfg.autoHangUp() {
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

// encodingOf maps a Telnyx media_format encoding name to the companding it
// names, defaulting to μ-law for anything else.
func encodingOf(enc string) wsserver.Encoding {
	if enc == EncodingPCMA {
		return wsserver.EncodingALaw
	}
	return wsserver.EncodingULaw
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
