// Package twilio is the wsserver.Serializer for Twilio Media Streams. Twilio
// streams call audio as base64 μ-law 8 kHz mono in JSON text messages; this
// serializer converts those to and from jargo audio frames, emits a "clear"
// message on barge-in, and optionally hangs the call up over Twilio's REST API
// when the pipeline ends.
package twilio

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

const (
	// sampleRate is Twilio Media Streams' fixed rate: 8 kHz mono μ-law.
	sampleRate = 8000
	hangupURL  = "https://api.twilio.com/2010-04-01/Accounts/%s/Calls/%s.json"
)

// Config configures the Twilio serializer.
type Config struct {
	// AccountSID and AuthToken authorize the REST hang-up call. They are only
	// needed when AutoHangUp is set.
	AccountSID string
	AuthToken  string
	// AutoHangUp ends the Twilio call over the REST API when the pipeline sends
	// an EndFrame or CancelFrame.
	AutoHangUp bool
	// HTTPClient is used for the hang-up request; nil uses http.DefaultClient.
	HTTPClient *http.Client
}

// Serializer implements wsserver.Serializer for Twilio. The stream and call SIDs
// are learned from the inbound "start" message, so no pre-handshake read is
// needed.
type Serializer struct {
	cfg  Config
	http *http.Client

	mu        sync.Mutex
	streamSID string
	callSID   string
	hungUp    bool
}

// New builds a Twilio serializer.
func New(cfg Config) *Serializer {
	h := cfg.HTTPClient
	if h == nil {
		h = http.DefaultClient
	}
	return &Serializer{cfg: cfg, http: h}
}

// Setup is a no-op: Twilio audio is always 8 kHz.
func (s *Serializer) Setup(*frames.StartFrame) error { return nil }

// Serialize converts an outbound frame to a Twilio message.
func (s *Serializer) Serialize(f frames.Frame) ([]byte, error) {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		return s.media(fr.Audio)
	case *frames.OutputAudioRawFrame:
		return s.media(fr.Audio)
	case *frames.InterruptionFrame:
		return s.clear()
	case *frames.EndFrame, *frames.CancelFrame:
		s.hangup()
		return nil, nil //nolint:nilnil // hang-up is a side effect; no wire message
	default:
		return nil, nil //nolint:nilnil // frame not sent to Twilio
	}
}

// Deserialize converts a Twilio message to a frame.
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
		return frames.NewInputAudioRawFrame(g711.DecodeULaw(ulaw), sampleRate, 1), nil
	case "start":
		s.mu.Lock()
		s.streamSID = m.Start.StreamSID
		s.callSID = m.Start.CallSID
		s.mu.Unlock()
		return nil, nil //nolint:nilnil // handshake message carries no frame
	default: // connected, mark, stop, dtmf
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
	payload := base64.StdEncoding.EncodeToString(g711.EncodeULaw(pcm))
	out := mediaOut{Event: "media", StreamSID: sid}
	out.Media.Payload = payload
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

func (s *Serializer) hangup() {
	if !s.cfg.AutoHangUp {
		return
	}
	s.mu.Lock()
	ready := !s.hungUp && s.callSID != "" && s.cfg.AccountSID != "" && s.cfg.AuthToken != ""
	if ready {
		s.hungUp = true
	}
	callSID := s.callSID
	s.mu.Unlock()
	if !ready {
		return
	}
	go s.doHangup(callSID)
}

func (s *Serializer) doHangup(callSID string) {
	endpoint := fmt.Sprintf(hangupURL, s.cfg.AccountSID, callSID)
	body := strings.NewReader(url.Values{"Status": {"completed"}}.Encode())
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, endpoint, body)
	if err != nil {
		slog.Warn("twilio: build hang-up request", "err", err)
		return
	}
	req.SetBasicAuth(s.cfg.AccountSID, s.cfg.AuthToken)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := s.http.Do(req)
	if err != nil {
		slog.Warn("twilio: hang-up call", "err", err)
		return
	}
	_ = resp.Body.Close()
}

// The JSON field names below are Twilio's wire protocol (camelCase), so the
// snake_case house style does not apply.

type mediaOut struct {
	Event     string `json:"event"`
	StreamSID string `json:"streamSid"` //nolint:tagliatelle // Twilio wire field
	Media     struct {
		Payload string `json:"payload"`
	} `json:"media"`
}

type clearOut struct {
	Event     string `json:"event"`
	StreamSID string `json:"streamSid"` //nolint:tagliatelle // Twilio wire field
}

type inbound struct {
	Event string `json:"event"`
	Media struct {
		Payload string `json:"payload"`
	} `json:"media"`
	Start struct {
		StreamSID string `json:"streamSid"` //nolint:tagliatelle // Twilio wire field
		CallSID   string `json:"callSid"`   //nolint:tagliatelle // Twilio wire field
	} `json:"start"`
}
