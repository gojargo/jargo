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
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

// defaultAPIRoot is Twilio's API host when no region, edge or explicit root is
// named. callResourceURL hangs the path off whichever root applies.
const defaultAPIRoot = "https://api.twilio.com"

// callResourcePath is the REST path of one Call resource, under the API root.
const callResourcePath = "/2010-04-01/Accounts/%s/Calls/%s.json"

// Config configures the Twilio serializer.
type Config struct {
	// AccountSID and AuthToken authorize the REST hang-up call. They are
	// required unless AutoHangUp is turned off.
	AccountSID string
	AuthToken  string
	// AutoHangUp ends the Twilio call over the REST API when the pipeline sends
	// an EndFrame or CancelFrame.
	//
	// Nil, the default, ends it. A pipeline that has finished leaves the caller
	// listening to silence, and the leg billing, until something hangs up. Set
	// it to false for a bot that is one step of a longer call the provider goes
	// on to route elsewhere.
	AutoHangUp *bool
	// Region and Edge name the Twilio region and edge the hang-up is sent to,
	// giving the host api.{Edge}.{Region}.twilio.com. A call carried on a
	// regional edge is ended there, not at the global host.
	//
	// Twilio's host format needs both, so naming one without the other is
	// refused. Neither is needed when BaseURL is set, which replaces the host
	// outright.
	Region string
	Edge   string
	// BaseURL is the API root the hang-up is sent to, scheme and host, used
	// verbatim: "https://api.example.test". It is for a Twilio-compatible or
	// self-hosted backend, and it takes the place of Region and Edge rather than
	// combining with them.
	BaseURL string
	// HTTPClient is used for the hang-up request; nil uses http.DefaultClient.
	HTTPClient *http.Client
	// Audio configures the conversion between Twilio's 8 kHz μ-law wire audio
	// and the rate the pipeline runs at. The zero value converts to and from
	// whatever the StartFrame carries, so the pipeline is free to run at a rate
	// its services are happier with than 8 kHz.
	Audio wsserver.AudioConfig
}

// ErrHangUpCredentials reports a configuration that is to hang the call up but
// has nothing to authorize it with.
var ErrHangUpCredentials = errors.New("twilio: AutoHangUp is on but credentials are missing")

// ErrRegionEdgePair reports a configuration naming one half of Twilio's
// region-and-edge host and not the other.
var ErrRegionEdgePair = errors.New("twilio: Region and Edge are set together or not at all")

// autoHangUp reports whether the call is ended when the pipeline finishes, which
// it is unless the configuration says otherwise.
func (c Config) autoHangUp() bool { return c.AutoHangUp == nil || *c.AutoHangUp }

// Validate reports whether the configuration is usable. A serializer that is to
// hang the call up needs the credentials to do it, so a missing one is refused
// when the pipeline starts rather than discovered later as a call that never
// ended.
//
// The call SID is not checked: it is learned from Twilio's own "start" message,
// so there is nothing to check until the call is under way.
func (c Config) Validate() error {
	if !c.autoHangUp() {
		return nil
	}
	var missing []string
	if c.AccountSID == "" {
		missing = append(missing, "AccountSID")
	}
	if c.AuthToken == "" {
		missing = append(missing, "AuthToken")
	}
	if len(missing) > 0 {
		return fmt.Errorf("%w: %s", ErrHangUpCredentials, strings.Join(missing, ", "))
	}
	// The host is built from both halves, so one on its own would silently
	// address a host that does not exist. A BaseURL replaces the host outright,
	// which is why the pairing does not apply there.
	if c.BaseURL == "" && (c.Region == "") != (c.Edge == "") {
		return fmt.Errorf("%w: Region=%q Edge=%q", ErrRegionEdgePair, c.Region, c.Edge)
	}
	return nil
}

// callResourceURL is the REST URL of the call resource to hang up.
//
// BaseURL, when set, is the API root as given. Otherwise the root is Twilio's
// own host, with the edge and region worked into it when they are named.
func (c Config) callResourceURL(callSID string) string {
	root := defaultAPIRoot
	switch {
	case c.BaseURL != "":
		root = strings.TrimRight(c.BaseURL, "/")
	case c.Region != "" || c.Edge != "":
		var host strings.Builder
		host.WriteString("https://api.")
		if c.Edge != "" {
			host.WriteString(c.Edge + ".")
		}
		if c.Region != "" {
			host.WriteString(c.Region + ".")
		}
		host.WriteString("twilio.com")
		root = host.String()
	}
	return root + fmt.Sprintf(callResourcePath, c.AccountSID, callSID)
}

// Serializer implements wsserver.Serializer for Twilio. The stream and call SIDs
// are learned from the inbound "start" message, so no pre-handshake read is
// needed.
type Serializer struct {
	cfg   Config
	http  *http.Client
	codec *wsserver.Codec

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
	return &Serializer{cfg: cfg, http: h, codec: wsserver.NewCodec(cfg.Audio)}
}

// Setup learns the pipeline's sample rate, so the 8 kHz μ-law on the wire can be
// converted to it and back.
func (s *Serializer) Setup(st processor.Setup) error {
	if err := s.cfg.Validate(); err != nil {
		return err
	}
	return s.codec.Setup(st)
}

// Close releases the resamplers.
func (s *Serializer) Close() { s.codec.Close() }

// Serialize converts an outbound frame to a Twilio message.
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
		pcm := s.codec.Decode(ulaw, wsserver.EncodingULaw)
		if len(pcm) == 0 {
			// The conversion has nothing to emit yet; no audio, no frame.
			return nil, nil //nolint:nilnil // no audio to carry
		}
		return frames.NewInputAudioRawFrame(pcm, s.codec.SampleRate(), 1), nil
	case "start":
		s.mu.Lock()
		s.streamSID = m.Start.StreamSID
		s.callSID = m.Start.CallSID
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

func (s *Serializer) media(a *frames.AudioRawData) ([]byte, error) {
	s.mu.Lock()
	sid := s.streamSID
	s.mu.Unlock()
	if sid == "" {
		return nil, nil //nolint:nilnil // stream not started yet; drop until "start" arrives
	}
	ulaw := s.codec.Encode(a.Audio, a.SampleRate, wsserver.EncodingULaw)
	if len(ulaw) == 0 {
		// The conversion has nothing to emit yet; no audio, no message.
		return nil, nil //nolint:nilnil // no audio to send
	}
	payload := base64.StdEncoding.EncodeToString(ulaw)
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
	if !s.cfg.autoHangUp() {
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
	endpoint := s.cfg.callResourceURL(callSID)
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
	DTMF struct {
		Digit string `json:"digit"`
	} `json:"dtmf"`
}
