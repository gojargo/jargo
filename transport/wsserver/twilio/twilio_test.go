package twilio

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/frames"
)

// pcm builds a deterministic 16-bit PCM buffer to push through the codec.
func pcm() []byte {
	b := make([]byte, 160*2)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}

// startMsg is the handshake that teaches the serializer both SIDs.
const startMsg = `{"event":"start","start":{"streamSid":"stream-1","callSid":"call-1"}}`

// started returns a serializer that has seen the "start" message.
func started(t *testing.T, cfg Config) *Serializer {
	t.Helper()
	s := New(cfg)
	f, err := s.Deserialize([]byte(startMsg))
	if err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	if f != nil {
		t.Errorf("start message should carry no frame, got %T", f)
	}
	return s
}

func TestSetup(t *testing.T) {
	// Twilio audio is always 8 kHz, so Setup has nothing to negotiate.
	if err := New(Config{}).Setup(frames.NewStartFrame()); err != nil {
		t.Errorf("Setup: %v", err)
	}
}

func TestDeserializeMediaDecodesULaw(t *testing.T) {
	s := New(Config{})
	ulaw := g711.EncodeULaw(pcm())
	payload := base64.StdEncoding.EncodeToString(ulaw)

	f, err := s.Deserialize([]byte(`{"event":"media","media":{"payload":"` + payload + `"}}`))
	if err != nil {
		t.Fatalf("deserialize media: %v", err)
	}
	af, ok := f.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("media frame type = %T, want *frames.InputAudioRawFrame", f)
	}
	if af.SampleRate != sampleRate {
		t.Errorf("sample rate = %d, want %d", af.SampleRate, sampleRate)
	}
	if af.NumChannels != 1 {
		t.Errorf("channels = %d, want 1", af.NumChannels)
	}
	// μ-law is lossy, so the check is against the codec, not the original PCM.
	if !bytes.Equal(af.Audio, g711.DecodeULaw(ulaw)) {
		t.Error("audio was not μ-law decoded")
	}
}

func TestDeserializeMediaBadBase64(t *testing.T) {
	if _, err := New(Config{}).Deserialize([]byte(`{"event":"media","media":{"payload":"!!not base64!!"}}`)); err == nil {
		t.Error("want an error for an undecodable payload")
	}
}

func TestDeserializeMalformedJSON(t *testing.T) {
	if _, err := New(Config{}).Deserialize([]byte(`{`)); err == nil {
		t.Error("want an error for malformed JSON")
	}
}

func TestDeserializeDTMF(t *testing.T) {
	tests := []struct {
		name  string
		msg   string
		want  frames.KeypadEntry
		frame bool
	}{
		{"digit", `{"event":"dtmf","dtmf":{"digit":"5"}}`, frames.KeypadFive, true},
		{"pound", `{"event":"dtmf","dtmf":{"digit":"#"}}`, frames.KeypadPound, true},
		{"empty keypress is dropped", `{"event":"dtmf","dtmf":{"digit":""}}`, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := New(Config{}).Deserialize([]byte(tt.msg))
			if err != nil {
				t.Fatalf("deserialize dtmf: %v", err)
			}
			if !tt.frame {
				if f != nil {
					t.Errorf("want no frame, got %T", f)
				}
				return
			}
			df, ok := f.(*frames.InputDTMFFrame)
			if !ok {
				t.Fatalf("frame type = %T, want *frames.InputDTMFFrame", f)
			}
			if df.Button != tt.want {
				t.Errorf("button = %q, want %q", df.Button, tt.want)
			}
		})
	}
}

// TestDeserializeIgnoredEvents covers the messages Twilio sends that carry no
// audio or input.
func TestDeserializeIgnoredEvents(t *testing.T) {
	for _, msg := range []string{
		`{"event":"connected"}`,
		`{"event":"mark"}`,
		`{"event":"stop"}`,
	} {
		f, err := New(Config{}).Deserialize([]byte(msg))
		if err != nil {
			t.Errorf("deserialize %s: %v", msg, err)
		}
		if f != nil {
			t.Errorf("deserialize %s produced %T, want no frame", msg, f)
		}
	}
}

func TestSerializeAudio(t *testing.T) {
	tests := []struct {
		name  string
		frame frames.Frame
	}{
		{"tts audio", frames.NewTTSAudioRawFrame(pcm(), sampleRate, 1)},
		{"output audio", frames.NewOutputAudioRawFrame(pcm(), sampleRate, 1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := started(t, Config{})
			msg, err := s.Serialize(tt.frame)
			if err != nil {
				t.Fatalf("serialize: %v", err)
			}
			var out mediaOut
			if err := json.Unmarshal(msg, &out); err != nil {
				t.Fatalf("unmarshal media: %v", err)
			}
			if out.Event != "media" {
				t.Errorf("event = %q, want media", out.Event)
			}
			if out.StreamSID != "stream-1" {
				t.Errorf("streamSid = %q, want the SID from the start message", out.StreamSID)
			}
			raw, err := base64.StdEncoding.DecodeString(out.Media.Payload)
			if err != nil {
				t.Fatalf("decode payload: %v", err)
			}
			if !bytes.Equal(raw, g711.EncodeULaw(pcm())) {
				t.Error("payload was not μ-law encoded")
			}
		})
	}
}

// TestSerializeBeforeStart checks audio and clears are dropped until the start
// message supplies a stream SID, since Twilio rejects messages without one.
func TestSerializeBeforeStart(t *testing.T) {
	for _, tt := range []struct {
		name  string
		frame frames.Frame
	}{
		{"audio", frames.NewTTSAudioRawFrame(pcm(), sampleRate, 1)},
		{"interruption", frames.NewInterruptionFrame()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			msg, err := New(Config{}).Serialize(tt.frame)
			if err != nil {
				t.Fatalf("serialize: %v", err)
			}
			if msg != nil {
				t.Errorf("want the frame dropped before start, got %q", msg)
			}
		})
	}
}

func TestSerializeInterruption(t *testing.T) {
	s := started(t, Config{})
	msg, err := s.Serialize(frames.NewInterruptionFrame())
	if err != nil {
		t.Fatalf("serialize interruption: %v", err)
	}
	var out clearOut
	if err := json.Unmarshal(msg, &out); err != nil {
		t.Fatalf("unmarshal clear: %v", err)
	}
	if out.Event != "clear" || out.StreamSID != "stream-1" {
		t.Errorf("clear = %+v, want a clear for stream-1", out)
	}
}

// TestSerializeUnhandledFrame checks frames Twilio has no representation for are
// silently skipped rather than erroring the connection.
func TestSerializeUnhandledFrame(t *testing.T) {
	s := started(t, Config{})
	msg, err := s.Serialize(frames.NewUserSpeakingFrame())
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	if msg != nil {
		t.Errorf("want no wire message, got %q", msg)
	}
}

// hangupServer stands in for Twilio's REST API and reports each hang-up it
// receives.
func hangupServer(t *testing.T) (*httptest.Server, chan *http.Request) {
	t.Helper()
	got := make(chan *http.Request, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("parse hang-up form: %v", err)
		}
		got <- r
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

// redirectToServer rewrites Twilio's REST host to the test server, since the
// endpoint is built from a package constant.
func redirectToServer(srv *httptest.Server) *http.Client {
	return &http.Client{
		Transport: rewriteHost{target: srv.URL, base: srv.Client().Transport},
	}
}

type rewriteHost struct {
	target string
	base   http.RoundTripper
}

func (r rewriteHost) RoundTrip(req *http.Request) (*http.Response, error) {
	u, err := url.Parse(r.target)
	if err != nil {
		return nil, err
	}
	req = req.Clone(req.Context())
	req.URL.Scheme, req.URL.Host, req.Host = u.Scheme, u.Host, u.Host
	return r.base.RoundTrip(req)
}

// TestAutoHangUp checks the REST hang-up: it targets the call SID learned from
// the start message, authorizes with the account credentials, and fires once.
func TestAutoHangUp(t *testing.T) {
	srv, got := hangupServer(t)
	s := started(t, Config{
		AccountSID: "AC123",
		AuthToken:  "token",
		AutoHangUp: true,
		HTTPClient: redirectToServer(srv),
	})

	if _, err := s.Serialize(frames.NewEndFrame()); err != nil {
		t.Fatalf("serialize end: %v", err)
	}

	var req *http.Request
	select {
	case req = <-got:
	case <-time.After(2 * time.Second):
		t.Fatal("no hang-up request arrived")
	}

	if req.Method != http.MethodPost {
		t.Errorf("method = %s, want POST", req.Method)
	}
	if want := "/2010-04-01/Accounts/AC123/Calls/call-1.json"; req.URL.Path != want {
		t.Errorf("path = %q, want %q", req.URL.Path, want)
	}
	user, pass, ok := req.BasicAuth()
	if !ok || user != "AC123" || pass != "token" {
		t.Errorf("basic auth = (%q, %q, %v), want the account credentials", user, pass, ok)
	}
	if got := req.PostFormValue("Status"); got != "completed" {
		t.Errorf("Status = %q, want completed", got)
	}

	// A CancelFrame after the EndFrame must not hang the call up twice.
	if _, err := s.Serialize(frames.NewCancelFrame()); err != nil {
		t.Fatalf("serialize cancel: %v", err)
	}
	select {
	case <-got:
		t.Error("hang-up fired twice")
	case <-time.After(200 * time.Millisecond):
	}
}

// TestAutoHangUpSkipped covers every reason the serializer declines to call the
// REST API. Each must be silent rather than an error, because the pipeline is
// already shutting down.
func TestAutoHangUpSkipped(t *testing.T) {
	tests := []struct {
		name     string
		cfg      Config
		sawStart bool // whether the start message (and so the call SID) arrived
	}{
		{"disabled", Config{AccountSID: "AC123", AuthToken: "token"}, true},
		{"no account SID", Config{AuthToken: "token", AutoHangUp: true}, true},
		{"no auth token", Config{AccountSID: "AC123", AutoHangUp: true}, true},
		{"no call SID", Config{AccountSID: "AC123", AuthToken: "token", AutoHangUp: true}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, got := hangupServer(t)
			cfg := tt.cfg
			cfg.HTTPClient = redirectToServer(srv)

			s := New(cfg)
			if tt.sawStart {
				if _, err := s.Deserialize([]byte(startMsg)); err != nil {
					t.Fatalf("deserialize start: %v", err)
				}
			}
			if _, err := s.Serialize(frames.NewEndFrame()); err != nil {
				t.Fatalf("serialize end: %v", err)
			}
			select {
			case <-got:
				t.Error("hang-up should not have been attempted")
			case <-time.After(200 * time.Millisecond):
			}
		})
	}
}

// TestNewDefaultsHTTPClient checks a caller that supplies no client still gets a
// usable one.
func TestNewDefaultsHTTPClient(t *testing.T) {
	if New(Config{}).http != http.DefaultClient {
		t.Error("a nil HTTPClient should fall back to http.DefaultClient")
	}
}
