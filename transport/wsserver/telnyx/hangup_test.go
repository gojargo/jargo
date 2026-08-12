package telnyx

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// Tests for ending the Telnyx call when the pipeline finishes. Hanging up is a
// REST call rather than a message on the socket, so it is a side effect of
// serializing the frame that ends the run.

// hangupServer stands in for Telnyx's REST API and reports each hang-up it is
// sent.
func hangupServer(t *testing.T) (*httptest.Server, chan *http.Request) {
	t.Helper()
	got := make(chan *http.Request, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got <- r
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

// rewriteHost sends a request to the test server instead of the real host, since
// the endpoint is built from a package constant.
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

func redirectToServer(srv *httptest.Server) *http.Client {
	return &http.Client{Transport: rewriteHost{target: srv.URL, base: srv.Client().Transport}}
}

// started returns a serializer that has seen the start message, so it knows the
// call it would be hanging up.
func started(t *testing.T, cfg Config) *Serializer {
	t.Helper()
	s := New(cfg)
	start := `{"event":"start","stream_id":"stream-1",` +
		`"start":{"call_control_id":"cc-1","media_format":{"encoding":"PCMU"}}}`
	if _, err := s.Deserialize([]byte(start)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	return s
}

// awaitHangup waits for the REST call, failing if it does not arrive.
func awaitHangup(t *testing.T, got chan *http.Request) *http.Request {
	t.Helper()
	select {
	case req := <-got:
		return req
	case <-time.After(2 * time.Second):
		t.Fatal("no hang-up request arrived")
		return nil
	}
}

// TestSetupIsANoOp checks the serializer needs nothing from the StartFrame:
// Telnyx audio is always 8 kHz, so there is no rate to reconcile.
func TestSetupIsANoOp(t *testing.T) {
	s := New(Config{})
	if err := s.Setup(frames.NewStartFrame()); err != nil {
		t.Fatalf("Setup: %v", err)
	}
}

// TestAutoHangUp checks the REST hang-up: it targets the call control id learned
// from the start message and authorizes with the configured key.
func TestAutoHangUp(t *testing.T) {
	srv, got := hangupServer(t)
	s := started(t, Config{APIKey: "key-1", AutoHangUp: true, HTTPClient: redirectToServer(srv)})

	if _, err := s.Serialize(frames.NewEndFrame()); err != nil {
		t.Fatalf("serialize end: %v", err)
	}

	req := awaitHangup(t, got)
	if req.Method != http.MethodPost {
		t.Errorf("method = %s, want POST", req.Method)
	}
	if !strings.Contains(req.URL.Path, "cc-1") {
		t.Errorf("path = %s, want the call control id from the start message", req.URL.Path)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer key-1" {
		t.Errorf("Authorization = %q, want the configured key as a bearer token", got)
	}
}

// TestAutoHangUpOnCancel checks a canceled run hangs up too. A call the bot
// abandoned must not be left open any more than one it ended cleanly.
func TestAutoHangUpOnCancel(t *testing.T) {
	srv, got := hangupServer(t)
	s := started(t, Config{APIKey: "key-1", AutoHangUp: true, HTTPClient: redirectToServer(srv)})

	if _, err := s.Serialize(frames.NewCancelFrame()); err != nil {
		t.Fatalf("serialize cancel: %v", err)
	}
	awaitHangup(t, got)
}

// TestAutoHangUpFiresOnce checks the call is ended once however many ending
// frames arrive, since an EndFrame and a CancelFrame can both pass through.
func TestAutoHangUpFiresOnce(t *testing.T) {
	srv, got := hangupServer(t)
	s := started(t, Config{APIKey: "key-1", AutoHangUp: true, HTTPClient: redirectToServer(srv)})

	for range 3 {
		if _, err := s.Serialize(frames.NewEndFrame()); err != nil {
			t.Fatalf("serialize end: %v", err)
		}
	}
	awaitHangup(t, got)

	select {
	case <-got:
		t.Error("the call was hung up more than once")
	case <-time.After(300 * time.Millisecond):
	}
}

// TestNoHangUpWhenNotConfigured checks the cases that must not reach the API:
// the feature off, no key to authorize with, and no call to hang up.
func TestNoHangUpWhenNotConfigured(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		started bool
	}{
		{name: "auto hang-up off", cfg: Config{APIKey: "key-1"}, started: true},
		{name: "no API key", cfg: Config{AutoHangUp: true}, started: true},
		{name: "no start message, so no call", cfg: Config{APIKey: "key-1", AutoHangUp: true}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, got := hangupServer(t)
			cfg := tt.cfg
			cfg.HTTPClient = redirectToServer(srv)

			var s *Serializer
			if tt.started {
				s = started(t, cfg)
			} else {
				s = New(cfg)
			}

			if _, err := s.Serialize(frames.NewEndFrame()); err != nil {
				t.Fatalf("serialize end: %v", err)
			}
			select {
			case <-got:
				t.Error("a hang-up was sent")
			case <-time.After(300 * time.Millisecond):
			}
		})
	}
}

// TestHangUpSurvivesAFailingAPI checks a REST call that fails is logged rather
// than propagated: the pipeline is already ending, and the socket closing ends
// the call anyway.
func TestHangUpSurvivesAFailingAPI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	s := started(t, Config{APIKey: "key-1", AutoHangUp: true, HTTPClient: redirectToServer(srv)})
	if _, err := s.Serialize(frames.NewEndFrame()); err != nil {
		t.Fatalf("serialize end: %v", err)
	}
	// The hang-up runs on its own goroutine; give it time to fail quietly.
	time.Sleep(200 * time.Millisecond)
}

// TestNewDefaultsHTTPClient checks a caller that supplies no client still gets
// one, so the hang-up does not panic on a nil client.
func TestNewDefaultsHTTPClient(t *testing.T) {
	if s := New(Config{}); s.http == nil {
		t.Fatal("New left the HTTP client nil")
	}
}
