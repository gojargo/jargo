package plivo

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Tests for ending the Plivo call when the pipeline finishes. Hanging up is a
// REST call rather than a message on the socket, so it is a side effect of
// serializing the frame that ends the run.

// hangupServer stands in for Plivo's REST API and reports each hang-up it is
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
	start := `{"event":"start","start":{"streamId":"stream-1","callId":"call-1"}}`
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

// TestSetupRefusesAHangUpItCannotAuthorize checks a configuration that is to end
// the call but names no credentials is refused when the pipeline starts, rather
// than running a whole conversation and then quietly leaving the leg up.
func TestSetupRefusesAHangUpItCannotAuthorize(t *testing.T) {
	s := New(Config{})
	if err := s.Setup(processor.Setup{}); !errors.Is(err, ErrHangUpCredentials) {
		t.Fatalf("Setup with no credentials = %v, want ErrHangUpCredentials", err)
	}
}

// TestSetupNeedsNothingElseFromTheStartFrame checks the serializer asks for the
// credentials only when it is going to use them, and needs nothing else:
// Plivo audio is always 8 kHz, so there is no rate to reconcile.
func TestSetupNeedsNothingElseFromTheStartFrame(t *testing.T) {
	off := false
	s := New(Config{AutoHangUp: &off})
	if err := s.Setup(processor.Setup{}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
}

// TestAutoHangUp checks the REST hang-up: it deletes the call learned from the
// start message and authorizes with the configured credentials.
func TestAutoHangUp(t *testing.T) {
	srv, got := hangupServer(t)
	s := started(t, creds(redirectToServer(srv)))

	if _, err := s.Serialize(frames.NewEndFrame()); err != nil {
		t.Fatalf("serialize end: %v", err)
	}

	req := awaitHangup(t, got)
	if req.Method != http.MethodDelete {
		t.Errorf("method = %s, want DELETE", req.Method)
	}
	if !strings.Contains(req.URL.Path, "call-1") {
		t.Errorf("path = %s, want the call id from the start message", req.URL.Path)
	}
	if !strings.Contains(req.URL.Path, "auth-1") {
		t.Errorf("path = %s, want the auth id in the endpoint", req.URL.Path)
	}
	id, token, ok := req.BasicAuth()
	if !ok || id != "auth-1" || token != "token-1" {
		t.Errorf("basic auth = (%q, %q, %v), want the configured credentials", id, token, ok)
	}
}

// creds is the configuration a hang-up needs, pointed at the test server.
func creds(c *http.Client) Config {
	return Config{AuthID: "auth-1", AuthToken: "token-1", HTTPClient: c}
}

// TestAutoHangUpOnCancel checks a canceled run hangs up too. A call the bot
// abandoned must not be left open any more than one it ended cleanly.
func TestAutoHangUpOnCancel(t *testing.T) {
	srv, got := hangupServer(t)
	s := started(t, creds(redirectToServer(srv)))

	if _, err := s.Serialize(frames.NewCancelFrame()); err != nil {
		t.Fatalf("serialize cancel: %v", err)
	}
	awaitHangup(t, got)
}

// TestAutoHangUpFiresOnce checks the call is ended once however many ending
// frames arrive, since an EndFrame and a CancelFrame can both pass through.
func TestAutoHangUpFiresOnce(t *testing.T) {
	srv, got := hangupServer(t)
	s := started(t, creds(redirectToServer(srv)))

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
	off := false
	tests := []struct {
		name    string
		cfg     Config
		started bool
	}{
		{name: "auto hang-up off", cfg: Config{AuthID: "auth-1", AuthToken: "token-1", AutoHangUp: &off}, started: true},
		{name: "no auth id", cfg: Config{AuthToken: "token-1"}, started: true},
		{name: "no auth token", cfg: Config{AuthID: "auth-1"}, started: true},
		{name: "no start message, so no call", cfg: Config{AuthID: "auth-1", AuthToken: "token-1"}},
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

	s := started(t, creds(redirectToServer(srv)))
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
