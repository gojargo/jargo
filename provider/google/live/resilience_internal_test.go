package live

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
)

// These tests are not ports of upstream tests: upstream has none for session
// resumption, reconnection or send errors.

// droppingSession is a Live server that acknowledges each session's setup,
// hands out a resumption handle, and then drops the first drops sessions.
type droppingSession struct {
	*httptest.Server
	mu     sync.Mutex
	setups []map[string]any
}

func newDroppingSession(t *testing.T, drops int) *droppingSession {
	t.Helper()
	d := &droppingSession{}
	d.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		c, err := websocket.Accept(w, req, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		_, data, err := c.Read(req.Context())
		if err != nil {
			return
		}
		var msg map[string]any
		_ = json.Unmarshal(data, &msg)
		setup, _ := msg["setup"].(map[string]any)
		d.mu.Lock()
		d.setups = append(d.setups, setup)
		n := len(d.setups)
		d.mu.Unlock()
		for _, m := range []map[string]any{
			{"setupComplete": map[string]any{}},
			{"sessionResumptionUpdate": map[string]any{"newHandle": "handle-1", "resumable": true}},
		} {
			raw, _ := json.Marshal(m)
			if c.Write(req.Context(), websocket.MessageText, raw) != nil {
				return
			}
		}
		if n <= drops {
			return // dropped
		}
		for {
			if _, _, err := c.Read(req.Context()); err != nil {
				return
			}
		}
	}))
	t.Cleanup(d.Close)
	return d
}

func (d *droppingSession) setupCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.setups)
}

func (d *droppingSession) setup(i int) map[string]any {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.setups[i]
}

// startedOn runs a service against url in a pipeline, with sink after it.
func startedOn(t *testing.T, url string) (*Service, *capture) {
	t.Helper()
	s, sink, _ := newTestServiceBothWays(t)
	s.cfg.BaseURL = url
	s.connector = apiKeyConnector{baseURL: url, apiKey: "k"}
	if err := s.connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(s.stop)
	return s, sink
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(3 * time.Second); !cond(); {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// The setup always asks for resumption, so the server hands out handles.
func TestTheSetupAsksForResumption(t *testing.T) {
	setup := setupOf(t, New(Config{APIKey: "k"}))
	if r, ok := setup["sessionResumption"].(map[string]any); !ok || len(r) != 0 {
		t.Errorf("sessionResumption = %v, want an empty request", setup["sessionResumption"])
	}
}

// A lost session is opened again, resuming from the last handle the server
// gave, and the resumed session takes input at once.
func TestALostSessionIsResumed(t *testing.T) {
	srv := newDroppingSession(t, 1)
	s, _ := startedOn(t, wsURL(&recordingSession{Server: srv.Server}))

	waitFor(t, "the session to be opened again", func() bool { return srv.setupCount() >= 2 })
	resumption, _ := srv.setup(1)["sessionResumption"].(map[string]any)
	if resumption["handle"] != "handle-1" {
		t.Errorf("sessionResumption = %v, want the handle the server gave", resumption)
	}
	waitFor(t, "the resumed session to take input", s.readyForInput.Load)
}

// After maxConsecutiveFailures lost sessions in a row, the service stops
// reopening and reports an error.
func TestRepeatedLossesAreReportedAndNotRetried(t *testing.T) {
	srv := newDroppingSession(t, 10)
	s, _, source := newTestServiceBothWays(t)
	url := wsURL(&recordingSession{Server: srv.Server})
	s.connector = apiKeyConnector{baseURL: url, apiKey: "k"}
	if err := s.connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(s.stop)

	waitFor(t, "the error to be reported", func() bool { return hasName(source.names(), "ErrorFrame") })
	time.Sleep(200 * time.Millisecond)
	if n := srv.setupCount(); n != maxConsecutiveFailures {
		t.Errorf("opened %d sessions, want %d before giving up", n, maxConsecutiveFailures)
	}
}

// A session closed by the service itself is not reopened.
func TestAStoppedServiceDoesNotReopen(t *testing.T) {
	srv := newDroppingSession(t, 0)
	s, _ := startedOn(t, wsURL(&recordingSession{Server: srv.Server}))
	waitFor(t, "the session to be ready", s.ready.Load)

	s.stop()
	time.Sleep(200 * time.Millisecond)
	if n := srv.setupCount(); n != 1 {
		t.Errorf("opened %d sessions, want the one", n)
	}
}

// A failed send on a ready session is reported as an error; one on a session
// not yet ready is expected and ignored.
func TestASendErrorIsReported(t *testing.T) {
	s, _, source := newTestServiceBothWays(t)

	s.handleSendError(context.Background(), errNotGenerator)
	if hasName(source.names(), "ErrorFrame") {
		t.Fatal("an error was reported for a session not yet ready")
	}

	s.ready.Store(true)
	s.handleSendError(context.Background(), errNotGenerator)
	var found bool
	for _, f := range source.frames() {
		if e, ok := f.(*frames.ErrorFrame); ok && !e.Fatal {
			found = true
		}
	}
	if !found {
		t.Error("a send error on a ready session was not reported")
	}
}
