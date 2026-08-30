package gladia

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/stt"
	errs "github.com/gojargo/jargo/utils/errors"
)

// Tests for the live session: Gladia is dialed in two stages, a REST call that
// settles the session and returns a socket URL, then the socket itself. Neither
// stage had any coverage, and a fault in either means the service never
// transcribes anything at all.

// liveSession is what the fake endpoint saw.
type liveSession struct {
	initQuery  url.Values
	initHeader http.Header
	initBody   map[string]any
	audio      chan string
	text       chan string
	// send carries messages the test wants the server to speak.
	send chan string
}

// liveServer stands up both stages: a session-init endpoint answering with the
// URL of a WebSocket endpoint on the same server.
func liveServer(t *testing.T, initStatus int) (initURL string, got *liveSession) {
	t.Helper()
	got = &liveSession{
		audio: make(chan string, 8),
		text:  make(chan string, 8),
		send:  make(chan string, 8),
	}

	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mux.HandleFunc("/v2/live", func(w http.ResponseWriter, r *http.Request) {
		got.initQuery = r.URL.Query()
		got.initHeader = r.Header.Clone()
		_ = json.NewDecoder(r.Body).Decode(&got.initBody)
		if initStatus != http.StatusOK {
			w.WriteHeader(initStatus)
			_, _ = w.Write([]byte(`{"message":"no"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/socket"
		_ = json.NewEncoder(w).Encode(map[string]string{"url": wsURL})
	})

	mux.HandleFunc("/socket", func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		go func() {
			for msg := range got.send {
				if err := c.Write(r.Context(), websocket.MessageText, []byte(msg)); err != nil {
					return
				}
			}
		}()
		for {
			typ, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			ch := got.text
			if typ == websocket.MessageBinary {
				ch = got.audio
			}
			select {
			case ch <- string(data):
			default:
			}
		}
	})

	return srv.URL + "/v2/live", got
}

// dialLive opens a session against the fake endpoint.
func dialLive(t *testing.T, cfg Config) (stt.Stream, *liveSession) {
	t.Helper()
	initURL, got := liveServer(t, http.StatusOK)
	cfg.URL = initURL
	cfg.APIKey = "test-key"
	c := &connector{cfg: withDefaults(cfg), http: &http.Client{}}

	s, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, got
}

// TestConnectSettlesTheSessionThenDialsIt checks both stages happen and that the
// session is settled with what the pipeline runs at, not with a guess: the audio
// shape is sent once at init and every chunk afterwards is read against it.
func TestConnectSettlesTheSessionThenDialsIt(t *testing.T) {
	_, got := dialLive(t, Config{Model: "solaria-1"})

	if key := got.initHeader.Get("x-gladia-key"); key != "test-key" {
		t.Errorf("init sent the key %q, want test-key", key)
	}
	if ct := got.initHeader.Get("Content-Type"); ct != "application/json" {
		t.Errorf("init Content-Type = %q, want application/json", ct)
	}
	if got.initBody["sample_rate"] != float64(16000) {
		t.Errorf("init sample_rate = %v, want the pipeline's 16000", got.initBody["sample_rate"])
	}
	if got.initBody["model"] != "solaria-1" {
		t.Errorf("init model = %v, want solaria-1", got.initBody["model"])
	}
}

// TestConnectPinsTheRegion checks the region reaches the init call as a query
// parameter, which is how the session is placed in one.
func TestConnectPinsTheRegion(t *testing.T) {
	_, got := dialLive(t, Config{Region: "eu-west"})
	if r := got.initQuery.Get("region"); r != "eu-west" {
		t.Errorf("init region = %q, want eu-west", r)
	}
}

// TestConnectReportsARefusedSession checks a session the server would not settle
// is reported rather than dialed: there is no socket URL to dial, and carrying on
// would fail later and less clearly.
func TestConnectReportsARefusedSession(t *testing.T) {
	initURL, _ := liveServer(t, http.StatusUnauthorized)
	c := &connector{
		cfg:  withDefaults(Config{URL: initURL, APIKey: "bad"}),
		http: &http.Client{},
	}
	_, err := c.Connect(t.Context(), 16000)
	// The status, not merely a failure: without the check the refusal body is
	// read as a session and the dial fails afterwards for an unrelated reason,
	// which tells a caller nothing about why the key was rejected.
	var status *errs.HTTPStatusError
	if !errors.As(err, &status) {
		t.Fatalf("Connect against a refused session = %v, want an HTTP status error", err)
	}
	if status.Status != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", status.Status, http.StatusUnauthorized)
	}
}

// TestSendWritesAudioAsBinary checks the audio goes out as binary frames, which
// is the only shape Gladia reads PCM from.
func TestSendWritesAudioAsBinary(t *testing.T) {
	s, got := dialLive(t, Config{})
	if err := s.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case audio := <-got.audio:
		if audio != string([]byte{1, 2, 3, 4}) { //nolint:gocritic // compares the bytes as sent
			t.Errorf("the server saw %q, want the PCM that was sent", audio)
		}
	case <-time.After(3 * time.Second):
		t.Error("no audio reached the server")
	}
}

// TestRecvReturnsATranscript checks a transcript message is mapped to a result,
// and that a message carrying nothing to say is skipped rather than returned as
// an empty transcript.
func TestRecvReturnsATranscript(t *testing.T) {
	s, got := dialLive(t, Config{})

	got.send <- `{"type":"transcript","data":{"is_final":false,"utterance":{"text":""}}}`
	got.send <- `{"type":"transcript","data":{"is_final":true,"utterance":{"text":"bonjour","language":"fr"}}}`

	results := recvOne(t, s)
	if len(results) != 1 {
		t.Fatalf("Recv returned %d results, want 1", len(results))
	}
	r := results[0]
	if r.Text != "bonjour" || !r.Final || !r.EndOfTurn || r.Language != "fr" {
		t.Errorf("result = %+v, want the final French transcript", r)
	}
}

// TestRecvSkipsUnreadableMessages checks a message the client cannot parse is
// stepped over rather than ending the session: one malformed frame must not cost
// the rest of the call.
func TestRecvSkipsUnreadableMessages(t *testing.T) {
	s, got := dialLive(t, Config{})

	got.send <- `not json at all`
	got.send <- `{"type":"transcript","data":{"is_final":true,"utterance":{"text":"still here"}}}`

	results := recvOne(t, s)
	if len(results) != 1 || results[0].Text != "still here" {
		t.Errorf("results = %+v, want the transcript that followed the bad message", results)
	}
}

// TestCloseStopsTheRecording checks the socket is not simply dropped: Gladia is
// told the recording has stopped, so it finalizes rather than timing the session
// out on its side.
func TestCloseStopsTheRecording(t *testing.T) {
	s, got := dialLive(t, Config{})
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case msg := <-got.text:
		if !strings.Contains(msg, "stop_recording") {
			t.Errorf("the server saw %q on close, want stop_recording", msg)
		}
	case <-time.After(3 * time.Second):
		t.Error("nothing was sent on close")
	}
}

// recvOne reads until a result arrives.
func recvOne(t *testing.T, s stt.Stream) []stt.Result {
	t.Helper()
	done := make(chan []stt.Result, 1)
	go func() {
		r, err := s.Recv()
		if err != nil {
			close(done)
			return
		}
		done <- r
	}()
	select {
	case r, ok := <-done:
		if !ok {
			t.Fatal("Recv failed")
		}
		return r
	case <-time.After(3 * time.Second):
		t.Fatal("Recv returned nothing")
		return nil
	}
}
