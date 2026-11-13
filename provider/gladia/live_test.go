package gladia

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/pipeline"
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
	// ended is signaled as each socket closes, once everything sent on it
	// has been read.
	ended chan struct{}
}

// liveServer stands up both stages: a session-init endpoint answering with the
// URL of a WebSocket endpoint on the same server.
func liveServer(t *testing.T, initStatus int) (initURL string, got *liveSession) {
	t.Helper()
	got = &liveSession{
		audio: make(chan string, 8),
		text:  make(chan string, 8),
		send:  make(chan string, 8),
		ended: make(chan struct{}, 8),
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
		defer func() {
			select {
			case got.ended <- struct{}{}:
			default:
			}
		}()
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

// TestStopStopsTheRecording checks the session is not simply dropped as the
// pipeline ends: Gladia is told the recording has stopped, so it finalizes
// rather than timing the session out on its side.
func TestStopStopsTheRecording(t *testing.T) {
	s, got := dialLive(t, Config{})
	st, ok := s.(stt.Stopper)
	if !ok {
		t.Fatal("the session cannot be stopped, so the recording is never stopped")
	}
	if err := st.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	select {
	case msg := <-got.text:
		if !strings.Contains(msg, "stop_recording") {
			t.Errorf("the server saw %q on stop, want stop_recording", msg)
		}
	case <-time.After(3 * time.Second):
		t.Error("nothing was sent on stop")
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

// captureReports routes the problems a session survives into a slice the test
// reads once Recv has moved past them.
func captureReports(t *testing.T, s stt.Stream) *[]string {
	t.Helper()
	st, ok := s.(*stream)
	if !ok {
		t.Fatalf("Connect returned %T, want *stream", s)
	}
	var reports []string
	st.report = func(_ context.Context, msg string) { reports = append(reports, msg) }
	return &reports
}

// TestRecvReportsARejectedAudioChunk checks a chunk Gladia rejected is reported
// rather than dropped without a word, and that the session carries on.
func TestRecvReportsARejectedAudioChunk(t *testing.T) {
	s, got := dialLive(t, Config{})
	reports := captureReports(t, s)

	got.send <- `{"type":"audio_chunk","acknowledged":true,"data":{"byte_range":[0,3200]}}`
	got.send <- `{"type":"audio_chunk","acknowledged":false,"error":{"message":"quota exceeded"},"data":null}`
	got.send <- `{"type":"transcript","data":{"is_final":true,"utterance":{"text":"still here"}}}`

	results := recvOne(t, s)
	if len(results) != 1 || results[0].Text != "still here" {
		t.Errorf("results = %+v, want the transcript that followed the rejection", results)
	}
	if len(*reports) != 1 || !strings.Contains((*reports)[0], "quota exceeded") {
		t.Errorf("reports = %q, want the one rejection with its message", *reports)
	}
}

// TestRecvReportsAFailedTranslation checks a failed translation, which carries
// an error and no payload, is reported and leaves the session running: it is
// the translation that failed, not the transcription.
func TestRecvReportsAFailedTranslation(t *testing.T) {
	s, got := dialLive(t, Config{})
	reports := captureReports(t, s)

	got.send <- `{"type":"translation","error":{"message":"translation addon failed"},"data":null}`
	got.send <- `{"type":"transcript","data":{"is_final":true,"utterance":{"text":"still here"}}}`

	results := recvOne(t, s)
	if len(results) != 1 || results[0].Text != "still here" {
		t.Errorf("results = %+v, want the transcript that followed the failure", results)
	}
	if len(*reports) != 1 || !strings.Contains((*reports)[0], "translation addon failed") {
		t.Errorf("reports = %q, want the one failure with its message", *reports)
	}
}

// TestRecvDoesNotReportASuccessfulTranslation checks a translation carrying no
// error is not mistaken for a failure.
func TestRecvDoesNotReportASuccessfulTranslation(t *testing.T) {
	s, got := dialLive(t, Config{})
	reports := captureReports(t, s)

	got.send <- `{"type":"translation","error":null,"data":{"original_language":"en",` +
		`"translated_utterance":{"language":"fr","text":"Bonjour"}}}`
	got.send <- `{"type":"transcript","data":{"is_final":true,"utterance":{"text":"hello"}}}`

	recvOne(t, s)
	if len(*reports) != 0 {
		t.Errorf("reports = %q, want none for a translation that succeeded", *reports)
	}
}

// Ported from upstream (test_stop_disconnects_once). Ending the pipeline closes
// the session once: the end frame and the cleanup after it both tear the
// service down, and the second must find nothing left to close, or Gladia is
// told twice that the recording stopped.
func TestStopDisconnectsOnce(t *testing.T) {
	stops := runToTheEnd(t, func(task *pipeline.Worker) { task.StopWhenDone() })
	if stops != 1 {
		t.Errorf("stop_recording sent %d times, want once", stops)
	}
}

// Ported from upstream (test_cancel_does_not_send_stop_recording). A canceled
// pipeline closes the session without telling Gladia the recording stopped.
func TestCancelDoesNotSendStopRecording(t *testing.T) {
	stops := runToTheEnd(t, func(task *pipeline.Worker) { task.Cancel(context.Background(), "test") })
	if stops != 0 {
		t.Errorf("stop_recording sent %d times on cancel, want never", stops)
	}
}

// runToTheEnd runs a Gladia service in a pipeline until its session is open,
// ends the pipeline with end, and reports how many times the server was told
// the recording stopped.
func runToTheEnd(t *testing.T, end func(*pipeline.Worker)) int {
	t.Helper()
	initURL, got := liveServer(t, http.StatusOK)
	svc := NewSTT(Config{APIKey: "test-key", URL: initURL, SampleRate: 16000})

	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	deadline := time.Now().Add(3 * time.Second)
	for !svc.Connected() {
		if time.Now().After(deadline) {
			t.Fatal("the session never opened")
		}
		time.Sleep(5 * time.Millisecond)
	}
	end(task)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not shut down")
	}

	// The server reads the socket in order, so once it has seen the socket
	// close it has seen everything sent before.
	select {
	case <-got.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never closed")
	}
	stops := 0
	for {
		select {
		case msg := <-got.text:
			if strings.Contains(msg, "stop_recording") {
				stops++
			}
			continue
		default:
		}
		return stops
	}
}
