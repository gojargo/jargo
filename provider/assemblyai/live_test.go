package assemblyai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/stt"
)

// Tests for the live session. AssemblyAI is configured entirely in the dial: the
// key travels as a header and every setting as a query parameter, so a fault
// there is a session that never transcribes. None of it had coverage.

// aaiSession is what the fake endpoint saw and what it will say.
type aaiSession struct {
	query  url.Values
	header http.Header
	audio  chan []byte
	text   chan string
	send   chan string
}

// aaiServer stands up a fake AssemblyAI streaming endpoint.
func aaiServer(t *testing.T) (endpoint string, got *aaiSession) {
	t.Helper()
	got = &aaiSession{
		audio: make(chan []byte, 8),
		text:  make(chan string, 8),
		send:  make(chan string, 8),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.query = r.URL.Query()
		got.header = r.Header.Clone()
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
			if typ == websocket.MessageBinary {
				select {
				case got.audio <- data:
				default:
				}
				continue
			}
			select {
			case got.text <- string(data):
			default:
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http"), got
}

// dialAAI opens a session against the fake endpoint.
func dialAAI(t *testing.T, cfg Config) (stt.Stream, *aaiSession) {
	t.Helper()
	endpoint, got := aaiServer(t)
	cfg.BaseURL = endpoint
	if cfg.APIKey == "" {
		cfg.APIKey = "test-key"
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	c := &connector{cfg: cfg}

	s, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, got
}

// TestConnectCarriesTheKeyAndTheAudioShape checks the dial says who is calling
// and what the audio is: both travel in the handshake and there is no second
// chance to send them.
func TestConnectCarriesTheKeyAndTheAudioShape(t *testing.T) {
	_, got := dialAAI(t, Config{})

	if key := got.header.Get("Authorization"); key != "test-key" {
		t.Errorf("dial sent the key %q, want test-key", key)
	}
	if rate := got.query.Get("sample_rate"); rate != "16000" {
		t.Errorf("dial sample_rate = %q, want the pipeline's 16000", rate)
	}
	if enc := got.query.Get("encoding"); enc != defaultEncoding {
		t.Errorf("dial encoding = %q, want %q", enc, defaultEncoding)
	}
}

// TestConnectCarriesTheU3ProSettings checks the settings gated on a U3 Pro model
// reach the dial when one is named. They are the reason to choose that model, so
// a session that dropped them would look like it was working and quietly not be.
func TestConnectCarriesTheU3ProSettings(t *testing.T) {
	_, got := dialAAI(t, Config{
		Model:         "universal-3-5-pro",
		LanguageCodes: []language.Language{language.EnglishUS, language.FrenchFR},
	})

	if model := got.query.Get("speech_model"); model != "universal-3-5-pro" {
		t.Errorf("dial speech_model = %q, want universal-3-5-pro", model)
	}
	if codes := got.query.Get("language_codes"); !strings.Contains(codes, "en") {
		t.Errorf("dial language_codes = %q, want the codes that were configured", codes)
	}
}

// TestSendWritesAudioAsBinary checks audio goes out as binary frames.
func TestSendWritesAudioAsBinary(t *testing.T) {
	s, got := dialAAI(t, Config{})
	if err := s.Send([]byte{4, 5, 6}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case audio := <-got.audio:
		if string(audio) != string([]byte{4, 5, 6}) {
			t.Errorf("the server saw %v, want the PCM that was sent", audio)
		}
	case <-time.After(3 * time.Second):
		t.Error("no audio reached the server")
	}
}

// TestRecvTreatsAnUnformattedTurnAsInterim checks the rule that decides when a
// turn is done: AssemblyAI ends a turn before it has formatted it, and treating
// that as final would commit the unpunctuated text and then commit it again.
func TestRecvTreatsAnUnformattedTurnAsInterim(t *testing.T) {
	s, got := dialAAI(t, Config{})

	got.send <- `{"type":"Turn","transcript":"hello there","end_of_turn":true,"turn_is_formatted":false}`
	r := recvAAI(t, s)
	if len(r) != 1 || r[0].Final || r[0].EndOfTurn {
		t.Fatalf("an ended but unformatted turn = %+v, want it interim", r)
	}

	got.send <- `{"type":"Turn","transcript":"Hello there.","end_of_turn":true,"turn_is_formatted":true}`
	r = recvAAI(t, s)
	if len(r) != 1 || !r[0].Final || !r[0].EndOfTurn || r[0].Text != "Hello there." {
		t.Errorf("the formatted turn = %+v, want it final", r)
	}
}

// TestRecvSkipsWhatCarriesNoTranscript checks the messages that are not a turn,
// and turns with nothing in them, are stepped over rather than surfacing as
// empty transcripts.
func TestRecvSkipsWhatCarriesNoTranscript(t *testing.T) {
	s, got := dialAAI(t, Config{})

	got.send <- `not json`
	got.send <- `{"type":"Begin","id":"session-1"}`
	got.send <- `{"type":"Turn","transcript":""}`
	got.send <- `{"type":"Turn","transcript":"finally"}`

	r := recvAAI(t, s)
	if len(r) != 1 || r[0].Text != "finally" {
		t.Errorf("results = %+v, want only the turn that carried a transcript", r)
	}
}

// TestDrainTerminatesTheSession checks the socket is not simply dropped when the
// pipeline stops: AssemblyAI is told the session is over so it finalizes rather
// than waiting, and the Termination acknowledging it is waited for.
func TestDrainTerminatesTheSession(t *testing.T) {
	s, got := dialAAI(t, Config{})
	drainer, ok := s.(stt.Drainer)
	if !ok {
		t.Fatal("the session cannot be drained")
	}
	window, err := drainer.Drain()
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if window != terminationTimeout {
		t.Errorf("drain window = %v, want %v for the Termination", window, terminationTimeout)
	}
	select {
	case msg := <-got.text:
		if !strings.Contains(msg, "Terminate") {
			t.Errorf("the server saw %q on drain, want a Terminate", msg)
		}
	case <-time.After(3 * time.Second):
		t.Error("nothing was sent on drain")
	}

	got.send <- `{"type":"Turn","transcript":"last words","end_of_turn":true,"turn_is_formatted":true}`
	got.send <- `{"type":"Termination","audio_duration_seconds":1,"session_duration_seconds":2}`
	if r := recvAAI(t, s); len(r) != 1 || r[0].Text != "last words" {
		t.Errorf("results = %+v, want the turn that arrived before the Termination", r)
	}
	if _, err := recvAAIError(t, s); !errors.Is(err, io.EOF) {
		t.Errorf("Recv after the Termination = %v, want io.EOF", err)
	}
}

// Ported from upstream (test_disconnect_terminates_an_open_session_before_closing).
// Ending the pipeline sends one Terminate while the socket is still open, and
// the teardown ends on the Termination that acknowledges it rather than
// waiting out its window.
func TestStopTerminatesTheSessionBeforeClosing(t *testing.T) {
	endpoint, got := terminatingServer(t)
	svc := NewSTT(Config{APIKey: "test-key", BaseURL: endpoint, SampleRate: 16000})
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	waitConnected(t, svc)

	started := time.Now()
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not shut down")
	}
	if took := time.Since(started); took >= terminationTimeout {
		t.Errorf("the teardown took %v, want it to end on the Termination", took)
	}
	select {
	case <-got.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never closed")
	}
	if n := got.terminates.Load(); n != 1 {
		t.Errorf("Terminate messages = %d, want 1", n)
	}
}

// A canceled pipeline terminates the session too, as every session the
// service ends deliberately is: one Terminate, acknowledged before the close.
func TestCancelTerminatesTheSession(t *testing.T) {
	endpoint, got := terminatingServer(t)
	svc := NewSTT(Config{APIKey: "test-key", BaseURL: endpoint, SampleRate: 16000})
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	waitConnected(t, svc)

	task.Cancel(context.Background(), "test")
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not shut down")
	}
	select {
	case <-got.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never closed")
	}
	if n := got.terminates.Load(); n != 1 {
		t.Errorf("Terminate messages = %d, want 1", n)
	}
}

// TestCloseSendsNoTerminate checks a session closed to replace it after it
// dropped is just closed: Terminate belongs to the service ending a session.
func TestCloseSendsNoTerminate(t *testing.T) {
	s, got := dialAAI(t, Config{})
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case msg := <-got.text:
		t.Errorf("the server saw %q on close, want nothing", msg)
	default:
	}
}

// terminated is what a terminatingServer saw.
type terminated struct {
	terminates atomic.Int32
	// ended receives once for each session the client has closed.
	ended chan struct{}
}

// terminatingServer is a fake AssemblyAI endpoint that answers a Terminate
// with Termination, and counts the Terminates it receives.
func terminatingServer(t *testing.T) (string, *terminated) {
	t.Helper()
	got := &terminated{ended: make(chan struct{}, 4)}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		defer func() { got.ended <- struct{}{} }()
		for {
			_, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if !strings.Contains(string(data), `"Terminate"`) {
				continue
			}
			got.terminates.Add(1)
			reply := `{"type":"Termination","audio_duration_seconds":1,"session_duration_seconds":2}`
			_ = c.Write(r.Context(), websocket.MessageText, []byte(reply))
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http"), got
}

// waitConnected polls until the service holds its session.
func waitConnected(t *testing.T, svc *stt.StreamService) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !svc.Connected() {
		if time.Now().After(deadline) {
			t.Fatal("the service never held its session")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// recvAAIError reads one batch, returning the error that ended the session,
// bounded like recvAAI.
func recvAAIError(t *testing.T, s stt.Stream) ([]stt.Result, error) {
	t.Helper()
	type outcome struct {
		results []stt.Result
		err     error
	}
	done := make(chan outcome, 1)
	go func() {
		r, err := s.Recv()
		done <- outcome{r, err}
	}()
	select {
	case o := <-done:
		return o.results, o.err
	case <-time.After(3 * time.Second):
		t.Fatal("Recv returned nothing")
		return nil, nil
	}
}

// Ported from upstream (test_explicit_sample_rate_overrides_setup). A rate
// configured on the service wins over the pipeline's input rate, and it is the
// one the session is opened at.
func TestExplicitSampleRateOverridesSetup(t *testing.T) {
	endpoint, got := aaiServer(t)
	svc := NewSTT(Config{APIKey: "test-key", BaseURL: endpoint, SampleRate: 16000})
	ctx := context.Background()
	if err := svc.Setup(ctx, processor.Setup{AudioInSampleRate: 8000, Clock: clock.NewSystem()}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(func() { _ = svc.Cleanup(ctx) })

	if !svc.Connected() {
		t.Fatal("Setup did not open a session")
	}
	if rate := got.query.Get("sample_rate"); rate != "16000" {
		t.Errorf("dial sample_rate = %q, want the configured 16000 over the pipeline's 8000", rate)
	}
}

// Ported from upstream (test_termination_acknowledges_without_ending_pipeline).
// AssemblyAI sends Termination in reply to the Terminate a closing session
// sends, including on a reopen a settings change asked for, so it is an
// acknowledgment: it surfaces nothing and does not end the read.
func TestTerminationAcknowledgesWithoutEndingTheSession(t *testing.T) {
	s, got := dialAAI(t, Config{})

	got.send <- `{"type":"Termination","audio_duration_seconds":1,"session_duration_seconds":2}`
	got.send <- `{"type":"Turn","transcript":"still here"}`

	r := recvAAI(t, s)
	if len(r) != 1 || r[0].Text != "still here" {
		t.Errorf("results = %+v, want only the turn after the Termination", r)
	}
}

// recvAAI reads one batch of results, bounded so a regression fails rather than
// hanging the suite.
func recvAAI(t *testing.T, s stt.Stream) []stt.Result {
	t.Helper()
	type outcome struct {
		results []stt.Result
		err     error
	}
	done := make(chan outcome, 1)
	go func() {
		r, err := s.Recv()
		done <- outcome{r, err}
	}()
	select {
	case o := <-done:
		if o.err != nil {
			t.Fatalf("Recv: %v", o.err)
		}
		return o.results
	case <-time.After(3 * time.Second):
		t.Fatal("Recv returned nothing")
		return nil
	}
}
