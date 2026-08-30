package assemblyai

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
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

// TestCloseTerminatesTheSession checks the socket is not simply dropped:
// AssemblyAI is told the session is over so it finalizes rather than waiting.
func TestCloseTerminatesTheSession(t *testing.T) {
	s, got := dialAAI(t, Config{})
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case msg := <-got.text:
		if !strings.Contains(msg, "Terminate") {
			t.Errorf("the server saw %q on close, want a Terminate", msg)
		}
	case <-time.After(3 * time.Second):
		t.Error("nothing was sent on close")
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
