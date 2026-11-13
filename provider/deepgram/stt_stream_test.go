package deepgram

import (
	"context"
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

// listenSession is what the fake live-transcription endpoint saw.
type listenSession struct {
	query  url.Values
	header http.Header
	audio  chan []byte
	text   chan string
	// ended is closed once the connection is over, from the server's side.
	ended chan struct{}
}

// listenServer starts a fake Deepgram live endpoint that records what the client
// sends and never speaks first.
func listenServer(t *testing.T) (endpoint string, got *listenSession) {
	t.Helper()
	got = &listenSession{audio: make(chan []byte, 8), text: make(chan string, 8), ended: make(chan struct{})}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.query = r.URL.Query()
		got.header = r.Header.Clone()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer close(got.ended)
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
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
	// The base URL is handed over with its scheme, and the derivation pairs
	// http:// with ws:// so the fake endpoint is dialed insecurely.
	return srv.URL, got
}

// sttConfigFor is a config pointed at endpoint with the defaults NewSTT applies.
func sttConfigFor(endpoint string) Config {
	return Config{
		APIKey:   "test-key",
		BaseURL:  endpoint,
		Model:    defaultSTTModel,
		Language: language.EnglishUS,
		Encoding: defaultEncoding,
		Channels: defaultChannels,
	}
}

// openSession dials a session and returns it with the teardown the base does:
// the stream is closed while its context is still live, so what it sends as it
// closes reaches the server, and the context is canceled after.
func openSession(t *testing.T, c *connector) stt.Stream {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	stream, err := c.Connect(ctx, 16000)
	if err != nil {
		cancel()
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
		cancel()
	})
	return stream
}

// TestCloseTellsDeepgramTheStreamIsOver checks closing a session sends
// CloseStream, on a connection still up for it, and ends the connection.
func TestCloseTellsDeepgramTheStreamIsOver(t *testing.T) {
	endpoint, got := listenServer(t)
	cfg := sttConfigFor(endpoint)
	c := &connector{cfg: cfg, live: newSettings(cfg)}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	stream, err := c.Connect(ctx, 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	if err := stream.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}

	select {
	case msg := <-got.text:
		if msg != closeStreamMessage {
			t.Errorf("sent %q, want CloseStream", msg)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("CloseStream never reached the server")
	}
	select {
	case <-got.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the connection is still open after Close")
	}
}

// TestACloseStuckBehindASendStillClosesTheConnection checks a teardown that
// cannot get CloseStream out, because a send is stuck on a full write buffer,
// does not leave the connection open: the socket is dropped once the wait for
// the message runs out.
func TestACloseStuckBehindASendStillClosesTheConnection(t *testing.T) {
	endpoint, got := listenServer(t)
	cfg := sttConfigFor(endpoint)
	c := &connector{cfg: cfg, live: newSettings(cfg), closeStreamTimeout: 100 * time.Millisecond}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	opened, err := c.Connect(ctx, 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	st, ok := opened.(*stream)
	if !ok {
		t.Fatalf("Connect returned a %T, want a listen session", opened)
	}

	// A send waiting on a full write buffer holds the session's writes.
	st.writeMu.Lock()
	closed := make(chan struct{})
	go func() {
		_ = st.Close()
		close(closed)
	}()
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("closing the session waited on the stuck send")
	}
	select {
	case <-got.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the connection is still open after Close")
	}
	st.writeMu.Unlock()
}

// TestConnectDescribesTheSession checks the session is dialed with Deepgram's
// token scheme and describes the audio it is about to receive. Deepgram takes
// all of it as query parameters, so getting it wrong transcribes the wrong
// thing rather than failing.
func TestConnectDescribesTheSession(t *testing.T) {
	endpoint, got := listenServer(t)
	cfg := sttConfigFor(endpoint)
	c := &connector{cfg: cfg, live: newSettings(cfg)}

	openSession(t, c)

	if h := got.header.Get("Authorization"); h != "Token test-key" {
		t.Errorf("Authorization = %q, want Deepgram's Token scheme", h)
	}
	if q := got.query.Get("model"); q != defaultSTTModel {
		t.Errorf("model = %q, want %q", q, defaultSTTModel)
	}
	if q := got.query.Get("sample_rate"); q != "16000" {
		t.Errorf("sample_rate = %q, want the rate the transport runs at", q)
	}
	if q := got.query.Get("encoding"); q != defaultEncoding {
		t.Errorf("encoding = %q, want %q", q, defaultEncoding)
	}
}

// TestConnectRefused checks a session that cannot be opened is reported.
func TestConnectRefused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	cfg := sttConfigFor("ws" + strings.TrimPrefix(srv.URL, "http"))
	c := &connector{cfg: cfg, live: newSettings(cfg)}
	if _, err := c.Connect(t.Context(), 16000); err == nil {
		t.Fatal("Connect on a refused session = nil, want an error")
	}
}

// TestStreamSendAndFinalize checks the two things the session writes: audio as a
// binary frame, and the Finalize that asks Deepgram to transcribe what it is
// holding rather than wait on its own endpointing. The answer to that Finalize
// is what closes an utterance, so the message has to go out as Deepgram spells
// it.
func TestStreamSendAndFinalize(t *testing.T) {
	endpoint, got := listenServer(t)
	cfg := sttConfigFor(endpoint)
	c := &connector{cfg: cfg, live: newSettings(cfg)}

	stream := openSession(t, c)

	pcm := []byte{1, 2, 3, 4}
	if err := stream.Send(pcm); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case sent := <-got.audio:
		if string(sent) != string(pcm) {
			t.Errorf("audio = % x, want % x", sent, pcm)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the endpoint never received the audio")
	}

	finalizer, ok := stream.(interface{ Finalize() error })
	if !ok {
		t.Fatal("the stream does not offer Finalize, so a VAD stop cannot close an utterance")
	}
	if err := finalizer.Finalize(); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	select {
	case msg := <-got.text:
		if msg != `{"type":"Finalize"}` {
			t.Errorf("finalize message = %q, want Deepgram's own spelling", msg)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the endpoint was never asked to finalize")
	}
}
