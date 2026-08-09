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
}

// listenServer starts a fake Deepgram live endpoint that records what the client
// sends and never speaks first.
func listenServer(t *testing.T) (endpoint string, got *listenSession) {
	t.Helper()
	got = &listenSession{audio: make(chan []byte, 8), text: make(chan string, 8)}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.query = r.URL.Query()
		got.header = r.Header.Clone()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
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
// the session context is canceled before the stream is closed, because Close
// waits for the keepalive goroutine and that goroutine only stops when the
// context does.
func openSession(t *testing.T, c *connector) stt.Stream {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	stream, err := c.Connect(ctx, 16000)
	if err != nil {
		cancel()
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() {
		cancel()
		if err := stream.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return stream
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
