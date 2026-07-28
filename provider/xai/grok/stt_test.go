package grok

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
)

// TestSTTConfigValidate pins which STTConfig fields the provider requires, and
// the multichannel rule the server would otherwise reject at connect time.
func TestSTTConfigValidate(t *testing.T) {
	yes, no := true, false
	two, one := 2, 1
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: STTConfig{APIKey: "k"}, Valid: true},
		{Name: "supported encoding", Cfg: STTConfig{APIKey: "k", Encoding: "mulaw"}, Valid: true},
		{Name: "unsupported encoding", Cfg: STTConfig{APIKey: "k", Encoding: "opus"}, Valid: false},
		{Name: "supported sample rate", Cfg: STTConfig{APIKey: "k", SampleRate: 24000}, Valid: true},
		{Name: "unsupported sample rate", Cfg: STTConfig{APIKey: "k", SampleRate: 12345}, Valid: false},
		{Name: "endpointing in range", Cfg: STTConfig{APIKey: "k", Endpointing: &two}, Valid: true},
		{Name: "multichannel with channels", Cfg: STTConfig{APIKey: "k", Multichannel: &yes, Channels: &two}, Valid: true},
		{Name: "multichannel without channels", Cfg: STTConfig{APIKey: "k", Multichannel: &yes}, Valid: false},
		{Name: "multichannel off without channels", Cfg: STTConfig{APIKey: "k", Multichannel: &no}, Valid: true},
		{Name: "too few channels", Cfg: STTConfig{APIKey: "k", Multichannel: &yes, Channels: &one}, Valid: false},
	})
}

// TestNewSTT checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewSTT(t *testing.T) {
	providertest.Service(t, "XAISTT", NewSTT(STTConfig{APIKey: "k"}))
}

// TestSTTEndpoint checks the session configuration that travels as query
// parameters: the transport's sample rate, the base language code, and the
// optional flags only when they are set.
func TestSTTEndpoint(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{APIKey: "k", URL: defaultSTTURL, Encoding: defaultSTTEncoding}}
		q := parseQuery(t, c.endpoint(16000))

		if got := q.Get("sample_rate"); got != "16000" {
			t.Errorf("sample_rate = %q, want the transport rate 16000", got)
		}
		if got := q.Get("encoding"); got != "pcm" {
			t.Errorf("encoding = %q, want pcm", got)
		}
		if got := q.Get("interim_results"); got != "true" {
			t.Errorf("interim_results = %q, want true by default", got)
		}
		for _, key := range []string{"language", "endpointing", "multichannel", "channels", "diarize"} {
			if q.Has(key) {
				t.Errorf("%s = %q, want it omitted when unset", key, q.Get(key))
			}
		}
	})

	t.Run("regional language sends its base code", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{APIKey: "k", URL: defaultSTTURL, Language: language.FrenchCA}}
		if got := parseQuery(t, c.endpoint(16000)).Get("language"); got != "fr" {
			t.Errorf("language = %q, want the base code fr", got)
		}
	})

	t.Run("optional flags", func(t *testing.T) {
		off, on := false, true
		ms, ch := 400, 2
		c := &sttConnector{cfg: STTConfig{
			APIKey:         "k",
			URL:            defaultSTTURL,
			InterimResults: &off,
			Endpointing:    &ms,
			Multichannel:   &on,
			Channels:       &ch,
			Diarize:        &on,
		}}
		q := parseQuery(t, c.endpoint(16000))

		want := map[string]string{
			"interim_results": "false",
			"endpointing":     "400",
			"multichannel":    "true",
			"channels":        "2",
			"diarize":         "true",
		}
		for key, val := range want {
			if got := q.Get(key); got != val {
				t.Errorf("%s = %q, want %q", key, got, val)
			}
		}
	})
}

// sttSession is a fake xAI STT endpoint: it replays a scripted event sequence,
// then records every frame the client sends.
type sttSession struct {
	url       string
	handshake func() http.Header
	received  <-chan sttFrame
}

// sttFrame is one WebSocket frame the fake endpoint received.
type sttFrame struct {
	typ  websocket.MessageType
	data []byte
}

// newSTTSession starts a fake endpoint that writes each of events as a JSON text
// frame before it begins reading.
func newSTTSession(t *testing.T, events []map[string]any) sttSession {
	t.Helper()
	received := make(chan sttFrame, 8)
	var mu sync.Mutex
	var handshake http.Header

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		handshake = r.Header.Clone()
		mu.Unlock()

		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		for _, ev := range events {
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			typ, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			received <- sttFrame{typ: typ, data: data}
		}
	}))
	t.Cleanup(srv.Close)

	return sttSession{
		url: wsURL(srv.URL),
		handshake: func() http.Header {
			mu.Lock()
			defer mu.Unlock()
			return handshake
		},
		received: received,
	}
}

// transcriptEvents is the scripted turn the fake endpoint replays: the session
// acknowledgement, an interim, a finalized transcript mid-turn, then one that
// ends the turn.
func transcriptEvents() []map[string]any {
	return []map[string]any{
		{"type": "transcript.created"},
		{"type": "transcript.partial", "text": "hel", "is_final": false},
		{"type": "transcript.partial", "text": "hello", "is_final": true, "speech_final": false},
		{"type": "transcript.partial", "text": "hello there", "is_final": true, "speech_final": true},
	}
}

// TestSTTHandshake checks the credentials and client identity reach the server.
func TestSTTHandshake(t *testing.T) {
	session := newSTTSession(t, nil)
	conn := &sttConnector{cfg: STTConfig{APIKey: "test-key", URL: session.url, Encoding: defaultSTTEncoding}}

	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	h := session.handshake()
	if got := h.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", got)
	}
	if got := h.Get("User-Agent"); got != sttUserAgent {
		t.Errorf("User-Agent = %q, want %q", got, sttUserAgent)
	}
}

// TestSTTRecvTranscripts checks the server's partial and endpointed results map
// onto interim transcriptions, finalized ones, and the end of the user's turn.
func TestSTTRecvTranscripts(t *testing.T) {
	session := newSTTSession(t, transcriptEvents())
	conn := &sttConnector{cfg: STTConfig{
		APIKey:   "k",
		URL:      session.url,
		Encoding: defaultSTTEncoding,
		Language: language.English,
	}}
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	want := []struct {
		desc      string
		text      string
		final     bool
		endOfTurn bool
	}{
		{"interim", "hel", false, false},
		{"finalized mid-turn", "hello", true, false},
		{"finalized ending the turn", "hello there", true, true},
	}
	for _, w := range want {
		res, err := stream.Recv()
		if err != nil {
			t.Fatalf("Recv %s: %v", w.desc, err)
		}
		if len(res) != 1 {
			t.Fatalf("Recv %s returned %d results, want 1", w.desc, len(res))
		}
		got := res[0]
		if got.Text != w.text || got.Final != w.final || got.EndOfTurn != w.endOfTurn {
			t.Errorf("%s = %+v, want text %q final=%v endOfTurn=%v", w.desc, got, w.text, w.final, w.endOfTurn)
		}
		if got.Language != "en" {
			t.Errorf("%s language = %q, want en", w.desc, got.Language)
		}
	}
}

// TestSTTRecvDone checks the closing transcript finalizes the turn, and that
// empty transcripts are skipped rather than emitted as blank frames.
func TestSTTRecvDone(t *testing.T) {
	session := newSTTSession(t, []map[string]any{
		{"type": "transcript.created"},
		{"type": "transcript.partial", "text": ""},
		{"type": "transcript.done", "text": ""},
		{"type": "transcript.done", "text": "goodbye"},
	})
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", URL: session.url, Encoding: defaultSTTEncoding}}
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 || res[0].Text != "goodbye" || !res[0].Final || !res[0].EndOfTurn {
		t.Fatalf("result = %+v, want a finalized \"goodbye\" ending the turn", res)
	}
}

// TestSTTSendGatedOnSessionReady checks audio is dropped until the server
// acknowledges the session, then streamed as binary frames, and that closing
// signals the end of the audio.
func TestSTTSendGatedOnSessionReady(t *testing.T) {
	session := newSTTSession(t, []map[string]any{{"type": "transcript.created"}})
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", URL: session.url, Encoding: defaultSTTEncoding}}
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Nothing has been read yet, so the session is not acknowledged and this
	// audio is dropped rather than sent.
	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send before ready: %v", err)
	}

	// Reading the acknowledgement opens the gate. It carries no transcript, so
	// Recv blocks until the connection closes; drive it on its own goroutine.
	readDone := make(chan struct{})
	go func() {
		defer close(readDone)
		_, _ = stream.Recv()
	}()
	s, ok := stream.(*sttStream)
	if !ok {
		t.Fatalf("Connect returned %T, want *sttStream", stream)
	}
	waitFor(t, s.ready.Load)

	if err := stream.Send([]byte{5, 6, 7, 8}); err != nil {
		t.Fatalf("Send after ready: %v", err)
	}
	got := <-session.received
	if got.typ != websocket.MessageBinary || string(got.data) != "\x05\x06\x07\x08" {
		t.Errorf("sent %v %q, want only the post-acknowledgement PCM as a binary frame", got.typ, got.data)
	}

	if err := stream.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	done := <-session.received
	if done.typ != websocket.MessageText || !strings.Contains(string(done.data), "audio.done") {
		t.Errorf("close sent %v %q, want the audio.done signal", done.typ, done.data)
	}
	<-readDone
}

// waitFor blocks until cond holds, failing the test if it never does.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for the session to be acknowledged")
		}
		time.Sleep(time.Millisecond)
	}
}

// TestSTTRecvError surfaces a server-reported failure to the caller.
func TestSTTRecvError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		b, _ := json.Marshal(map[string]any{"type": "error", "message": "bad rate"})
		_ = c.Write(r.Context(), websocket.MessageText, b)
	}))
	defer srv.Close()

	conn := &sttConnector{cfg: STTConfig{APIKey: "k", URL: wsURL(srv.URL), Encoding: defaultSTTEncoding}}
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if _, err := stream.Recv(); err == nil {
		t.Fatal("Recv() = nil error, want the server error surfaced")
	} else if !strings.Contains(err.Error(), "bad rate") {
		t.Errorf("Recv() error = %v, want it to carry the server message", err)
	}
}

// parseQuery pulls the query parameters off a built endpoint URL.
func parseQuery(t *testing.T, endpoint string) url.Values {
	t.Helper()
	u, err := url.Parse(endpoint)
	if err != nil {
		t.Fatalf("endpoint %q is not a URL: %v", endpoint, err)
	}
	return u.Query()
}

// wsURL turns an httptest server URL into a WebSocket one.
func wsURL(httpURL string) string { return "ws" + strings.TrimPrefix(httpURL, "http") }
