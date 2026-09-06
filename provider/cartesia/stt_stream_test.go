package cartesia

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
)

// sttSession is what the fake STT endpoint saw: the query it was dialed with,
// the headers, and everything the client wrote before the socket closed.
type sttSession struct {
	query  url.Values
	header http.Header
	audio  chan []byte
	text   chan string
}

// sttServer starts a fake Cartesia STT endpoint that replays scripted messages
// and records what the client sends it.
func sttServer(t *testing.T, reply []map[string]any) (endpoint string, got *sttSession) {
	t.Helper()
	got = &sttSession{
		audio: make(chan []byte, 8),
		text:  make(chan string, 8),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.query = r.URL.Query()
		got.header = r.Header.Clone()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		for _, m := range reply {
			b, err := json.Marshal(m)
			if err != nil {
				t.Errorf("encoding a reply message: %v", err)
				return
			}
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			typ, data, err := c.Read(ctx)
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

// sttConfigFor is a config pointed at endpoint with the defaults NewSTT applies.
func sttConfigFor(endpoint string) STTConfig {
	return STTConfig{
		APIKey:   "test-key",
		URL:      endpoint,
		Version:  defaultSTTVersion,
		Model:    defaultSTTModel,
		Encoding: defaultSTTEncoding,
	}
}

// TestSTTConnectSendsTheSessionParameters checks the session is dialed with the
// Cartesia headers and describes the audio it is about to receive. Cartesia
// takes all of it as query parameters, so getting it wrong is a session that
// transcribes the wrong thing rather than one that fails.
func TestSTTConnectSendsTheSessionParameters(t *testing.T) {
	endpoint, got := sttServer(t, nil)
	c := newSTTConnector(sttConfigFor(endpoint))

	stream, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if h := got.header.Get("X-API-Key"); h != "test-key" {
		t.Errorf("X-API-Key = %q, want the configured key", h)
	}
	if h := got.header.Get("Cartesia-Version"); h != defaultSTTVersion {
		t.Errorf("Cartesia-Version = %q, want the pinned %q", h, defaultSTTVersion)
	}
	if q := got.query.Get("model"); q != defaultSTTModel {
		t.Errorf("model = %q, want %q", q, defaultSTTModel)
	}
	if q := got.query.Get("encoding"); q != defaultSTTEncoding {
		t.Errorf("encoding = %q, want %q", q, defaultSTTEncoding)
	}
	if q := got.query.Get("sample_rate"); q != "16000" {
		t.Errorf("sample_rate = %q, want the rate the transport runs at", q)
	}
	if q := got.query.Get("language"); q != defaultSTTLanguage {
		t.Errorf("language = %q, want the default %q", q, defaultSTTLanguage)
	}
}

// TestSTTConnectRefused checks a session that cannot be opened is reported.
func TestSTTConnectRefused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	c := newSTTConnector(sttConfigFor("ws" + strings.TrimPrefix(srv.URL, "http")))
	if _, err := c.Connect(t.Context(), 16000); err == nil {
		t.Fatal("Connect on a refused session = nil, want an error")
	}
}

// TestSTTStreamSendsAudioAsBinary checks the PCM goes out as a binary frame,
// which is how Cartesia distinguishes audio from the control messages that share
// the socket.
func TestSTTStreamSendsAudioAsBinary(t *testing.T) {
	endpoint, got := sttServer(t, nil)
	c := newSTTConnector(sttConfigFor(endpoint))

	stream, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

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
}

// TestSTTStreamRecv checks the transcript messages the service surfaces: an
// interim result and a final one both reach the pipeline, carrying the language
// the server recognized. Finals do not close the turn here; the upstream turn
// detector does that.
func TestSTTStreamRecv(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{"type": "transcript", "text": "hello", "is_final": false, "language": "en"},
		{"type": "transcript", "text": "", "is_final": false},
		{"type": "flush_done"},
		{"type": "transcript", "text": "hello there", "is_final": true, "language": "en"},
	})
	c := newSTTConnector(sttConfigFor(endpoint))

	stream, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	first, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(first) != 1 || first[0].Text != "hello" || first[0].Final {
		t.Errorf("first result = %+v, want the interim transcript", first)
	}
	if first[0].Language != "en" {
		t.Errorf("language = %q, want the recognized language", first[0].Language)
	}

	// The empty transcript and the unmodeled message are both skipped, so the
	// next thing Recv returns is the final.
	second, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(second) != 1 || second[0].Text != "hello there" || !second[0].Final {
		t.Errorf("second result = %+v, want the final transcript", second)
	}
}

// TestSTTStreamRecvServerError checks an error message is reported rather than
// leaving the pipeline waiting for a transcript that will not come.
func TestSTTStreamRecvServerError(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{"type": "error", "message": "unsupported sample rate"},
	})
	c := newSTTConnector(sttConfigFor(endpoint))

	stream, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if _, err := stream.Recv(); !errors.Is(err, errSTTProtocol) {
		t.Fatalf("Recv error = %v, want errSTTProtocol", err)
	} else if !strings.Contains(err.Error(), "unsupported sample rate") {
		t.Errorf("error = %v, want it to carry the server's message", err)
	}
}

// TestSTTStreamCloseSaysDone checks the socket is closed only after telling
// Cartesia the audio is complete, so it finalizes what it has rather than
// discarding the tail of the utterance.
func TestSTTStreamCloseSaysDone(t *testing.T) {
	endpoint, got := sttServer(t, nil)
	c := newSTTConnector(sttConfigFor(endpoint))

	stream, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	select {
	case msg := <-got.text:
		if msg != "done" {
			t.Errorf("closing message = %q, want done", msg)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the endpoint was never told the audio was done")
	}
}

// TestSTTStreamFinalizes checks the service asks Cartesia to flush the
// transcript when the speech ends, rather than waiting on Cartesia's own
// endpointing. The stream implements stt.Finalizer, which is what has the base
// call it as the VAD reports the user stopped.
func TestSTTStreamFinalizes(t *testing.T) {
	endpoint, got := sttServer(t, nil)
	c := newSTTConnector(sttConfigFor(endpoint))

	stream, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	f, ok := stream.(stt.Finalizer)
	if !ok {
		t.Fatal("the stream does not implement stt.Finalizer, so the base never flushes it")
	}
	if err := f.Finalize(); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	select {
	case msg := <-got.text:
		if msg != "finalize" {
			t.Errorf("flush message = %q, want finalize", msg)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the endpoint was never asked to flush")
	}
}

// TestSTTKeepalive checks an idle session is held open. Cartesia closes a
// connection that has carried nothing for three minutes, and silence submitted
// inside that window resets its timer.
func TestSTTKeepalive(t *testing.T) {
	c := newSTTConnector(sttConfigFor("wss://example.invalid"))
	k, ok := any(c).(stt.Keepaliver)
	if !ok {
		t.Fatal("the connector does not implement stt.Keepaliver, so an idle session is dropped")
	}
	opts := k.Keepalive()
	if opts.Timeout != sttKeepaliveTimeout || opts.Interval != sttKeepaliveInterval {
		t.Errorf("keepalive = %+v, want timeout %s every %s",
			opts, sttKeepaliveTimeout, sttKeepaliveInterval)
	}
	if opts.Timeout >= 3*time.Minute {
		t.Errorf("keepalive timeout %s does not fit inside Cartesia's three-minute idle window",
			opts.Timeout)
	}
}

// TestSTTExtraHeadersReachTheHandshake checks a deployment can put its own
// headers on the connection, which is how a proxy in front of Cartesia
// authorizes or routes the session. They are applied last, so one of them can
// also replace what the service sets.
func TestSTTExtraHeadersReachTheHandshake(t *testing.T) {
	endpoint, got := sttServer(t, nil)
	cfg := sttConfigFor(endpoint)
	cfg.Headers = map[string]string{
		"X-Tenant":  "acme",
		"X-API-Key": "proxy-key",
	}
	c := newSTTConnector(cfg)

	stream, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if h := got.header.Get("X-Tenant"); h != "acme" {
		t.Errorf("X-Tenant = %q, want the configured header", h)
	}
	if h := got.header.Get("X-API-Key"); h != "proxy-key" {
		t.Errorf("X-API-Key = %q, want the configured header to win", h)
	}
}
