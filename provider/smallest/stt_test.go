package smallest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/stt"
)

func TestSTTValidate(t *testing.T) {
	if err := (STTConfig{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (STTConfig{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
}

func TestSTTConnectAndRecv(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil { // consume the binary audio
			return
		}
		interim, _ := json.Marshal(map[string]any{"is_final": false, "transcript": "hel"})
		_ = c.Write(ctx, websocket.MessageText, interim)
		final, _ := json.Marshal(map[string]any{"is_final": true, "transcript": "hello", "language": "en"})
		_ = c.Write(ctx, websocket.MessageText, final)
	}))
	defer srv.Close()

	conn := &connector{cfg: STTConfig{APIKey: "k", BaseURL: wsURL(srv.URL), Encoding: defaultSTTEncoding}}
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv interim: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hel" || res[0].Final {
		t.Fatalf("interim result = %+v", res)
	}

	res, err = stream.Recv()
	if err != nil {
		t.Fatalf("Recv final: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hello" || !res[0].Final || !res[0].EndOfTurn || res[0].Language != "en" {
		t.Fatalf("final result = %+v", res)
	}
}

// The session is opened on the v4 endpoint, carrying every option the service
// accepts. Keywords are the exception: an empty value would register one empty
// keyword, so the parameter is left off entirely.
func TestSTTQueryCarriesEveryOption(t *testing.T) {
	yes, no := true, false
	c := &connector{cfg: STTConfig{
		APIKey: "k", Model: "pulse", Language: "en", Encoding: defaultSTTEncoding,
		Numerals: "auto", WordTimestamps: true, FullTranscript: true,
		SentenceTimestamps: true, RedactPII: true, RedactPCI: true, Diarize: true,
		Endpointing: &no, Format: &yes,
	}}

	q := c.query(16000)
	want := map[string]string{
		"model": "pulse", "language": "en", "encoding": "linear16",
		"sample_rate": "16000", "word_timestamps": "true", "full_transcript": "true",
		"sentence_timestamps": "true", "redact_pii": "true", "redact_pci": "true",
		"numerals": "auto", "diarize": "true", "endpointing": "false", "format": "true",
	}
	for k, v := range want {
		if got := q.Get(k); got != v {
			t.Errorf("%s = %q, want %q", k, got, v)
		}
	}
	if q.Has("keywords") {
		t.Errorf("keywords sent with none configured: %q", q.Get("keywords"))
	}

	c.cfg.Keywords = "NVIDIA:2"
	if got := c.query(16000).Get("keywords"); got != "NVIDIA:2" {
		t.Errorf("keywords = %q, want the configured pairs", got)
	}
}

// Endpointing and formatting are on unless turned off, which is what the service
// itself does with them.
func TestSTTQueryDefaultsEndpointingAndFormat(t *testing.T) {
	c := &connector{cfg: STTConfig{APIKey: "k", Encoding: defaultSTTEncoding, Numerals: "auto"}}

	q := c.query(16000)
	if q.Get("endpointing") != "true" || q.Get("format") != "true" {
		t.Errorf("endpointing = %q, format = %q, want both on", q.Get("endpointing"), q.Get("format"))
	}
}

// The end of the user's speech is flushed with a finalize message, and the
// session stays open for the next utterance rather than being torn down.
func TestSTTFinalizeFlushesTheUtterance(t *testing.T) {
	got := make(chan string, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
			if typ == websocket.MessageText {
				select {
				case got <- string(data):
				default:
				}
			}
		}
	}))
	defer srv.Close()

	c := &connector{cfg: STTConfig{APIKey: "k", BaseURL: wsURL(srv.URL), Encoding: defaultSTTEncoding}}
	stream, err := c.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	fin, ok := stream.(stt.Finalizer)
	if !ok {
		t.Fatal("the stream cannot be told the speech ended, so a turn waits on the service's own endpointing")
	}
	if err := fin.Finalize(); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	select {
	case msg := <-got:
		if msg != `{"type":"finalize"}` {
			t.Errorf("sent %q, want a finalize message", msg)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no finalize reached the service")
	}

	// The session carries on: audio for the next utterance still goes through.
	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Errorf("the session was closed by the finalize: %v", err)
	}
}
