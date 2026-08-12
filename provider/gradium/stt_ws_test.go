package gradium

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
)

// Tests for the transcription session. Gradium acknowledges a setup handshake
// before audio flows, takes base64 audio in fixed chunks, and returns text
// fragments that accumulate until a flush settles the utterance.

// sttCapture is what the fake endpoint saw.
type sttCapture struct {
	header http.Header
	got    chan map[string]any
}

// sttServer starts a fake Gradium ASR endpoint. It answers the setup with ready
// unless setupReply overrides it, then sends the scripted messages.
func sttServer(t *testing.T, setupReply map[string]any, reply []map[string]any) (string, *sttCapture) {
	t.Helper()
	c := &sttCapture{got: make(chan map[string]any, 16)}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c.header = r.Header.Clone()
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		_, data, err := conn.Read(ctx)
		if err != nil {
			return
		}
		m := map[string]any{}
		if err := json.Unmarshal(data, &m); err != nil {
			t.Errorf("decoding the setup: %v", err)
			return
		}
		c.got <- m

		ack := setupReply
		if ack == nil {
			ack = map[string]any{msgType: "ready"}
		}
		b, err := json.Marshal(ack)
		if err != nil {
			t.Errorf("encoding the acknowledgement: %v", err)
			return
		}
		if conn.Write(ctx, websocket.MessageText, b) != nil {
			return
		}

		for _, e := range reply {
			b, err := json.Marshal(e)
			if err != nil {
				t.Errorf("encoding a reply: %v", err)
				return
			}
			if conn.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}

		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			m := map[string]any{}
			if json.Unmarshal(data, &m) == nil {
				select {
				case c.got <- m:
				default:
				}
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), c
}

func (c *sttCapture) await(t *testing.T) map[string]any {
	t.Helper()
	select {
	case m := <-c.got:
		return m
	case <-time.After(2 * time.Second):
		t.Fatal("no message arrived at the endpoint")
		return nil
	}
}

// sttConn builds a connector with the defaults NewSTT fills in.
func sttConn(endpoint string, opts ...func(*STTConfig)) *sttConnector {
	cfg := STTConfig{APIKey: "test-key", URL: endpoint, Model: defaultSTTModel}
	for _, o := range opts {
		o(&cfg)
	}
	return &sttConnector{cfg: cfg}
}

func TestSTTConnectSendsTheSetup(t *testing.T) {
	endpoint, seen := sttServer(t, nil, nil)

	s, err := sttConn(endpoint, func(c *STTConfig) { c.Language = language.French }).
		Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	if seen.header.Get("x-api-key") != "test-key" {
		t.Errorf("x-api-key = %q, want the configured key", seen.header.Get("x-api-key"))
	}

	m := seen.await(t)
	if m[msgType] != "setup" {
		t.Fatalf("first message = %v, want the setup", m)
	}
	if m["model_name"] != defaultSTTModel {
		t.Errorf("model_name = %v, want %q", m["model_name"], defaultSTTModel)
	}
	cfg, _ := m["json_config"].(map[string]any)
	if cfg["language"] != "fr" {
		t.Errorf("language = %v, want the base code", cfg["language"])
	}
	if cfg["delay_in_frames"] != float64(defaultDelayInFrames) {
		t.Errorf("delay_in_frames = %v, want the default %d", cfg["delay_in_frames"], defaultDelayInFrames)
	}
}

// TestSTTConnectOmitsTheLanguageHintWhenUnset checks an unset language leaves
// the hint out, so Gradium detects it rather than being told the wrong one.
func TestSTTConnectOmitsTheLanguageHintWhenUnset(t *testing.T) {
	endpoint, seen := sttServer(t, nil, nil)

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	cfg, _ := seen.await(t)["json_config"].(map[string]any)
	if v, ok := cfg["language"]; ok {
		t.Errorf("language = %v, want it omitted when unset", v)
	}
}

// TestSTTConnectRefusesAnUnacknowledgedSetup checks the session is not handed
// back unless the provider said it is ready, so audio is never sent into a
// session that never started.
func TestSTTConnectRefusesAnUnacknowledgedSetup(t *testing.T) {
	tests := []struct {
		name  string
		reply map[string]any
	}{
		{name: "an error", reply: map[string]any{msgType: "error", "message": "bad model"}},
		{name: "something else entirely", reply: map[string]any{msgType: "text", "text": "hi"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			endpoint, _ := sttServer(t, tt.reply, nil)

			s, err := sttConn(endpoint).Connect(context.Background(), 16000)
			if err == nil {
				_ = s.Close()
				t.Fatal("Connect accepted a setup that was never acknowledged")
			}
			if !errors.Is(err, errSTTProtocol) {
				t.Errorf("error = %v, want it reported as a protocol error", err)
			}
		})
	}
}

// TestSTTSendChunksTheAudio checks the audio goes in the fixed-size chunks
// Gradium expects, with the remainder held back rather than sent short.
func TestSTTSendChunksTheAudio(t *testing.T) {
	endpoint, seen := sttServer(t, nil, nil)

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()
	seen.await(t) // the setup

	chunk := sttChunkMS * 16000 * 2 / 1000
	if err := s.Send(make([]byte, chunk*2+100)); err != nil {
		t.Fatalf("Send: %v", err)
	}

	for range 2 {
		m := seen.await(t)
		if m[msgType] != msgAudio {
			t.Fatalf("message = %v, want an audio chunk", m)
		}
		encoded, ok := m[msgAudio].(string)
		if !ok {
			t.Fatalf("the audio field is %T, want a base64 string", m[msgAudio])
		}
		raw, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			t.Fatalf("the audio is not base64: %v", err)
		}
		if len(raw) != chunk {
			t.Errorf("chunk was %d bytes, want %d", len(raw), chunk)
		}
	}
	select {
	case m := <-seen.got:
		t.Errorf("a third message %v was sent, want the remainder held back", m)
	case <-time.After(200 * time.Millisecond):
	}
}

// TestSTTRecvAccumulatesUntilTheFlush checks text fragments build the running
// transcript and a flush settles it, ending the turn.
func TestSTTRecvAccumulatesUntilTheFlush(t *testing.T) {
	endpoint, _ := sttServer(t, nil, []map[string]any{
		{msgType: msgText, "text": "hello"},
		{msgType: msgText, "text": "there"},
		{msgType: msgFlushed},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	first, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if first[0].Text != "hello" || first[0].Final {
		t.Errorf("result = %+v, want the interim hello", first[0])
	}

	second, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if second[0].Text != "hello there" {
		t.Errorf("text = %q, want the fragments joined", second[0].Text)
	}

	final, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if final[0].Text != "hello there" || !final[0].Final || !final[0].EndOfTurn {
		t.Errorf("result = %+v, want the settled utterance ending the turn", final[0])
	}
}

// TestSTTRecvGathersTrailingWords is the point of the aggregation window: a
// flush only says the buffered audio was processed, so words still arriving
// belong to this utterance rather than the front of the next one.
func TestSTTRecvGathersTrailingWords(t *testing.T) {
	endpoint, _ := sttServer(t, nil, []map[string]any{
		{msgType: msgText, "text": "hello"},
		{msgType: msgFlushed},
		{msgType: msgText, "text": "there"},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	if _, err := s.Recv(); err != nil { // the interim
		t.Fatalf("Recv: %v", err)
	}
	final, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if final[0].Text != "hello there" {
		t.Errorf("text = %q, want the trailing word folded in", final[0].Text)
	}
}

// TestSTTRecvIgnoresAFlushWithNothingHeld checks a flush with no text settles
// nothing, so silence does not close a turn.
func TestSTTRecvIgnoresAFlushWithNothingHeld(t *testing.T) {
	endpoint, _ := sttServer(t, nil, []map[string]any{
		{msgType: msgFlushed},
		{msgType: msgText, "text": "after"},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	got, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if got[0].Text != "after" {
		t.Errorf("text = %q, want the empty flush stepped over", got[0].Text)
	}
}

// TestSTTRecvReportsAProtocolError checks an error is surfaced carrying what the
// provider said.
func TestSTTRecvReportsAProtocolError(t *testing.T) {
	endpoint, _ := sttServer(t, nil, []map[string]any{
		{msgType: msgError, "message": "stream aborted"},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	if _, err := s.Recv(); err == nil {
		t.Fatal("Recv accepted an error message")
	} else if !errors.Is(err, errSTTProtocol) || !strings.Contains(err.Error(), "stream aborted") {
		t.Errorf("error = %v, want the protocol error carrying its reason", err)
	}
}

// TestSTTConnectReportsAnUnreachableEndpoint checks a dial that fails is
// reported.
func TestSTTConnectReportsAnUnreachableEndpoint(t *testing.T) {
	if _, err := sttConn("ws://127.0.0.1:1").Connect(context.Background(), 16000); err == nil {
		t.Fatal("Connect accepted an unreachable endpoint")
	}
}
