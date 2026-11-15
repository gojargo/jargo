package wsclient_test

// Runs the RTVI client transport in a pipeline against a tiny RTVI WebSocket
// server and checks the bot's server messages arrive as the right frames.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsclient"
)

func rtviMessage(t *testing.T, msgType string, data map[string]any) []byte {
	t.Helper()
	raw, err := json.Marshal(map[string]any{"label": rtvi.MessageLabel, "type": msgType, "id": "x", "data": data})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// botServer is a tiny RTVI server: on client-ready it replies bot-ready, then a
// scripted turn.
type botServer struct {
	url string

	mu       sync.Mutex
	received []map[string]any
}

func newBotServer(t *testing.T, replies [][]byte) *botServer {
	t.Helper()
	b := &botServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		ctx := r.Context()
		for {
			_, raw, err := conn.Read(ctx)
			if err != nil {
				return
			}
			var m map[string]any
			_ = json.Unmarshal(raw, &m)
			b.mu.Lock()
			b.received = append(b.received, m)
			b.mu.Unlock()
			if m["type"] != "client-ready" {
				continue
			}
			ready := rtviMessage(t, "bot-ready", map[string]any{"version": rtvi.ProtocolVersion})
			for _, reply := range append([][]byte{ready}, replies...) {
				if err := conn.Write(ctx, websocket.MessageText, reply); err != nil {
					return
				}
			}
		}
	}))
	t.Cleanup(srv.Close)
	b.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return b
}

func (b *botServer) sawClientReady() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, m := range b.received {
		if m["type"] == "client-ready" {
			return true
		}
	}
	return false
}

// sink records every frame that reaches the end of the pipeline.
type sink struct {
	*processor.Base

	mu   sync.Mutex
	seen []frames.Frame
}

func newSink() *sink {
	s := &sink{}
	s.Base = processor.New("Sink", s)
	return s
}

func (s *sink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	s.mu.Lock()
	s.seen = append(s.seen, f)
	s.mu.Unlock()
	return s.PushFrame(ctx, f, dir)
}

// conversation is the frames a scripted turn maps to, in the order they
// arrived.
func (s *sink) conversation() []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []frames.Frame
	for _, f := range s.seen {
		switch f.(type) {
		case *frames.LLMFullResponseStartFrame, *frames.LLMTextFrame,
			*frames.LLMFullResponseEndFrame, *frames.TranscriptionFrame:
			out = append(out, f)
		}
	}
	return out
}

func TestBotMessagesBecomeFrames(t *testing.T) {
	replies := [][]byte{
		rtviMessage(t, "bot-llm-started", nil),
		rtviMessage(t, "bot-llm-text", map[string]any{"text": "Paris"}),
		rtviMessage(t, "bot-llm-stopped", nil),
		rtviMessage(t, "user-transcription", map[string]any{
			"text": "hello", "user_id": "u", "timestamp": "t", "final": true,
		}),
	}
	server := newBotServer(t, replies)
	tr := wsclient.NewRTVITransport(server.url, wsclient.Params{})
	out := newSink()

	noRTVI := false
	worker := pipeline.NewWorker(pipeline.New(tr.Input(), out), pipeline.WorkerConfig{EnableRTVI: &noRTVI})
	done := make(chan error, 1)
	go func() { done <- worker.Run(t.Context()) }()

	// Let connect, the handshake and the replies flow.
	deadline := time.Now().Add(5 * time.Second)
	for len(out.conversation()) < len(replies) && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	worker.StopWhenDone()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}

	// The bot's server messages arrived as the expected pipeline frames, in
	// order.
	var kinds []reflect.Type
	for _, f := range out.conversation() {
		kinds = append(kinds, reflect.TypeOf(f))
	}
	want := []reflect.Type{
		reflect.TypeFor[*frames.LLMFullResponseStartFrame](),
		reflect.TypeFor[*frames.LLMTextFrame](),
		reflect.TypeFor[*frames.LLMFullResponseEndFrame](),
		reflect.TypeFor[*frames.TranscriptionFrame](),
	}
	if !reflect.DeepEqual(kinds, want) {
		t.Fatalf("frames = %v, want %v", kinds, want)
	}
	if text, _ := out.conversation()[1].(*frames.LLMTextFrame); text.Text != "Paris" {
		t.Errorf("text = %q, want Paris", text.Text)
	}

	// The transport completed the handshake: the server saw client-ready.
	if !server.sawClientReady() {
		t.Error("the server never saw client-ready")
	}
	if !tr.BotReady() {
		t.Error("the transport is not bot-ready")
	}
}
