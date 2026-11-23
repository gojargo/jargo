package wsclient_test

// Runs the RTVI client transport in a pipeline against a tiny RTVI WebSocket
// server and checks the bot's server messages arrive as raw messages, in order
// and downstream only.

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
	"github.com/gojargo/jargo/transport/wsclient/rtviclient"
	"github.com/gojargo/jargo/utils/events"
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

// recorder records the bot's messages that reach it, in the direction it
// watches, and passes every frame on.
type recorder struct {
	*processor.Base
	dir processor.Direction

	mu       sync.Mutex
	messages []*frames.InputTransportMessageFrame
}

func newRecorder(name string, dir processor.Direction) *recorder {
	r := &recorder{dir: dir}
	r.Base = processor.New(name, r)
	return r
}

func (r *recorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if mf, ok := f.(*frames.InputTransportMessageFrame); ok && dir == r.dir {
		r.mu.Lock()
		r.messages = append(r.messages, mf)
		r.mu.Unlock()
	}
	return r.PushFrame(ctx, f, dir)
}

// reports are the bot's messages recorded so far, decoded, in the order they
// arrived.
func (r *recorder) reports(t *testing.T) []map[string]any {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]map[string]any, 0, len(r.messages))
	for _, mf := range r.messages {
		var m map[string]any
		if err := json.Unmarshal(mf.Message, &m); err != nil {
			t.Fatalf("%s: %v", mf.Message, err)
		}
		out = append(out, m)
	}
	return out
}

// runTransport runs tr's input between a recorder of what travels upstream of
// it and one of what travels downstream, until done reports true or five
// seconds pass, and returns the two recorders.
func runTransport(t *testing.T, tr *wsclient.RTVITransport, done func(down *recorder) bool) (down, up *recorder) {
	t.Helper()
	up = newRecorder("Upstream", processor.Upstream)
	down = newRecorder("Downstream", processor.Downstream)

	noRTVI := false
	worker := pipeline.NewWorker(pipeline.New(up, tr.Input(), down), pipeline.WorkerConfig{EnableRTVI: &noRTVI})
	finished := make(chan error, 1)
	go func() { finished <- worker.Run(t.Context()) }()

	// Let connect, the handshake and the replies flow.
	deadline := time.Now().Add(5 * time.Second)
	for !done(down) && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	worker.StopWhenDone()
	select {
	case err := <-finished:
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	return down, up
}

func TestBotMessagesArriveAsRawMessages(t *testing.T) {
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
	// bot-ready, then the replies.
	down, up := runTransport(t, tr, func(down *recorder) bool { return len(down.reports(t)) > len(replies) })

	// The bot's reports arrived as raw messages, in the order it sent them.
	reports := down.reports(t)
	types := make([]any, 0, len(reports))
	for _, m := range reports {
		types = append(types, m["type"])
	}
	want := []any{"bot-ready", "bot-llm-started", "bot-llm-text", "bot-llm-stopped", "user-transcription"}
	if !reflect.DeepEqual(types, want) {
		t.Fatalf("messages = %v, want %v", types, want)
	}
	if data, _ := reports[2]["data"].(map[string]any); data["text"] != "Paris" {
		t.Errorf("text = %v, want Paris", data["text"])
	}
	// Downstream only: the worker's RTVI processor, ahead of the pipeline,
	// would take them for a client's messages to it.
	if got := up.reports(t); len(got) != 0 {
		t.Errorf("messages pushed upstream: %v", got)
	}

	// The transport completed the handshake: the server saw client-ready.
	if !server.sawClientReady() {
		t.Error("the server never saw client-ready")
	}
	if !tr.BotReady() {
		t.Error("the transport is not bot-ready")
	}
}

// recordingSerializer notes the type of every message it is handed, before it
// deserializes it.
type recordingSerializer struct {
	*rtviclient.Serializer
	record func(string)
}

func (s *recordingSerializer) Deserialize(data []byte) (frames.Frame, error) {
	var message struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(data, &message)
	s.record(message.Type)
	return s.Serializer.Deserialize(data)
}

// The bot reports right after bot-ready; the handler, checking what the bot
// reported, runs before any of that reaches the pipeline.
func TestBotReadyIsHandledBeforeTheBotsNextMessage(t *testing.T) {
	var (
		mu    sync.Mutex
		order []string
	)
	record := func(what string) {
		mu.Lock()
		order = append(order, what)
		mu.Unlock()
	}
	recorded := func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), order...)
	}

	server := newBotServer(t, [][]byte{rtviMessage(t, "bot-llm-started", nil)})
	tr := wsclient.NewRTVITransport(server.url, wsclient.Params{
		Serializer: &recordingSerializer{Serializer: rtviclient.New(), record: record},
	})
	events.OnSignal(tr.Events(), wsclient.EventBotReady, func(context.Context) {
		time.Sleep(100 * time.Millisecond)
		record("on_bot_ready")
	})
	runTransport(t, tr, func(*recorder) bool { return len(recorded()) >= 3 })

	if want := []string{"on_bot_ready", "bot-ready", "bot-llm-started"}; !reflect.DeepEqual(recorded(), want) {
		t.Fatalf("order = %v, want %v", recorded(), want)
	}
}
