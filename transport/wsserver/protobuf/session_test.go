package protobuf_test

// An end-to-end test of the wire a client connects over: a real HTTP server, a
// real WebSocket, a real pipeline, and the protobuf serializer between them.
// Only the client is a test, and it speaks the same encoding a browser does.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/protobuf"
)

const sessionRate = 16000

// tap records every frame that passes it, since frames the output consumes never
// reach the end of the pipeline.
type tap struct {
	*processor.Base
	mu   sync.Mutex
	seen []frames.Frame
}

func newTap() *tap {
	t := &tap{}
	t.Base = processor.New("Tap", t)
	return t
}

func (t *tap) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := t.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	t.mu.Lock()
	t.seen = append(t.seen, f)
	t.mu.Unlock()
	return t.PushFrame(ctx, f, dir)
}

func (t *tap) frames() []frames.Frame {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]frames.Frame(nil), t.seen...)
}

// session is one client connected to a bot over the protobuf wire.
type session struct {
	client *websocket.Conn
	task   *pipeline.Worker
	tap    *tap
	done   chan error
}

// connect serves the transport, dials it, and runs a pipeline behind it.
func connect(t *testing.T) *session {
	t.Helper()

	s := &session{done: make(chan error, 1)}
	ready := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		params := wsserver.DefaultParams()
		params.AudioInSampleRate = sessionRate
		params.AudioOutSampleRate = sessionRate
		params.AudioOut10msChunks = 1

		tr, err := wsserver.Accept(w, r, protobuf.New(), params)
		if err != nil {
			t.Errorf("Accept: %v", err)
			close(ready)
			return
		}
		s.tap = newTap()
		noRTVI := false
		s.task = pipeline.NewWorker(
			pipeline.New(tr.Input(), s.tap, tr.Output()),
			pipeline.WorkerConfig{
				EnableRTVI: &noRTVI,
				Params: pipeline.Params{
					AudioInSampleRate:  sessionRate,
					AudioOutSampleRate: sessionRate,
				},
			})
		close(ready)
		s.done <- s.task.Run(ctx)
	}))
	t.Cleanup(srv.Close)

	conn, resp, err := websocket.Dial(t.Context(), "ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	s.client = conn
	t.Cleanup(func() { _ = conn.CloseNow() })

	select {
	case <-ready:
	case <-time.After(5 * time.Second):
		t.Fatal("the server never accepted the connection")
	}
	t.Cleanup(func() {
		s.task.StopWhenDone()
		select {
		case <-s.done:
		case <-time.After(5 * time.Second):
			cancel()
		}
	})
	return s
}

// send encodes a frame the way a client does and writes it as a binary message.
func (s *session) send(t *testing.T, f frames.Frame) {
	t.Helper()
	msg, err := protobuf.New().Serialize(f)
	if err != nil {
		t.Fatalf("encode %T: %v", f, err)
	}
	if err := s.client.Write(t.Context(), websocket.MessageBinary, msg.Data); err != nil {
		t.Fatalf("client write: %v", err)
	}
}

// receive reads one message and decodes it the way a client does.
func (s *session) receive(t *testing.T) (websocket.MessageType, frames.Frame) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	typ, data, err := s.client.Read(ctx)
	if err != nil {
		t.Fatalf("client read: %v", err)
	}
	f, err := protobuf.New().Deserialize(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return typ, f
}

// TestTheClientIsHeard is the inbound half: microphone audio a client encodes
// has to reach the pipeline as mic audio, which is what VAD, turn detection and
// the transcriber all run on.
func TestTheClientIsHeard(t *testing.T) {
	s := connect(t)

	// 10 ms of audio at 16 kHz mono 16-bit is 320 bytes.
	mic := make([]byte, 320)
	for i := range mic {
		mic[i] = byte(i)
	}
	s.send(t, frames.NewOutputAudioRawFrame(mic, sessionRate, 1))

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		for _, f := range s.tap.frames() {
			if af, ok := f.(*frames.InputAudioRawFrame); ok && len(af.AudioData().Audio) > 0 {
				return
			}
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("the client's microphone audio never reached the pipeline")
}

// TestTheBotIsHeard is the outbound half, and the one rtviws could never do:
// the bot's synthesized speech has to reach the client, framed as binary.
func TestTheBotIsHeard(t *testing.T) {
	s := connect(t)

	s.task.QueueFrame(frames.NewTTSAudioRawFrame(make([]byte, 320), sessionRate, 1))

	typ, f := s.receive(t)
	if typ != websocket.MessageBinary {
		t.Errorf("message type = %v, want %v", typ, websocket.MessageBinary)
	}
	af, ok := f.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("the client received %T, want audio", f)
	}
	if got := af.AudioData().SampleRate; got != sessionRate {
		t.Errorf("sample rate = %d, want %d", got, sessionRate)
	}
}

// TestAClientMessageReachesThePipeline covers the control channel: an RTVI
// message a client sends has to arrive as a transport message, which is what
// the pipeline's RTVI processor parses and routes.
func TestAClientMessageReachesThePipeline(t *testing.T) {
	s := connect(t)

	s.send(t, frames.NewOutputTransportMessageUrgentFrame(
		map[string]any{"label": "rtvi-ai", "type": "client-ready"}))

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		for _, f := range s.tap.frames() {
			mf, ok := f.(*frames.InputTransportMessageFrame)
			if ok && strings.Contains(string(mf.Message), "client-ready") {
				return
			}
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("the client's message never reached the pipeline")
}
