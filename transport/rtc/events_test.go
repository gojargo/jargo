package rtc_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/rtc"
	"github.com/gojargo/jargo/utils/events"
)

// eventPair connects a test client through a request handler and runs a
// pipeline on the transport, with handlers on the client events attached before
// it starts. It returns the client and the channels the events arrive on.
func eventPair(t *testing.T) (*testClient, chan *rtc.Connection, chan *rtc.Connection, chan string, *pipeline.Worker) {
	t.Helper()
	connected := make(chan *rtc.Connection, 4)
	disconnected := make(chan *rtc.Connection, 4)
	messages := make(chan string, 4)

	h := rtc.NewRequestHandler(rtc.RequestHandlerConfig{})
	t.Cleanup(func() { _ = h.Close() })
	client := newTestClient(t, true)
	var worker *pipeline.Worker
	a, err := h.HandleWebRequest(request(client.offer(t)), func(c *rtc.Connection) error {
		params := transport.DefaultParams()
		params.AudioInSampleRate = opus.SampleRate
		params.AudioOutSampleRate = opus.SampleRate
		tr := rtc.NewTransport(c, params)
		events.On(tr.Events(), transport.EventClientConnected, func(_ context.Context, c *rtc.Connection) {
			connected <- c
		})
		events.On(tr.Events(), transport.EventClientDisconnected, func(_ context.Context, c *rtc.Connection) {
			disconnected <- c
		})
		tr.Events().Add(transport.EventAppMessage, func(_ context.Context, _ any, args ...any) {
			raw, _ := args[0].(json.RawMessage)
			messages <- string(raw)
		})
		worker = pipeline.NewWorker(pipeline.New(tr.Input(), tr.Output()), pipeline.WorkerConfig{
			Params: pipeline.Params{AudioInSampleRate: opus.SampleRate, AudioOutSampleRate: opus.SampleRate},
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	client.accept(t, a)
	return client, connected, disconnected, messages, worker
}

func runWorker(t *testing.T, w *pipeline.Worker) (context.CancelFunc, <-chan struct{}) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = w.Run(ctx); close(done) }()
	t.Cleanup(func() {
		cancel()
		<-done
	})
	return cancel, done
}

// TestClientEventsOnTheWebRTCTransport covers a session from the bot's side:
// the client connecting is raised with its connection, its messages are raised
// as app messages, and its leaving is raised as a disconnection.
func TestClientEventsOnTheWebRTCTransport(t *testing.T) {
	client, connected, disconnected, messages, worker := eventPair(t)
	open := make(chan struct{})
	client.dc.OnOpen(func() { close(open) })
	runWorker(t, worker)

	select {
	case c := <-connected:
		if c == nil || c.PCID() == "" {
			t.Fatal("connected event without its connection")
		}
	case <-time.After(15 * time.Second):
		t.Fatal("no client connected event")
	}

	waitFor(t, open, "the data channel to open")
	if err := client.dc.SendText(`{"type":"hello"}`); err != nil {
		t.Fatal(err)
	}
	select {
	case m := <-messages:
		if m != `{"type":"hello"}` {
			t.Fatalf("app message = %s", m)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no app message event")
	}

	_ = client.pc.Close()
	select {
	case <-disconnected:
	case <-time.After(15 * time.Second):
		t.Fatal("no client disconnected event after the client closed")
	}
}

// TestTheBotLeavingIsNotAClientDisconnection covers the pipeline ending: the
// transport closes the connection itself, which is not the client leaving.
func TestTheBotLeavingIsNotAClientDisconnection(t *testing.T) {
	_, connected, disconnected, _, worker := eventPair(t)
	cancel, done := runWorker(t, worker)
	select {
	case <-connected:
	case <-time.After(15 * time.Second):
		t.Fatal("no client connected event")
	}
	cancel()
	<-done
	select {
	case <-disconnected:
		t.Fatal("the bot's own disconnect was reported as the client leaving")
	case <-time.After(500 * time.Millisecond):
	}
}
