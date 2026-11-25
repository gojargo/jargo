package wsclient_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/transport/wsclient"
	"github.com/gojargo/jargo/transport/wsserver"
)

// The opening messages go out first on the connection, ahead of the
// client-ready handshake.
func TestOpeningMessagesGoOutBeforeClientReady(t *testing.T) {
	server := newBotServer(t, nil)
	tr := wsclient.NewRTVITransport(server.url, wsclient.Params{})
	tr.SetOpeningMessages(wsserver.Message{Data: []byte(`{"label":"rtvi-ai","type":"client-message","id":"hello"}`)})

	noRTVI := false
	worker := pipeline.NewWorker(pipeline.New(tr.Input(), tr.Output()), pipeline.WorkerConfig{EnableRTVI: &noRTVI})
	done := make(chan error, 1)
	go func() { done <- worker.Run(t.Context()) }()
	deadline := time.Now().Add(5 * time.Second)
	for !server.sawClientReady() && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	worker.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatalf("Run: %v", err)
	}

	server.mu.Lock()
	defer server.mu.Unlock()
	if len(server.received) < 2 || server.received[0]["id"] != "hello" || server.received[1]["type"] != "client-ready" {
		t.Fatalf("received %v, want the opening message, then client-ready", server.received)
	}
}

// A server closing the connection leaves how it closed it, its code and reason,
// for the disconnected event's handlers to read.
func TestCloseErrorCarriesHowTheServerClosed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		_ = conn.Close(websocket.StatusInternalError, "Concurrency limit reached")
	}))
	defer srv.Close()

	tr := wsclient.NewRTVITransport("ws"+strings.TrimPrefix(srv.URL, "http"), wsclient.Params{})
	closed := make(chan error, 1)
	event.On(tr.Events(), wsclient.EventDisconnected, func(context.Context, *websocket.Conn) {
		closed <- tr.CloseError()
	})
	noRTVI := false
	worker := pipeline.NewWorker(pipeline.New(tr.Input(), tr.Output()), pipeline.WorkerConfig{EnableRTVI: &noRTVI})
	done := make(chan error, 1)
	go func() { done <- worker.Run(t.Context()) }()

	select {
	case err := <-closed:
		var ce websocket.CloseError
		if !errors.As(err, &ce) || ce.Code != websocket.StatusInternalError || ce.Reason != "Concurrency limit reached" {
			t.Fatalf("close error = %v, want the server's code and reason", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the server closing was never reported")
	}
	worker.StopWhenDone()
	<-done
}
