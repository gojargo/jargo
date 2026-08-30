package proxy_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/gojargo/jargo/workers/proxy"
)

// The two halves talking to each other, which is what the pair is for and what
// testing each side against a fake cannot show. Each half sits on a bus of its
// own, as they would in two processes.

// pair is a client and a server joined by a real socket, each on its own bus.
type pair struct {
	client *proxy.Client
	server *proxy.Server
	// clientSide is the bus the client proxy sits on, where the worker named
	// "voice" lives; serverSide is the server's, where "worker" lives.
	clientSide *env
	serverSide *env
}

// newPair stands up both halves and connects them.
func newPair(t *testing.T, forward ...bus.Message) *pair {
	t.Helper()
	clientSide, serverSide := newEnv(t), newEnv(t)

	// The endpoint the client dials: it accepts the connection and hands it to a
	// server proxy, which is how an application wires one up.
	var (
		ready = make(chan *proxy.Server, 1)
		once  sync.Once
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		conn := wsutil.Adopt(raw, 1<<20)
		s := proxy.NewServer(proxy.ServerConfig{
			Name:            "gateway",
			Conn:            conn,
			LocalWorker:     "worker",
			RemoteWorker:    "voice",
			ForwardMessages: forward,
		})
		s.Attach(serverSide.ctx, serverSide.registry, serverSide.bus.Bus)
		s.Start(serverSide.ctx)
		once.Do(func() { ready <- s })
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	url := "ws" + strings.TrimPrefix(srv.URL, "http")
	c := proxy.NewClient(proxy.ClientConfig{
		Name:            "proxy",
		URL:             url,
		RemoteWorker:    "worker",
		LocalWorker:     "voice",
		ForwardMessages: forward,
	})
	c.Attach(clientSide.ctx, clientSide.registry, clientSide.bus.Bus)
	c.Start(clientSide.ctx)
	c.OnActivated(clientSide.ctx, nil)

	var s *proxy.Server
	select {
	case s = <-ready:
	case <-time.After(3 * time.Second):
		t.Fatal("the server proxy never accepted the connection")
	}

	t.Cleanup(func() {
		c.Stop(context.WithoutCancel(clientSide.ctx))
		s.Stop(context.WithoutCancel(serverSide.ctx))
	})
	return &pair{client: c, server: s, clientSide: clientSide, serverSide: serverSide}
}

// TestPairCarriesAMessageEachWay checks a message addressed across arrives on
// the other bus, in both directions. It is the whole point of the pair: two
// workers in two processes addressing each other by name.
func TestPairCarriesAMessageEachWay(t *testing.T) {
	p := newPair(t)

	// The client side's worker speaks to the server side's.
	p.client.OnBusMessage(p.clientSide.ctx, dataMessage("voice", "worker"))
	got := p.serverSide.recorder.awaitFrom(t, "voice")
	if got.Target() != "worker" {
		t.Errorf("arrived addressed to %q, want worker", got.Target())
	}

	// And back the other way.
	p.server.OnBusMessage(p.serverSide.ctx, dataMessage("worker", "voice"))
	got = p.clientSide.recorder.awaitFrom(t, "worker")
	if got.Target() != "voice" {
		t.Errorf("arrived addressed to %q, want voice", got.Target())
	}
}

// TestPairCarriesAFrameWhole checks a frame crosses with what it holds intact,
// which is what a bridged worker in another process depends on.
func TestPairCarriesAFrameWhole(t *testing.T) {
	p := newPair(t, &bus.FrameMessage{})

	m := &bus.FrameMessage{Frame: frames.NewTranscriptionFrame("bonjour", "user-1", "ts")}
	m.From = "voice"
	p.client.OnBusMessage(p.clientSide.ctx, m)

	got := p.serverSide.recorder.awaitFrom(t, "voice")
	fm, ok := got.(*bus.FrameMessage)
	if !ok {
		t.Fatalf("arrived as %T, want a FrameMessage", got)
	}
	tf, ok := fm.Frame.(*frames.TranscriptionFrame)
	if !ok {
		t.Fatalf("the frame arrived as %T, want a TranscriptionFrame", fm.Frame)
	}
	if tf.Text != "bonjour" || tf.UserID != "user-1" {
		t.Errorf("the frame arrived as %+v, want the transcription that was sent", tf)
	}
}

// TestPairTellsTheClientTheWorkerIsReady checks the discovery half: the client
// side learns that the worker it is addressing exists, which is what the server
// announces when its local worker registers.
func TestPairTellsTheClientTheWorkerIsReady(t *testing.T) {
	p := newPair(t)

	p.server.OnWorkerReady(p.serverSide.ctx,
		registry.WorkerReadyData{WorkerName: "worker", Runner: runnerName})

	got := p.clientSide.recorder.awaitFrom(t, "gateway")
	snapshot, ok := got.(*bus.WorkerRegistryMessage)
	if !ok {
		t.Fatalf("arrived as %T, want a WorkerRegistryMessage", got)
	}
	if len(snapshot.Workers) != 1 || snapshot.Workers[0].Name != "worker" {
		t.Errorf("arrived naming %+v, want the worker on the far side", snapshot.Workers)
	}
}
