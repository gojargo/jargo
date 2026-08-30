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
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/proxy"
)

// Ported from upstream's proxy suite. What it guards is the routing: which
// messages leave this process and which arriving ones this process acts on. Get
// either wrong and a bus either leaks its whole traffic onto the wire or
// silently drops the work it was carrying.
//
// Upstream drives the two proxies against a fake socket object. Here they are
// driven against a real one, since Go has a WebSocket at both ends of a test.

const runnerName = "test-runner"

// peer is the far side of a proxy's connection: it collects what the proxy sent
// and speaks what a test tells it to.
type peer struct {
	mu   sync.Mutex
	got  [][]byte
	send chan []byte
}

func newPeer() *peer { return &peer{send: make(chan []byte, 8)} }

func (p *peer) received() [][]byte {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([][]byte(nil), p.got...)
}

// serve accepts one connection and runs the peer on it.
func (p *peer) serve(t *testing.T) (url string) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		go func() {
			for data := range p.send {
				if err := c.Write(r.Context(), websocket.MessageBinary, data); err != nil {
					return
				}
			}
		}()
		for {
			_, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			p.mu.Lock()
			p.got = append(p.got, data)
			p.mu.Unlock()
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

// env is a running bus with a recorder on it, which is what shows whether a
// message the proxy read actually reached this side.
type env struct {
	ctx      context.Context
	bus      *bus.AsyncQueueBus
	registry *registry.WorkerRegistry
	recorder *recorder
}

func newEnv(t *testing.T) *env {
	t.Helper()
	e := &env{
		ctx:      t.Context(),
		bus:      bus.NewAsyncQueueBus(),
		registry: registry.New(runnerName),
		recorder: &recorder{name: "recorder"},
	}
	e.bus.Subscribe(e.recorder)
	e.bus.Start(e.ctx)
	t.Cleanup(e.bus.Stop)
	return e
}

// recorder collects every message delivered on the bus.
type recorder struct {
	name string
	mu   sync.Mutex
	got  []bus.Message
}

func (r *recorder) Name() string { return r.name }

func (r *recorder) OnBusMessage(_ context.Context, m bus.Message) {
	r.mu.Lock()
	r.got = append(r.got, m)
	r.mu.Unlock()
}

// awaitFrom waits for a message from source to reach the bus.
func (r *recorder) awaitFrom(t *testing.T, source string) bus.Message {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		for _, m := range r.got {
			if m.Source() == source {
				r.mu.Unlock()
				return m
			}
		}
		r.mu.Unlock()
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("no message from %q reached the bus", source)
	return nil
}

// sawFrom reports whether a message from source reached the bus within a short
// grace period. It is the negative counterpart of awaitFrom.
func (r *recorder) sawFrom(source string) bool {
	deadline := time.Now().Add(300 * time.Millisecond)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		for _, m := range r.got {
			if m.Source() == source {
				r.mu.Unlock()
				return true
			}
		}
		r.mu.Unlock()
		time.Sleep(5 * time.Millisecond)
	}
	return false
}

// dataMessage is a plain addressed message, which is what upstream's tests send.
func dataMessage(source, target string) bus.Message {
	m := &bus.TTSSpeakMessage{Text: "hello"}
	m.From, m.To = source, target
	return m
}

// awaitSent waits for the peer to have received n messages.
func awaitSent(t *testing.T, p *peer, n int) [][]byte {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if got := p.received(); len(got) >= n {
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
	got := p.received()
	t.Fatalf("the peer received %d messages, want %d", len(got), n)
	return nil
}

// sentNothing reports whether the peer stayed empty over a short grace period.
func sentNothing(p *peer) bool {
	time.Sleep(300 * time.Millisecond)
	return len(p.received()) == 0
}

// startClient builds a client proxy, attaches it and connects it.
func startClient(t *testing.T, e *env, url string, forward ...bus.Message) *proxy.Client {
	t.Helper()
	c := proxy.NewClient(proxy.ClientConfig{
		Name:            "proxy",
		URL:             url,
		RemoteWorker:    "worker",
		LocalWorker:     "voice",
		ForwardMessages: forward,
	})
	c.Attach(e.ctx, e.registry, e.bus.Bus)
	c.Start(e.ctx)
	c.OnActivated(e.ctx, nil)
	t.Cleanup(func() { c.Stop(context.WithoutCancel(e.ctx)) })
	return c
}

// startServer builds a server proxy over a dialed connection and starts it.
func startServer(t *testing.T, e *env, url string, forward ...bus.Message) *proxy.Server {
	t.Helper()
	conn, err := wsutil.Dial(e.ctx, url, nil, 1<<20)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	s := proxy.NewServer(proxy.ServerConfig{
		Name:            "gateway",
		Conn:            conn,
		LocalWorker:     "worker",
		RemoteWorker:    "voice",
		ForwardMessages: forward,
	})
	s.Attach(e.ctx, e.registry, e.bus.Bus)
	s.Start(e.ctx)
	t.Cleanup(func() { s.Stop(context.WithoutCancel(e.ctx)) })
	return s
}

//
// The client proxy.
//

// TestClientForwardsTargetedMessages is upstream's
// test_forwards_targeted_messages.
func TestClientForwardsTargetedMessages(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	c := startClient(t, e, p.serve(t))

	c.OnBusMessage(e.ctx, dataMessage("voice", "worker"))

	sent := awaitSent(t, p, 1)
	restored, err := bus.NewJSONSerializer(nil).Deserialize(sent[0])
	if err != nil {
		t.Fatalf("the message that crossed will not deserialize: %v", err)
	}
	if restored.Source() != "voice" || restored.Target() != "worker" {
		t.Errorf("crossed %q -> %q, want voice -> worker", restored.Source(), restored.Target())
	}
}

// TestClientSkipsMessagesForOtherWorkers is upstream's
// test_skips_messages_for_other_workers.
func TestClientSkipsMessagesForOtherWorkers(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	c := startClient(t, e, p.serve(t))

	c.OnBusMessage(e.ctx, dataMessage("voice", "other_task"))

	if !sentNothing(p) {
		t.Error("a message for another worker crossed")
	}
}

// TestClientSkipsBroadcastMessages is upstream's test_skips_broadcast_messages.
// A broadcast is addressed to nobody, so it is not the far side's.
func TestClientSkipsBroadcastMessages(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	c := startClient(t, e, p.serve(t))

	c.OnBusMessage(e.ctx, dataMessage("voice", ""))

	if !sentNothing(p) {
		t.Error("a broadcast crossed")
	}
}

// TestClientSkipsLocalMessages is upstream's test_skips_local_messages. A local
// message carries an object rather than data, which the far side could not act
// on even if it arrived.
func TestClientSkipsLocalMessages(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	c := startClient(t, e, p.serve(t))

	m := &bus.AddWorkerMessage{Worker: newStub("child")}
	m.From, m.To = "voice", "worker"
	c.OnBusMessage(e.ctx, m)

	if !sentNothing(p) {
		t.Error("a local message crossed")
	}
}

// TestClientAcceptsInboundForLocalWorker is upstream's
// test_accepts_inbound_for_local_worker.
func TestClientAcceptsInboundForLocalWorker(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	startClient(t, e, p.serve(t))

	raw, err := bus.NewJSONSerializer(nil).Serialize(dataMessage("worker", "voice"))
	if err != nil {
		t.Fatal(err)
	}
	p.send <- raw

	got := e.recorder.awaitFrom(t, "worker")
	if got.Target() != "voice" {
		t.Errorf("the message reached the bus addressed to %q, want voice", got.Target())
	}
}

// TestClientDropsInboundForOtherWorkers is upstream's
// test_drops_inbound_for_other_workers.
func TestClientDropsInboundForOtherWorkers(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	startClient(t, e, p.serve(t))

	raw, err := bus.NewJSONSerializer(nil).Serialize(dataMessage("worker", "other_task"))
	if err != nil {
		t.Fatal(err)
	}
	p.send <- raw

	if e.recorder.sawFrom("worker") {
		t.Error("a message addressed elsewhere was put on this bus")
	}
}

//
// The server proxy.
//

// TestServerForwardsMessagesFromLocalWorker is upstream's
// test_forwards_messages_from_local_worker.
func TestServerForwardsMessagesFromLocalWorker(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	s := startServer(t, e, p.serve(t))

	s.OnBusMessage(e.ctx, dataMessage("worker", "voice"))

	sent := awaitSent(t, p, 1)
	restored, err := bus.NewJSONSerializer(nil).Deserialize(sent[0])
	if err != nil {
		t.Fatalf("the message that crossed will not deserialize: %v", err)
	}
	if restored.Source() != "worker" || restored.Target() != "voice" {
		t.Errorf("crossed %q -> %q, want worker -> voice", restored.Source(), restored.Target())
	}
}

// TestServerSkipsMessagesFromOtherWorkers is upstream's
// test_skips_messages_from_other_workers. This side may carry the traffic of
// several workers and the client has no business seeing the rest.
func TestServerSkipsMessagesFromOtherWorkers(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	s := startServer(t, e, p.serve(t))

	s.OnBusMessage(e.ctx, dataMessage("other_task", "voice"))

	if !sentNothing(p) {
		t.Error("another worker's message crossed")
	}
}

// TestServerSkipsMessagesToOtherTargets is upstream's
// test_skips_messages_to_other_targets.
func TestServerSkipsMessagesToOtherTargets(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	s := startServer(t, e, p.serve(t))

	s.OnBusMessage(e.ctx, dataMessage("worker", "other_task"))

	if !sentNothing(p) {
		t.Error("a message for another target crossed")
	}
}

// TestServerAcceptsInboundForLocalWorker is upstream's
// test_accepts_inbound_for_local_worker.
func TestServerAcceptsInboundForLocalWorker(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	startServer(t, e, p.serve(t))

	raw, err := bus.NewJSONSerializer(nil).Serialize(dataMessage("voice", "worker"))
	if err != nil {
		t.Fatal(err)
	}
	p.send <- raw

	got := e.recorder.awaitFrom(t, "voice")
	if got.Target() != "worker" {
		t.Errorf("the message reached the bus addressed to %q, want worker", got.Target())
	}
}

// TestServerDropsInboundForOtherWorkers is upstream's
// test_drops_inbound_for_other_workers.
func TestServerDropsInboundForOtherWorkers(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	startServer(t, e, p.serve(t))

	raw, err := bus.NewJSONSerializer(nil).Serialize(dataMessage("voice", "other_task"))
	if err != nil {
		t.Fatal(err)
	}
	p.send <- raw

	if e.recorder.sawFrom("voice") {
		t.Error("a message addressed elsewhere was put on this bus")
	}
}

//
// The rules upstream has but does not test directly.
//

// TestForwardedTypesCrossWhoeverTheyAreAddressedTo checks the exception the
// rules are built around: a frame carries no target, so a bridged worker in
// another process would never see one under the addressing rule alone.
func TestForwardedTypesCrossWhoeverTheyAreAddressedTo(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	c := startClient(t, e, p.serve(t), &bus.FrameMessage{})

	m := &bus.FrameMessage{Frame: frames.NewTranscriptionFrame("bonjour", "user-1", "ts")}
	m.From = "voice" // the local worker, and addressed to nobody
	c.OnBusMessage(e.ctx, m)

	sent := awaitSent(t, p, 1)
	restored, err := bus.NewJSONSerializer(nil).Deserialize(sent[0])
	if err != nil {
		t.Fatalf("the frame that crossed will not deserialize: %v", err)
	}
	fm, ok := restored.(*bus.FrameMessage)
	if !ok {
		t.Fatalf("crossed as %T, want a FrameMessage", restored)
	}
	if tf, ok := fm.Frame.(*frames.TranscriptionFrame); !ok || tf.Text != "bonjour" {
		t.Errorf("the frame arrived as %v, want the transcription that was sent", fm.Frame)
	}
}

// TestForwardedTypesOnlyCrossFromTheLocalWorker checks the guard on that
// exception: a frame this process did not produce is not echoed back to the
// process it came from.
func TestForwardedTypesOnlyCrossFromTheLocalWorker(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	c := startClient(t, e, p.serve(t), &bus.FrameMessage{})

	m := &bus.FrameMessage{Frame: frames.NewTranscriptionFrame("bonjour", "user-1", "ts")}
	m.From = "somebody-else"
	c.OnBusMessage(e.ctx, m)

	if !sentNothing(p) {
		t.Error("a frame from another worker was echoed across")
	}
}

// TestClientAcceptsARegistrySnapshot checks the one message that arrives
// addressed to nobody and is still acted on: it is how this side learns which
// workers the far side has.
func TestClientAcceptsARegistrySnapshot(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	startClient(t, e, p.serve(t))

	m := &bus.WorkerRegistryMessage{
		Runner:  "remote-runner",
		Workers: []registry.WorkerRegistryEntry{{Name: "worker"}},
	}
	m.From = "gateway"
	raw, err := bus.NewJSONSerializer(nil).Serialize(m)
	if err != nil {
		t.Fatal(err)
	}
	p.send <- raw

	got := e.recorder.awaitFrom(t, "gateway")
	snapshot, ok := got.(*bus.WorkerRegistryMessage)
	if !ok {
		t.Fatalf("the snapshot reached the bus as %T", got)
	}
	if len(snapshot.Workers) != 1 || snapshot.Workers[0].Name != "worker" {
		t.Errorf("snapshot = %+v, want the one worker the far side named", snapshot.Workers)
	}
}

// TestServerAnnouncesTheLocalWorker checks the client is told the worker it is
// addressing exists. Without it the client would be sending to a name it has no
// reason to believe in.
func TestServerAnnouncesTheLocalWorker(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	s := startServer(t, e, p.serve(t))

	s.OnWorkerReady(e.ctx, registry.WorkerReadyData{WorkerName: "worker", Runner: runnerName})

	sent := awaitSent(t, p, 1)
	restored, err := bus.NewJSONSerializer(nil).Deserialize(sent[0])
	if err != nil {
		t.Fatalf("the announcement will not deserialize: %v", err)
	}
	snapshot, ok := restored.(*bus.WorkerRegistryMessage)
	if !ok {
		t.Fatalf("announced as %T, want a WorkerRegistryMessage", restored)
	}
	if len(snapshot.Workers) != 1 || snapshot.Workers[0].Name != "worker" {
		t.Errorf("announced %+v, want the local worker", snapshot.Workers)
	}
}

// TestServerSaysNothingAboutOtherWorkers checks only the worker this proxy
// carries is announced.
func TestServerSaysNothingAboutOtherWorkers(t *testing.T) {
	e := newEnv(t)
	p := newPeer()
	s := startServer(t, e, p.serve(t))

	s.OnWorkerReady(e.ctx, registry.WorkerReadyData{WorkerName: "somebody-else", Runner: runnerName})

	if !sentNothing(p) {
		t.Error("another worker's readiness was announced")
	}
}

// stub takes part in the protocol and does nothing else, so a local message has
// something to carry.
type stub struct{ *workers.Base }

func newStub(name string) *stub {
	w := &stub{}
	w.Base = workers.New(workers.Config{Name: name}, w)
	return w
}
