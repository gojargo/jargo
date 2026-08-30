package proxy

import (
	"context"
	"log/slog"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/workers"
)

// EventConnected fires once the client has opened its connection to the server.
// EventDisconnected fires when that connection ends.
//
// Neither carries an argument: the proxy raising it is the source, and the
// socket behind it belongs to the proxy.
const (
	EventConnected    = "on_connected"
	EventDisconnected = "on_disconnected"
)

// ClientConfig configures a Client.
type ClientConfig struct {
	// Name is what other workers on this bus address the proxy by.
	Name string
	// URL is the server proxy's WebSocket endpoint.
	URL string
	// RemoteWorker is the name of the worker on the far side. A message
	// addressed to it is what crosses, and nothing else is.
	RemoteWorker string
	// LocalWorker is the name of the worker on this side that answers. Only a
	// message arriving addressed to it is put on this bus; anything else is
	// dropped, since it was not this process's to act on.
	LocalWorker string
	// ForwardMessages are message types that cross whoever they are addressed
	// to, given as sample values (&bus.FrameMessage{}). It is how frames reach a
	// bridged worker in another process. Outbound, only the ones this side's
	// worker sent cross.
	ForwardMessages []bus.Message
	// Headers are sent with the WebSocket handshake, which is where a proxy that
	// has to authenticate does it.
	Headers map[string]string
	// Serializer converts messages to bytes and back; nil uses the JSON one.
	Serializer bus.MessageSerializer
	// Active reports whether the proxy connects as soon as it starts. Nil leaves
	// it inactive, because connecting is almost always something another event
	// decides: a client arriving, a call beginning. Activate it to connect.
	Active *bool
}

// Client forwards bus messages to a worker in another process, over a
// connection it opens itself.
//
// It is an ordinary worker on this bus. A message addressed to the worker named
// by RemoteWorker crosses; what arrives addressed to LocalWorker is put on this
// bus. Neither worker knows the other is remote.
//
// It starts inactive unless told otherwise, so nothing dials until something
// decides it should.
type Client struct {
	*workers.Base
	cfg  ClientConfig
	link *link

	mu       sync.Mutex
	stopRecv context.CancelFunc
	recvWG   sync.WaitGroup
}

// NewClient builds a client proxy.
func NewClient(cfg ClientConfig) *Client {
	serializer := cfg.Serializer
	if serializer == nil {
		serializer = bus.NewJSONSerializer(nil)
	}
	inactive := false
	active := cfg.Active
	if active == nil {
		active = &inactive
	}

	c := &Client{cfg: cfg, link: &link{serializer: serializer}}
	c.Base = workers.New(workers.Config{Name: cfg.Name, Active: active}, c)
	c.Events().Register(EventConnected, false)
	c.Events().Register(EventDisconnected, false)
	return c
}

// OnActivated opens the connection and starts reading from it.
func (c *Client) OnActivated(ctx context.Context, args map[string]any) {
	c.Base.OnActivated(ctx, args)

	slog.Debug("proxy: connecting", "worker", c.Name(), "url", c.cfg.URL)
	conn, err := dial(ctx, c.cfg.URL, c.cfg.Headers)
	if err != nil {
		slog.Error("proxy: could not connect", "worker", c.Name(), "url", c.cfg.URL, "err", err)
		return
	}
	c.link.set(conn)
	slog.Debug("proxy: connected", "worker", c.Name(), "url", c.cfg.URL)
	c.Events().Call(ctx, EventConnected, c)

	recvCtx, cancel := context.WithCancel(ctx)
	c.mu.Lock()
	c.stopRecv = cancel
	c.mu.Unlock()
	c.recvWG.Add(1)
	go c.receiveLoop(recvCtx)
}

// Stop closes the connection and waits for the read to finish.
func (c *Client) Stop(ctx context.Context) {
	c.mu.Lock()
	cancel := c.stopRecv
	c.stopRecv = nil
	c.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	c.link.close()
	c.recvWG.Wait()
	c.Base.Stop(ctx)
}

// OnBusMessage forwards what is meant for the far side.
//
// A message addressed to the remote worker crosses. A message of a forwarded
// type crosses only when this side's own worker sent it, which is what keeps a
// frame from being echoed back to the process it came from.
func (c *Client) OnBusMessage(ctx context.Context, m bus.Message) {
	c.Base.OnBusMessage(ctx, m)

	if c.link.get() == nil || !crosses(m) {
		return
	}
	switch {
	case m.Target() == c.cfg.RemoteWorker:
	case forwards(m, c.cfg.ForwardMessages) && m.Source() == c.cfg.LocalWorker:
	default:
		return
	}
	if gone := c.link.send(ctx, m); gone {
		c.reportDisconnected(ctx)
	}
}

// receiveLoop reads what the far side sends and puts it on this bus.
func (c *Client) receiveLoop(ctx context.Context) {
	defer c.recvWG.Done()
	for {
		m, ok, gone := c.link.receive(ctx)
		if gone {
			if ctx.Err() == nil {
				slog.Warn("proxy: the connection closed", "worker", c.Name())
				c.reportDisconnected(ctx)
			}
			return
		}
		if !ok {
			continue
		}
		if !c.accepts(m) {
			slog.Warn("proxy: dropped a message addressed elsewhere",
				"worker", c.Name(), "target", m.Target())
			continue
		}
		c.SendBusMessage(ctx, m)
	}
}

// accepts reports whether an arriving message is this process's to act on.
//
// A registry snapshot is taken whatever it is addressed to: it is how this side
// learns which workers the far side has, and it is addressed to nobody.
func (c *Client) accepts(m bus.Message) bool {
	if _, isRegistry := m.(*bus.WorkerRegistryMessage); isRegistry {
		return true
	}
	if forwards(m, c.cfg.ForwardMessages) {
		return true
	}
	return m.Target() == c.cfg.LocalWorker
}

// reportDisconnected drops the socket and raises the event, once.
func (c *Client) reportDisconnected(ctx context.Context) {
	if conn := c.link.take(); conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
		c.Events().Call(ctx, EventDisconnected, c)
	}
}
