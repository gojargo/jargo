package proxy

import (
	"context"
	"log/slog"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/gojargo/jargo/workers"
)

// EventClientConnected fires once the server proxy is ready to carry messages
// for the client that connected. EventClientDisconnected fires when that client
// goes.
//
// Neither carries an argument: the proxy raising it is the source, and the
// socket behind it belongs to the proxy.
const (
	EventClientConnected    = "on_client_connected"
	EventClientDisconnected = "on_client_disconnected"
)

// ServerConfig configures a Server.
type ServerConfig struct {
	// Name is what other workers on this bus address the proxy by.
	Name string
	// Conn is the accepted WebSocket connection to the client, which the
	// endpoint upgrading the request hands over.
	Conn *wsutil.Conn
	// LocalWorker is the name of the worker on this side. Only what it sends
	// crosses, and only what arrives addressed to it is put on this bus.
	LocalWorker string
	// RemoteWorker is the name of the worker on the client. A message addressed
	// to it is what crosses.
	RemoteWorker string
	// ForwardMessages are message types that cross whoever they are addressed
	// to, given as sample values (&bus.FrameMessage{}). It is how frames reach a
	// bridged worker in another process. Outbound, only the ones the local
	// worker sent cross.
	ForwardMessages []bus.Message
	// Serializer converts messages to bytes and back; nil uses the JSON one.
	Serializer bus.MessageSerializer
}

// Server carries bus messages for a worker in another process, over a
// connection that process opened.
//
// It is the far half of a Client: the same traffic, the same rules, and the
// socket arriving rather than being dialed. Build one per accepted connection,
// from the endpoint that upgraded the request, and add it to the runner.
//
// It also tells the client when the local worker becomes ready, so the client's
// side learns that the worker it is addressing exists.
type Server struct {
	*workers.Base
	cfg  ServerConfig
	link *link

	mu       sync.Mutex
	stopRecv context.CancelFunc
	recvWG   sync.WaitGroup
}

// NewServer builds a server proxy over an accepted connection.
func NewServer(cfg ServerConfig) *Server {
	serializer := cfg.Serializer
	if serializer == nil {
		serializer = bus.NewJSONSerializer(nil)
	}
	s := &Server{cfg: cfg, link: &link{serializer: serializer}}
	s.link.set(cfg.Conn)
	s.Base = workers.New(workers.Config{Name: cfg.Name}, s)
	s.Events().Register(EventClientConnected, false)
	s.Events().Register(EventClientDisconnected, false)
	return s
}

// Start begins reading from the client and watches the local worker, so the
// client can be told when it is ready.
func (s *Server) Start(ctx context.Context) {
	s.Base.Start(ctx)

	slog.Debug("proxy: ready to carry messages for the client", "worker", s.Name())
	s.Events().Call(ctx, EventClientConnected, s)

	recvCtx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.stopRecv = cancel
	s.mu.Unlock()
	s.recvWG.Add(1)
	go s.receiveLoop(recvCtx)

	s.WatchWorkers(ctx, s.cfg.LocalWorker)
}

// Stop closes the connection and waits for the read to finish.
func (s *Server) Stop(ctx context.Context) {
	s.mu.Lock()
	cancel := s.stopRecv
	s.stopRecv = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	s.link.close()
	s.recvWG.Wait()
	s.Base.Stop(ctx)
}

// OnWorkerReady tells the client that the local worker exists and can be
// addressed. Without it the client's side would be sending to a name it has no
// reason to believe in.
func (s *Server) OnWorkerReady(ctx context.Context, data registry.WorkerReadyData) {
	s.Base.OnWorkerReady(ctx, data)

	if s.link.get() == nil || data.WorkerName != s.cfg.LocalWorker {
		return
	}
	slog.Debug("proxy: telling the client the local worker is ready",
		"worker", s.Name(), "local", s.cfg.LocalWorker)

	m := &bus.WorkerRegistryMessage{
		Runner:  data.Runner,
		Workers: []registry.WorkerRegistryEntry{{Name: s.cfg.LocalWorker}},
	}
	m.From = s.Name()
	if gone := s.link.send(ctx, m); gone {
		s.reportDisconnected(ctx)
	}
}

// OnBusMessage forwards what the local worker sends to the client.
//
// Only the local worker's own messages cross: this side may carry the traffic
// of several workers and the client has no business seeing the rest. Of those,
// a message addressed to the remote worker crosses, as does one of a forwarded
// type whatever it is addressed to.
func (s *Server) OnBusMessage(ctx context.Context, m bus.Message) {
	s.Base.OnBusMessage(ctx, m)

	if s.link.get() == nil || !crosses(m) || m.Source() != s.cfg.LocalWorker {
		return
	}
	if m.Target() != s.cfg.RemoteWorker && !forwards(m, s.cfg.ForwardMessages) {
		return
	}
	if gone := s.link.send(ctx, m); gone {
		s.reportDisconnected(ctx)
	}
}

// receiveLoop reads what the client sends and puts it on this bus.
func (s *Server) receiveLoop(ctx context.Context) {
	defer s.recvWG.Done()
	for {
		m, ok, gone := s.link.receive(ctx)
		if gone {
			if ctx.Err() == nil {
				slog.Warn("proxy: the client disconnected", "worker", s.Name())
				s.reportDisconnected(ctx)
			}
			return
		}
		if !ok {
			continue
		}
		if !forwards(m, s.cfg.ForwardMessages) && m.Target() != s.cfg.LocalWorker {
			slog.Warn("proxy: dropped a message addressed elsewhere",
				"worker", s.Name(), "target", m.Target())
			continue
		}
		s.SendBusMessage(ctx, m)
	}
}

// reportDisconnected drops the socket and raises the event, once.
func (s *Server) reportDisconnected(ctx context.Context) {
	if conn := s.link.take(); conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
		s.Events().Call(ctx, EventClientDisconnected, s)
	}
}
