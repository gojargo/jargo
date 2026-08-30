// Package proxy carries bus messages between two processes over a WebSocket.
//
// A worker in one process and a worker in another cannot see each other's bus.
// A pair of proxies joins them: each sits on its own bus as an ordinary worker,
// forwards the messages meant for the far side, and puts what arrives from
// there onto its own bus. Neither worker knows the other is remote.
//
// The pair is a client and a server because one of them has to dial. The
// difference is only how the socket is obtained and which way the routing rules
// point; what travels, and how, is the same either way.
//
// Which messages cross is deliberately narrow. A message addressed to the
// worker on the far side crosses, and nothing else does, so a bus carrying a
// conversation's whole traffic does not put all of it on the wire. Message
// types named in ForwardMessages cross as well, addressed or not, which is how
// frames reach a bridged worker in another process.
package proxy

import (
	"context"
	"log/slog"
	"net/http"
	"reflect"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/service/wsutil"
)

// readLimit bounds a single message off the socket. A frame carrying audio is
// the large case, and a generous limit keeps a long turn from being cut off.
const readLimit = 8 << 20

// link is the socket a proxy holds, with the serializer it reads and writes
// through. Both proxies use one: they differ in how the socket arrives and in
// what they choose to send, not in how a message crosses.
type link struct {
	serializer bus.MessageSerializer

	mu   sync.Mutex
	conn *wsutil.Conn
	// writeMu serializes writes, which the WebSocket library requires.
	writeMu sync.Mutex
}

// set records the socket the proxy is now holding.
func (l *link) set(conn *wsutil.Conn) {
	l.mu.Lock()
	l.conn = conn
	l.mu.Unlock()
}

// get returns the socket, or nil once it has gone.
func (l *link) get() *wsutil.Conn {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.conn
}

// take removes the socket and returns what was there, so the side that noticed
// the connection end is the only one to report it.
func (l *link) take() *wsutil.Conn {
	l.mu.Lock()
	defer l.mu.Unlock()
	conn := l.conn
	l.conn = nil
	return conn
}

// send writes one message. It reports whether the connection has gone, which is
// what tells the caller to stop forwarding and say so.
func (l *link) send(ctx context.Context, m bus.Message) (gone bool) {
	conn := l.get()
	if conn == nil {
		return false
	}
	data, err := l.serializer.Serialize(m)
	if err != nil {
		slog.Error("proxy: a message could not be serialized and was not sent",
			"message", reflect.TypeOf(m).String(), "err", err)
		return false
	}

	l.writeMu.Lock()
	err = conn.Write(ctx, websocket.MessageBinary, data)
	l.writeMu.Unlock()
	if err == nil {
		return false
	}
	if ctx.Err() != nil {
		return false
	}
	slog.Warn("proxy: the connection closed, so nothing more is being forwarded", "err", err)
	return true
}

// receive reads one message. It reports the message, or that the connection has
// gone. A message that will not deserialize is reported and skipped: the far
// end may be running a build that knows a type this one does not, and one
// message it cannot read is no reason to drop the connection.
func (l *link) receive(ctx context.Context) (m bus.Message, ok bool, gone bool) {
	conn := l.get()
	if conn == nil {
		return nil, false, true
	}
	_, data, err := conn.Read(ctx)
	if err != nil {
		return nil, false, true
	}
	m, err = l.serializer.Deserialize(data)
	if err != nil {
		slog.Warn("proxy: a message from the far side could not be read", "err", err)
		return nil, false, false
	}
	return m, true, false
}

// close shuts the socket down, if one is still held.
func (l *link) close() {
	if conn := l.take(); conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
	}
}

// forwards reports whether m is one of the types named, which cross whoever
// they are addressed to.
func forwards(m bus.Message, named []bus.Message) bool {
	if len(named) == 0 {
		return false
	}
	t := reflect.TypeOf(m)
	for _, sample := range named {
		if reflect.TypeOf(sample) == t {
			return true
		}
	}
	return false
}

// crosses reports whether a message may leave this process at all. A local
// message never does: it carries something the far side could not act on, an
// object rather than data.
func crosses(m bus.Message) bool {
	_, local := m.(bus.LocalMessage)
	return !local
}

// dial opens the socket a client proxy talks over, sending headers with the
// handshake, which is where a proxy that has to authenticate does it.
func dial(ctx context.Context, url string, headers map[string]string) (*wsutil.Conn, error) {
	var header http.Header
	if len(headers) > 0 {
		header = http.Header{}
		for k, v := range headers {
			header.Set(k, v)
		}
	}
	return wsutil.Dial(ctx, url, header, readLimit)
}
