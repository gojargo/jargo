// Package wsutil holds small shared helpers for the provider WebSocket clients.
// The streaming STT and TTS providers each dial a vendor WebSocket the same way;
// this package factors out that dial boilerplate so a provider only writes the
// request and reads the vendor's message frames.
package wsutil

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	neturl "net/url"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

// DefaultReadLimit bounds a single inbound WebSocket message. It guards against
// a server that streams without end, not against large messages: a TTS provider
// generating a long sentence in one go sends base64 audio well past a megabyte,
// and a limit tight enough to cut that off fails the synthesis mid-reply.
const DefaultReadLimit = 16 << 20

// DefaultCloseTimeout bounds how long a close waits for the peer to answer.
// Disconnecting happens while a service handles the EndFrame, before the frame
// carries on downstream, so a close the peer never acknowledges holds up the end
// of the pipeline for that long. The library waits 5s, which is long enough to
// be felt once several WebSocket services tear down one after another.
const DefaultCloseTimeout = 2 * time.Second

// DefaultSendTimeout is the default ceiling on a single send. A send only waits
// once the peer has stopped taking data and the write buffers are full, so on a
// working connection it returns at once.
const DefaultSendTimeout = 5 * time.Second

// DefaultPingInterval is how often a dialed connection pings its peer, and
// DefaultPingTimeout how long it waits for the answer before dropping the
// connection. The pings keep an idle connection alive through a NAT or proxy
// that expires quiet flows, and a peer that stopped answering is found within
// about the two together rather than when TCP gives up, minutes later.
const (
	DefaultPingInterval = 20 * time.Second
	DefaultPingTimeout  = 20 * time.Second
)

// Conn is a WebSocket connection whose closing handshake and sends are bounded,
// so a peer that never answers, or stops reading, cannot hold up a teardown or
// the service writing to it. A dialed one also pings its peer, and drops the
// connection when the peer stops answering. Everything else is the underlying
// connection's.
type Conn struct {
	*websocket.Conn
	// closeTimeout is how long Close waits for the peer's answer.
	closeTimeout time.Duration
	// sendTimeout is how long one Write may wait on a peer that has stopped
	// reading; zero waits indefinitely.
	sendTimeout time.Duration
	// reading counts the reads in progress. The library only takes in a pong
	// while a read is in progress, so a ping that goes unanswered says the peer
	// is gone only while something was reading.
	reading atomic.Int32
	// stopPinging ends the keepalive, and keepaliveDone is closed once it has
	// ended. Both are nil when the connection has none.
	stopPinging   context.CancelFunc
	keepaliveDone chan struct{}
}

// Read reads the next message. It is the underlying connection's, marked as a
// read in progress for the keepalive.
func (c *Conn) Read(ctx context.Context) (websocket.MessageType, []byte, error) {
	c.reading.Add(1)
	defer c.reading.Add(-1)
	return c.Conn.Read(ctx)
}

// startKeepalive pings the peer every interval and drops the connection when a
// ping goes unanswered for timeout.
func (c *Conn) startKeepalive(interval, timeout time.Duration) {
	ctx, cancel := context.WithCancel(context.Background())
	c.stopPinging = cancel
	c.keepaliveDone = make(chan struct{})
	go func() {
		defer close(c.keepaliveDone)
		c.keepalive(ctx, interval, timeout)
	}()
}

// keepalive runs the pings until the connection closes or the peer stops
// answering. Dropping the connection fails the read in progress, which is how
// the service reading it learns the peer is gone and reconnects.
func (c *Conn) keepalive(ctx context.Context, interval, timeout time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		if c.reading.Load() == 0 {
			// Nothing would take in the pong, so its absence would say nothing.
			continue
		}
		pingCtx, cancel := context.WithTimeout(ctx, timeout)
		err := c.Conn.Ping(pingCtx)
		cancel()
		switch {
		case err == nil:
			continue
		case ctx.Err() != nil:
			return
		case !errors.Is(err, context.DeadlineExceeded):
			// The connection is already closed, or the ping could not be sent,
			// which closes it: the read in progress fails either way.
			return
		case c.reading.Load() == 0:
			// The read ended while the ping waited, so the pong may simply have
			// gone unread.
			continue
		}
		slog.Warn("websocket peer did not answer a keepalive ping in time, dropping the connection",
			"timeout", timeout)
		_ = c.Conn.CloseNow()
		return
	}
}

// CloseNow closes the connection without the closing handshake.
func (c *Conn) CloseNow() error {
	c.endKeepalive()
	return c.Conn.CloseNow()
}

// endKeepalive stops the pings, if the connection sends any.
func (c *Conn) endKeepalive() {
	if c.stopPinging != nil {
		c.stopPinging()
	}
}

// Write sends a message, dropping the connection if the send stalls.
//
// The library puts no bound of its own on a send: once the peer stops reading
// and the write buffer fills, the send waits until the connection drops, and TCP
// can take many minutes to notice, holding up the service writing and every
// processor after it. The send is bounded by the send timeout instead. A send
// that outlasts it closes the connection, as the library does whenever a
// write's context ends, so the send fails the way any dropped connection does
// and the service's reconnect handling takes over.
func (c *Conn) Write(ctx context.Context, typ websocket.MessageType, p []byte) error {
	if c.sendTimeout <= 0 {
		return c.Conn.Write(ctx, typ, p)
	}
	sendCtx, cancel := context.WithTimeout(ctx, c.sendTimeout)
	defer cancel()
	err := c.Conn.Write(sendCtx, typ, p)
	if err != nil && ctx.Err() == nil && errors.Is(sendCtx.Err(), context.DeadlineExceeded) {
		slog.Warn("websocket send did not complete in time; the peer has stopped reading, "+
			"dropping the connection", "timeout", c.sendTimeout)
		return fmt.Errorf("%w within %s: %w", ErrSendTimeout, c.sendTimeout, err)
	}
	return err
}

// ErrSendTimeout is wrapped by the error of a send that outlasted the send
// timeout, which dropped the connection.
//
//nolint:gochecknoglobals // sentinel error
var ErrSendTimeout = errors.New("wsutil: websocket send did not complete")

// SetSendTimeout sets how long one Write may wait on a peer that has stopped
// reading before the connection is dropped. Zero or less waits indefinitely.
func (c *Conn) SetSendTimeout(d time.Duration) {
	c.sendTimeout = max(d, 0)
}

// Close closes the connection, giving up on the closing handshake once the
// timeout passes rather than waiting the library's own deadline out. The peer
// not answering is logged, since a teardown that quietly cost the full timeout
// would otherwise go unnoticed; the library drops the connection behind us.
func (c *Conn) Close(code websocket.StatusCode, reason string) error {
	c.endKeepalive()
	done := make(chan error, 1)
	go func() { done <- c.Conn.Close(code, reason) }()
	select {
	case err := <-done:
		return err
	case <-time.After(c.closeTimeout):
		slog.Debug("peer did not acknowledge the websocket close, dropping the connection",
			"timeout", c.closeTimeout)
		return nil
	}
}

// SetCloseTimeout sets how long Close waits for the peer to answer. Raise it for
// a peer that needs longer to close gracefully; zero sends the close and does
// not wait at all, for a peer that never acknowledges it. A negative value is
// ignored.
func (c *Conn) SetCloseTimeout(d time.Duration) {
	if d >= 0 {
		c.closeTimeout = d
	}
}

// HandshakeError is returned when the server refused the WebSocket handshake. It
// carries the status it refused with, so a caller can tell a request the server
// will go on refusing (a rejected key, a model the account cannot use) from a
// failure that may not repeat.
type HandshakeError struct {
	// StatusCode is the HTTP status the server refused the handshake with.
	StatusCode int
	// Header is the header of the response that refused the handshake, where a
	// provider may explain the refusal.
	Header http.Header
	// Err is the underlying dial failure.
	Err error
}

// Error implements error.
func (e *HandshakeError) Error() string {
	return fmt.Sprintf("websocket handshake refused with status %d: %v", e.StatusCode, e.Err)
}

// Unwrap returns the underlying dial failure.
func (e *HandshakeError) Unwrap() error { return e.Err }

// HTTPStatusCode implements the interface the shared error classification looks
// for, so a refused handshake is classified by the status the server refused
// with: rejected credentials from a server that is merely unwell.
func (e *HandshakeError) HTTPStatusCode() int { return e.StatusCode }

// Permanent reports whether the server refused the request itself, rather than
// failing in a way that might not repeat. Retrying a refusal only spends the
// time before the caller learns of it.
func (e *HandshakeError) Permanent() bool {
	return e.StatusCode >= http.StatusBadRequest && e.StatusCode < http.StatusInternalServerError
}

// Permanent reports whether err is a refusal the server will repeat, so there is
// nothing to gain by dialing again.
func Permanent(err error) bool {
	var he *HandshakeError
	return errors.As(err, &he) && he.Permanent()
}

// Adopt wraps a connection that was accepted rather than dialed, so a server
// side gets the same bounded close a dialed one has. It applies readLimit when
// that is positive.
//
// The caller owns the connection either way: Accept hands it over, and this only
// takes over how it is closed.
func Adopt(conn *websocket.Conn, readLimit int64) *Conn {
	if readLimit > 0 {
		conn.SetReadLimit(readLimit)
	}
	return &Conn{Conn: conn, closeTimeout: DefaultCloseTimeout}
}

// Dial opens a WebSocket to url with the given headers, closes the handshake
// response body, and applies readLimit when it is positive. The caller owns the
// returned connection and must Close it. A handshake the server refused comes
// back as a *HandshakeError carrying the status.
//
// The connection pings the peer every DefaultPingInterval while it is being
// read, and drops itself when a ping goes unanswered for DefaultPingTimeout.
//
// A dial that fails reports the host it tried and why, but never the query: a
// provider may carry its credential there, and the error ends up in the log.
func Dial(ctx context.Context, url string, header http.Header, readLimit int64) (*Conn, error) {
	return dial(ctx, url, header, readLimit, DefaultPingInterval, DefaultPingTimeout)
}

// dial is Dial with the keepalive's timing given.
func dial(
	ctx context.Context, url string, header http.Header, readLimit int64,
	pingInterval, pingTimeout time.Duration,
) (*Conn, error) {
	conn, resp, err := websocket.Dial(ctx, url, &websocket.DialOptions{HTTPHeader: header})
	status := 0
	var respHeader http.Header
	if resp != nil {
		status = resp.StatusCode
		respHeader = resp.Header
		if resp.Body != nil {
			_ = resp.Body.Close()
		}
	}
	if err != nil {
		err = redactDialError(err)
		if status != 0 {
			return nil, &HandshakeError{StatusCode: status, Header: respHeader, Err: err}
		}
		return nil, err
	}
	if readLimit > 0 {
		conn.SetReadLimit(readLimit)
	}
	c := &Conn{Conn: conn, closeTimeout: DefaultCloseTimeout, sendTimeout: DefaultSendTimeout}
	c.startKeepalive(pingInterval, pingTimeout)
	return c, nil
}

// redactDialError rewrites a dial error so it no longer carries the query of
// the URL it dialed. The URL is spelled out by the HTTP client in the error
// text itself, which the library then wraps, so the whole message is rebuilt
// around the cause rather than the URL replaced in place.
func redactDialError(err error) error {
	var ue *neturl.Error
	if !errors.As(err, &ue) {
		return err
	}
	u, perr := neturl.Parse(ue.URL)
	if perr == nil && u.RawQuery == "" && u.User == nil && u.Fragment == "" {
		return err
	}
	// A URL that does not parse is left out altogether, since nothing says
	// which part of it is safe to show.
	redacted := ""
	if perr == nil {
		u.RawQuery, u.User, u.Fragment = "", nil, ""
		redacted = u.String()
	}
	return fmt.Errorf("websocket dial: %w", &neturl.Error{Op: ue.Op, URL: redacted, Err: ue.Err})
}
