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

// Conn is a WebSocket connection whose closing handshake is bounded, so a peer
// that never answers cannot hold up a teardown. Everything else is the
// underlying connection's.
type Conn struct {
	*websocket.Conn
	// closeTimeout is how long Close waits for the peer's answer.
	closeTimeout time.Duration
}

// Close closes the connection, giving up on the closing handshake once the
// timeout passes rather than waiting the library's own deadline out. The peer
// not answering is logged, since a teardown that quietly cost the full timeout
// would otherwise go unnoticed; the library drops the connection behind us.
func (c *Conn) Close(code websocket.StatusCode, reason string) error {
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
// a peer that needs longer to close gracefully.
func (c *Conn) SetCloseTimeout(d time.Duration) {
	if d > 0 {
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
	// Err is the underlying dial failure.
	Err error
}

// Error implements error.
func (e *HandshakeError) Error() string {
	return fmt.Sprintf("websocket handshake refused with status %d: %v", e.StatusCode, e.Err)
}

// Unwrap returns the underlying dial failure.
func (e *HandshakeError) Unwrap() error { return e.Err }

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

// Dial opens a WebSocket to url with the given headers, closes the handshake
// response body, and applies readLimit when it is positive. The caller owns the
// returned connection and must Close it. A handshake the server refused comes
// back as a *HandshakeError carrying the status.
func Dial(ctx context.Context, url string, header http.Header, readLimit int64) (*Conn, error) {
	conn, resp, err := websocket.Dial(ctx, url, &websocket.DialOptions{HTTPHeader: header})
	status := 0
	if resp != nil {
		status = resp.StatusCode
		if resp.Body != nil {
			_ = resp.Body.Close()
		}
	}
	if err != nil {
		if status != 0 {
			return nil, &HandshakeError{StatusCode: status, Err: err}
		}
		return nil, err
	}
	if readLimit > 0 {
		conn.SetReadLimit(readLimit)
	}
	return &Conn{Conn: conn, closeTimeout: DefaultCloseTimeout}, nil
}
