// Package wsutil holds small shared helpers for the provider WebSocket clients.
// The streaming STT and TTS providers each dial a vendor WebSocket the same way;
// this package factors out that dial boilerplate so a provider only writes the
// request and reads the vendor's message frames.
package wsutil

import (
	"context"
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

// Dial opens a WebSocket to url with the given headers, closes the handshake
// response body, and applies readLimit when it is positive. The caller owns the
// returned connection and must Close it.
func Dial(ctx context.Context, url string, header http.Header, readLimit int64) (*Conn, error) {
	conn, resp, err := websocket.Dial(ctx, url, &websocket.DialOptions{HTTPHeader: header})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	if readLimit > 0 {
		conn.SetReadLimit(readLimit)
	}
	return &Conn{Conn: conn, closeTimeout: DefaultCloseTimeout}, nil
}
