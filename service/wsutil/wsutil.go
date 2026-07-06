// Package wsutil holds small shared helpers for the provider WebSocket clients.
// The streaming STT and TTS providers each dial a vendor WebSocket the same way;
// this package factors out that dial boilerplate so a provider only writes the
// request and reads the vendor's message frames.
package wsutil

import (
	"context"
	"net/http"

	"github.com/coder/websocket"
)

// DefaultReadLimit bounds a single inbound WebSocket message. Providers stream
// base64-encoded audio chunks, which fit comfortably in 1 MiB.
const DefaultReadLimit = 1 << 20

// Dial opens a WebSocket to url with the given headers, closes the handshake
// response body, and applies readLimit when it is positive. The caller owns the
// returned connection and must Close it.
func Dial(ctx context.Context, url string, header http.Header, readLimit int64) (*websocket.Conn, error) {
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
	return conn, nil
}
