package eval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/processor/rtvi"
)

// readLimit bounds a single inbound message; RTVI control messages are small.
const readLimit = 1 << 20

// client is an RTVI client over a plain WebSocket. It reads server messages in
// the background and delivers them on incoming.
type client struct {
	conn     *websocket.Conn
	incoming chan rtvi.Incoming

	// mu guards readErr, which the reader writes and the session reads.
	mu sync.Mutex
	// readErr is why reading stopped, set before incoming is closed.
	readErr error
}

// dial connects to a bot's RTVI WebSocket endpoint and starts reading messages.
func dial(ctx context.Context, url string) (*client, error) {
	conn, resp, err := websocket.Dial(ctx, url, nil)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("eval: dial %s: %w", url, err)
	}
	conn.SetReadLimit(readLimit)
	// A generous buffer so events emitted while the harness is busy streaming a
	// turn's audio (audio mode) are not dropped before matching reads them.
	c := &client{conn: conn, incoming: make(chan rtvi.Incoming, 256)}
	go c.readLoop(ctx)
	return c, nil
}

// errConnectTimedOut is returned when the connect timeout passed without a
// single attempt to reach the bot failing on its own.
//
//nolint:gochecknoglobals // sentinel error
var errConnectTimedOut = errors.New("timed out")

// waitForBot waits until the bot accepts connections, for up to timeout (zero
// uses DefaultConnectTimeout).
//
// It retries a plain TCP connect until the bot is listening, and stops short of
// the WebSocket handshake on purpose: a full connection would make the bot
// greet, and that connection would then be thrown away. It returns the last
// connect error when the bot never accepted in time.
func waitForBot(ctx context.Context, botURL string, timeout time.Duration) error {
	if timeout <= 0 {
		timeout = DefaultConnectTimeout
	}
	u, err := url.Parse(botURL)
	if err != nil {
		return err
	}
	host, port := u.Hostname(), u.Port()
	if host == "" {
		host = "localhost"
	}
	if port == "" {
		port = "80"
		if u.Scheme == "wss" {
			port = "443"
		}
	}
	addr := net.JoinHostPort(host, port)
	deadline := time.Now().Add(timeout)
	var (
		dialer  net.Dialer
		lastErr error
	)
	for time.Now().Before(deadline) {
		attemptCtx, cancel := context.WithDeadline(ctx, deadline)
		conn, err := dialer.DialContext(attemptCtx, "tcp", addr)
		cancel()
		if err == nil {
			_ = conn.Close()
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// Not accepting connections yet.
		lastErr = err
		if err := sleep(ctx, connectRetry); err != nil {
			return err
		}
	}
	if lastErr == nil {
		return errConnectTimedOut
	}
	return lastErr
}

// readLoop reads RTVI messages until the socket closes or ctx is canceled.
func (c *client) readLoop(ctx context.Context) {
	defer close(c.incoming)
	for {
		_, data, err := c.conn.Read(ctx)
		if err != nil {
			c.mu.Lock()
			c.readErr = err
			c.mu.Unlock()
			return
		}
		in, err := rtvi.ParseIncoming(data)
		if err != nil || in.Label != rtvi.MessageLabel {
			continue
		}
		select {
		case c.incoming <- in:
		case <-ctx.Done():
			return
		}
	}
}

// send writes an RTVI message to the bot.
func (c *client) send(ctx context.Context, msg rtvi.Message) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.conn.Write(ctx, websocket.MessageText, data)
}

// closedBy is why the connection stopped reading, or nil while it still reads.
func (c *client) closedBy() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.readErr
}

// closeDescription is how the bot closed the connection, with its close code
// and reason when it sent them.
func closeDescription(err error) string {
	var ce websocket.CloseError
	if !errors.As(err, &ce) {
		return "the bot closed the connection before bot-ready"
	}
	reason := ""
	if ce.Reason != "" {
		reason = " " + ce.Reason
	}
	return fmt.Sprintf("the bot closed the connection before bot-ready (%d%s)", int(ce.Code), reason)
}

// close shuts the connection down.
func (c *client) close() {
	_ = c.conn.Close(websocket.StatusNormalClosure, "")
}
