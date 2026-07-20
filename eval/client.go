package eval

import (
	"context"
	"encoding/json"
	"fmt"

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
	c := &client{conn: conn, incoming: make(chan rtvi.Incoming, 64)}
	go c.readLoop(ctx)
	return c, nil
}

// readLoop reads RTVI messages until the socket closes or ctx is canceled.
func (c *client) readLoop(ctx context.Context) {
	defer close(c.incoming)
	for {
		_, data, err := c.conn.Read(ctx)
		if err != nil {
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

// close shuts the connection down.
func (c *client) close() {
	_ = c.conn.Close(websocket.StatusNormalClosure, "")
}
