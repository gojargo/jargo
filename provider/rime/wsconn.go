package rime

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/wsutil"
)

// readerConn is one WebSocket connection, with the goroutine reading it.
//
// The reading runs apart from the receive loop so stopping the loop does not
// close the connection: a read whose context ends closes the connection under
// it, and closing it is the service's to do, telling Rime the stream has ended
// first.
type readerConn struct {
	conn *wsutil.Conn
	// reads carries each message read, and finally the error that ended the
	// reading.
	reads chan connRead
	// stop is closed when the connection is being closed, which ends the reading
	// goroutine once nothing takes its messages any more.
	stop     chan struct{}
	stopOnce sync.Once
	// readEnded is set once the reading has failed, which is the connection
	// closing as far as the service is concerned.
	readEnded atomic.Bool
}

// connRead is one message read from the connection, or the error that ended
// the reading.
type connRead struct {
	typ  websocket.MessageType
	data []byte
	err  error
}

// newReaderConn starts reading conn.
func newReaderConn(conn *wsutil.Conn) *readerConn {
	c := &readerConn{conn: conn, reads: make(chan connRead), stop: make(chan struct{})}
	go c.read()
	return c
}

// read reads the connection until it ends, handing each message to whoever
// receives from reads.
func (c *readerConn) read() {
	for {
		typ, data, err := c.conn.Read(context.Background())
		if err != nil {
			c.readEnded.Store(true)
		}
		select {
		case c.reads <- connRead{typ: typ, data: data, err: err}:
		case <-c.stop:
			return
		}
		if err != nil {
			return
		}
	}
}

// next waits for the next message, until ctx ends.
func (c *readerConn) next(ctx context.Context) (websocket.MessageType, []byte, error) {
	select {
	case <-ctx.Done():
		return 0, nil, ctx.Err()
	case r := <-c.reads:
		return r.typ, r.data, r.err
	}
}

// open reports whether the connection is still being read.
func (c *readerConn) open() bool { return !c.readEnded.Load() }

// halt stops handing messages on, for a connection about to be closed.
func (c *readerConn) halt() { c.stopOnce.Do(func() { close(c.stop) }) }

// drop closes the connection outright, for one with nothing left to say.
func (c *readerConn) drop() {
	c.halt()
	_ = c.conn.CloseNow()
}
