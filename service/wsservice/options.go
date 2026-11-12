package wsservice

import (
	"context"
	"net/http"
	"time"

	"github.com/gojargo/jargo/service/wsutil"
)

// Options are the connection settings every service holding a WebSocket open
// takes in its configuration: whether and how it reconnects, and how long it
// waits on a peer that has stopped answering. The zero value takes every
// default.
type Options struct {
	// ReconnectOnError says whether a failed connection is reconnected at all.
	// Nil enables it. False turns every failure into a reported error, for a
	// session that cannot be resumed by redialing.
	ReconnectOnError *bool
	// ReconnectBackoffMinWait is the shortest wait between two reconnection
	// attempts; zero uses four seconds.
	ReconnectBackoffMinWait time.Duration
	// ReconnectBackoffMaxWait is the longest wait between two reconnection
	// attempts; zero uses ten seconds.
	ReconnectBackoffMaxWait time.Duration
	// CloseTimeout bounds how long closing the connection waits for the peer to
	// answer the close. Nil uses wsutil.DefaultCloseTimeout; zero sends the
	// close without waiting, for a peer that never acknowledges it.
	CloseTimeout *time.Duration
	// SendTimeout bounds how long one send may wait on a peer that has stopped
	// reading. Nil uses wsutil.DefaultSendTimeout; zero waits as long as it
	// takes.
	SendTimeout *time.Duration
}

// Config is the reconnection part of the options, for a Base.
func (o Options) Config() Config {
	return Config{
		ReconnectOnError:        o.ReconnectOnError,
		ReconnectBackoffMinWait: o.ReconnectBackoffMinWait,
		ReconnectBackoffMaxWait: o.ReconnectBackoffMaxWait,
	}
}

// Apply sets the timeouts the options give on an open connection.
func (o Options) Apply(conn *wsutil.Conn) {
	if o.CloseTimeout != nil {
		conn.SetCloseTimeout(*o.CloseTimeout)
	}
	if o.SendTimeout != nil {
		conn.SetSendTimeout(*o.SendTimeout)
	}
}

// Dial opens a connection as wsutil.Dial does and applies the timeouts the
// options give to it.
func (o Options) Dial(ctx context.Context, url string, header http.Header, readLimit int64) (*wsutil.Conn, error) {
	conn, err := wsutil.Dial(ctx, url, header, readLimit)
	if err != nil {
		return nil, err
	}
	o.Apply(conn)
	return conn, nil
}

// Optioner is an optional interface a service's connector implements to hand
// the base the options its configuration carries.
type Optioner interface {
	WebsocketOptions() Options
}
