// Package wsservice is the shared base for services that talk to a provider
// over a WebSocket held open for the length of a call.
//
// Such a connection drops for ordinary reasons: a network blip, a server
// recycling, a proxy timing out an idle socket. Left alone that ends the
// service for the rest of the call, so the base reconnects. It runs the receive
// loop, and when the connection fails it retries with an exponential backoff,
// reporting each failed attempt without ending the call. Retrying does have a
// limit: a connection that keeps failing the instant it is established is not
// waiting on the network, so the base gives up rather than retry forever.
//
// A service supplies the provider-specific half through Handler and owns the
// socket itself; the base only says when to open it, close it, and read from it.
package wsservice

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/utils/network"
)

// DefaultMaxRetries is how many times the base redials before it reports the
// connection lost.
const DefaultMaxRetries = 3

// ErrVerification is returned by a reconnect attempt that redialed without an
// error but did not leave a live connection behind.
//
//nolint:gochecknoglobals // sentinel error
var ErrVerification = errors.New("wsservice: websocket reconnection failed verification")

// ErrNotConnected is returned by a send that had no connection to send on.
//
//nolint:gochecknoglobals // sentinel error
var ErrNotConnected = errors.New("wsservice: no websocket connected")

// ReportError reports a connection error to the pipeline. The base calls it for
// every failed reconnect attempt and once more when it stops retrying, so a
// service typically pushes a non-fatal error frame: the call continues, and the
// application decides what a lost provider connection means for it.
type ReportError func(ctx context.Context, message string)

// Handler is the service-specific half of a WebSocket service. The service owns
// the socket; these are the three things the base needs to drive it, plus the
// liveness check that decides whether a redial actually worked.
type Handler interface {
	// ConnectWebsocket opens the connection. It is called for the redial of
	// every reconnect attempt, so it must be safe to call repeatedly.
	ConnectWebsocket(ctx context.Context) error
	// DisconnectWebsocket closes the connection. It is called before every
	// redial, so it must tolerate there being nothing to close.
	DisconnectWebsocket(ctx context.Context) error
	// ReceiveMessages reads from the connection until it ends, handling each
	// message. Returning nil means the peer closed the connection gracefully;
	// returning an error means it failed. Either way the base decides whether
	// to reconnect and calls it again.
	ReceiveMessages(ctx context.Context) error
	// Connected reports whether there is a live connection. The base asks after
	// a redial, because a service that reports its dial failure by leaving no
	// connection behind would otherwise look like it had succeeded.
	Connected() bool
}

// Config configures a Base. A zero field takes its default.
type Config struct {
	// ReconnectOnError says whether a failed connection is reconnected at all.
	// Nil enables it. Setting it to false turns every failure into a reported
	// error and ends the receive loop, for a service whose session cannot be
	// resumed by redialing.
	ReconnectOnError *bool
	// MaxRetries is how many times a reconnect redials before giving up.
	MaxRetries int
	// QuickFailure configures when repeated instant failures end the retrying.
	QuickFailure network.QuickFailureConfig
}

// Base runs the receive loop and the reconnection around it. Embed it in a
// service that holds a WebSocket open, and hand it the Handler that owns the
// socket.
type Base struct {
	handler          Handler
	reconnectOnError bool
	maxRetries       int
	tracker          *network.QuickFailureTracker

	mu                  sync.Mutex
	disconnecting       bool
	reconnectInProgress bool
	lastConnect         time.Time

	// now and sleep are the clock, replaced in tests so a backoff does not cost
	// the wall-clock time it asks for.
	now   func() time.Time
	sleep func(ctx context.Context, d time.Duration)
}

// New builds a Base driving handler.
func New(handler Handler, cfg Config) *Base {
	b := &Base{
		handler:          handler,
		reconnectOnError: cfg.ReconnectOnError == nil || *cfg.ReconnectOnError,
		maxRetries:       cfg.MaxRetries,
		tracker:          network.NewQuickFailureTracker(cfg.QuickFailure),
		now:              time.Now,
		sleep:            sleepContext,
	}
	if b.maxRetries <= 0 {
		b.maxRetries = DefaultMaxRetries
	}
	return b
}

// Connect readies the base for a fresh connection: reconnection is allowed
// again, and the streak of instant failures starts over. Call it from the
// service's own connect, before opening the socket.
func (b *Base) Connect() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.disconnecting = false
	b.tracker.Reset()
}

// Disconnect marks the disconnection as deliberate, so the receive loop ending
// is read as the shutdown it is rather than as a connection to restore. Call it
// from the service's own disconnect, before closing the socket.
func (b *Base) Disconnect() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.disconnecting = true
}

// Disconnecting reports whether a deliberate disconnect is under way.
func (b *Base) Disconnecting() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.disconnecting
}

// ReceiveTaskHandler reads from the connection until there is nothing left to
// recover, reconnecting in between. It returns when the peer closed the
// connection normally, when the service is disconnecting, or when reconnecting
// has stopped being worth trying. Run it in a goroutine of its own for as long
// as the service is connected.
func (b *Base) ReceiveTaskHandler(ctx context.Context, report ReportError) {
	for {
		b.setLastConnect(b.now())

		err := b.handler.ReceiveMessages(ctx)

		var message string
		switch {
		case err == nil:
			// Reading ended with no error to report, which is what a peer
			// closing the connection gracefully looks like from here. The
			// connection is still gone, so it is worth restoring.
			message = "connection closed by server"
		case isNormalClosure(err):
			slog.Debug("websocket connection closed normally", "err", err)
			return
		case isCloseError(err):
			message = fmt.Sprintf("connection closed, but with an error: %v", err)
		default:
			message = fmt.Sprintf("error receiving messages: %v", err)
		}

		if !b.maybeTryReconnect(ctx, message, err, report) {
			return
		}
	}
}

// maybeTryReconnect decides whether the connection just lost is worth restoring
// and restores it when it is. It reports whether the receive loop should carry
// on.
func (b *Base) maybeTryReconnect(ctx context.Context, message string, err error, report ReportError) bool {
	if b.Disconnecting() {
		// The service asked for this, so there is nothing to report and nothing
		// to reconnect.
		slog.Debug("websocket receive loop ended during disconnect", "err", err)
		return false
	}

	if last := b.lastConnectTime(); !last.IsZero() {
		lasted := b.now().Sub(last)
		result := b.tracker.Record(lasted)
		if result.QuickFailure {
			slog.Warn("websocket connection failed immediately after connecting",
				"lasted", lasted,
				"consecutive", b.tracker.Count(),
				"limit", b.tracker.MaxConsecutiveFailures())
		}
		if result.GiveUp {
			// Redialing keeps succeeding and the connection keeps dying, so
			// waiting longer between attempts is not the answer.
			giveUp := fmt.Sprintf("connection failed %d times immediately after connecting",
				b.tracker.MaxConsecutiveFailures())
			slog.Error(giveUp)
			report(ctx, giveUp)
			return false
		}
	}

	slog.Warn(message)

	if !b.reconnectOnError {
		report(ctx, message)
		return false
	}
	return b.TryReconnect(ctx, report)
}

// TryReconnect redials until it has a live connection or has run out of
// attempts, waiting longer after each failure. It reports whether it succeeded.
// Every failed attempt is reported, and so is running out of them.
func (b *Base) TryReconnect(ctx context.Context, report ReportError) bool {
	if !b.beginReconnect() {
		slog.Warn("websocket reconnect attempt aborted: already in progress")
		return false
	}
	defer b.endReconnect()

	var lastErr error
	for attempt := 1; attempt <= b.maxRetries; attempt++ {
		ok, err := b.reconnectWebsocket(ctx, attempt)
		switch {
		case err != nil:
			lastErr = err
			slog.Error("websocket reconnection attempt failed", "attempt", attempt, "err", err)
			if report != nil {
				report(ctx, fmt.Sprintf("reconnection attempt %d failed: %v", attempt, err))
			}
		case ok:
			slog.Info("websocket reconnected", "attempt", attempt)
			b.setLastConnect(b.now())
			return true
		}
		b.sleep(ctx, network.ExponentialBackoffTime(attempt,
			network.DefaultMinWait, network.DefaultMaxWait, network.DefaultMultiplier))
	}

	message := fmt.Sprintf("failed to reconnect after %d attempts", b.maxRetries)
	if lastErr != nil {
		message = fmt.Sprintf("%s: %v", message, lastErr)
	}
	slog.Error(message)
	if report != nil {
		report(ctx, message)
	}
	return false
}

// reconnectWebsocket closes what is left of the old connection and opens a new
// one, then checks it is really there.
func (b *Base) reconnectWebsocket(ctx context.Context, attempt int) (bool, error) {
	slog.Warn("reconnecting websocket", "attempt", attempt)
	if err := b.handler.DisconnectWebsocket(ctx); err != nil {
		return false, err
	}
	if err := b.handler.ConnectWebsocket(ctx); err != nil {
		return false, err
	}
	if !b.handler.Connected() {
		return false, ErrVerification
	}
	return true, nil
}

// SendWithRetry sends and, when that fails, reconnects and sends once more. It
// returns the failure that stood when there was no connection left to send on,
// and nil once the message is away.
func (b *Base) SendWithRetry(ctx context.Context, send func() error, report ReportError) error {
	err := b.sendOnce(send)
	if err == nil {
		return nil
	}
	slog.Error("websocket send failed, will try to reconnect", "err", err)

	if !b.TryReconnect(ctx, report) {
		slog.Error("websocket send failed; unable to reconnect", "err", err)
		return err
	}
	slog.Info("websocket reconnected, retrying the send")
	return b.sendOnce(send)
}

// sendOnce sends, treating a missing connection as the send failure it is so the
// caller reconnects rather than silently dropping the message.
func (b *Base) sendOnce(send func() error) error {
	if !b.handler.Connected() {
		return ErrNotConnected
	}
	return send()
}

func (b *Base) beginReconnect() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.reconnectInProgress {
		return false
	}
	b.reconnectInProgress = true
	return true
}

func (b *Base) endReconnect() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.reconnectInProgress = false
}

func (b *Base) setLastConnect(t time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.lastConnect = t
}

func (b *Base) lastConnectTime() time.Time {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.lastConnect
}

// isCloseError reports whether err is the peer closing the connection, as
// opposed to a read failing for some other reason.
func isCloseError(err error) bool {
	return websocket.CloseStatus(err) != -1
}

// isNormalClosure reports whether err is the peer closing the connection the way
// a peer that is done closes it. There is nothing to restore in that case: the
// session ended because it was meant to.
func isNormalClosure(err error) bool {
	switch websocket.CloseStatus(err) {
	case websocket.StatusNormalClosure, websocket.StatusGoingAway:
		return true
	default:
		return false
	}
}

// sleepContext waits for d, or until ctx ends.
func sleepContext(ctx context.Context, d time.Duration) {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}
