package tts

import (
	"context"
	"log/slog"
	"sync"
	"time"
)

// KeepaliveHandler is the provider's half of a Keepalive.
type KeepaliveHandler interface {
	// KeepaliveReady reports whether a keepalive can be sent now, which is
	// whether the websocket is open.
	KeepaliveReady() bool
	// SendKeepalive sends the provider's keepalive message over the websocket.
	SendKeepalive(ctx context.Context) error
}

// Keepalive holds a websocket synthesizer's connection open for a provider that
// closes an idle one. A synthesizer keeps one, starts it from its connect and
// stops it from its disconnect; the zero value is ready to use.
//
// Every interval it sends the provider's keepalive while the websocket is
// open. A failed send is logged and the loop keeps going: the receive loop
// handles a lost connection, and the next keepalive goes out on the
// reconnected socket.
type Keepalive struct {
	mu   sync.Mutex
	stop context.CancelFunc
	done chan struct{}
}

// Start starts the keepalive loop if it isn't running, sending through h every
// interval. A loop that has ended, its context done, is replaced. The loop
// runs until Stop, or until ctx is done.
func (k *Keepalive) Start(ctx context.Context, interval time.Duration, h KeepaliveHandler) {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.done != nil {
		select {
		case <-k.done:
		default:
			return
		}
	}
	runCtx, stop := context.WithCancel(ctx)
	done := make(chan struct{})
	k.stop, k.done = stop, done
	go func() {
		defer close(done)
		runKeepalive(runCtx, interval, h)
	}()
}

// Stop stops the keepalive loop if running. It does not wait for the loop to
// end, so it may be called while holding a lock the handler takes.
func (k *Keepalive) Stop() {
	k.mu.Lock()
	stop := k.stop
	k.stop, k.done = nil, nil
	k.mu.Unlock()
	if stop != nil {
		stop()
	}
}

// Running reports whether the keepalive loop is running.
func (k *Keepalive) Running() bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.done == nil {
		return false
	}
	select {
	case <-k.done:
		return false
	default:
		return true
	}
}

// runKeepalive sends a keepalive every interval while the websocket is open,
// until ctx is done.
func runKeepalive(ctx context.Context, interval time.Duration, h KeepaliveHandler) {
	tick := time.NewTicker(interval)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		if !h.KeepaliveReady() {
			continue
		}
		if err := h.SendKeepalive(ctx); err != nil && ctx.Err() == nil {
			slog.Warn("tts keepalive error", "err", err)
		}
	}
}
