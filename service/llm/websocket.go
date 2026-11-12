package llm

import (
	"context"
	"errors"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/wsservice"
)

// ErrWebsocketNotConnected is returned by Websocket.EnsureConnected when no
// connection could be established.
//
//nolint:gochecknoglobals // sentinel error
var ErrWebsocketNotConnected = errors.New("llm: failed to establish the websocket connection")

// errInlineReceive is what the receive loop would get, were it ever started: a
// websocket LLM service reads each response inline, during its inference.
//
//nolint:gochecknoglobals // sentinel error
var errInlineReceive = errors.New("llm: a websocket LLM service receives inline, not in a loop")

// WebsocketHandler is the connection half of an LLM service that runs each
// inference over a WebSocket held open for the call. The service owns the
// socket; these are what Websocket needs to open it, close it, and know it is
// still there.
type WebsocketHandler interface {
	// ConnectWebsocket opens the connection, doing nothing when one is open.
	// It is called for every reconnect attempt.
	ConnectWebsocket(ctx context.Context) error
	// DisconnectWebsocket closes the connection and forgets what it held. It
	// is called before every redial, so it must tolerate there being nothing
	// to close.
	DisconnectWebsocket(ctx context.Context) error
	// Connected reports whether there is a live connection.
	Connected() bool
}

// Websocket gives an LLM service the connection lifecycle of one that runs
// each inference over a WebSocket held open for the call.
//
// Each inference is a discrete exchange: send one request, read its events
// until the last, then wait for the next. So unlike a streaming STT or TTS
// service there is no background receive loop. What Websocket adds is the
// connection around those exchanges: it opens it when the service is set up
// and closes it when the pipeline ends, makes sure it is there before each
// inference, and restores it when a send or a read finds it gone. Restoring
// retries with an exponential backoff and reports each failed attempt, and
// running out of attempts reports a permanent error, which makes the service
// unusable (see service/wsservice).
type Websocket struct {
	llm     *Base
	handler WebsocketHandler
	ws      *wsservice.Base
}

// NewWebsocket builds the connection lifecycle for the service b, whose socket
// handler owns.
func NewWebsocket(b *Base, handler WebsocketHandler, cfg wsservice.Config) *Websocket {
	w := &Websocket{llm: b, handler: handler}
	w.ws = wsservice.New(wsHandler{w}, cfg)
	return w
}

// Connect readies the connection and opens it. A connection that cannot be
// opened is reported, and the next inference tries again.
func (w *Websocket) Connect(ctx context.Context) {
	w.ws.Connect()
	if err := w.handler.ConnectWebsocket(ctx); err != nil {
		w.llm.PushError(ctx, "error connecting to websocket", err, false)
	}
}

// Disconnect marks the disconnect deliberate, so nothing restores it, and
// closes the connection.
func (w *Websocket) Disconnect(ctx context.Context) {
	w.ws.Disconnect()
	if err := w.handler.DisconnectWebsocket(ctx); err != nil {
		w.llm.PushError(ctx, "error disconnecting from websocket", err, false)
	}
}

// EnsureConnected makes sure there is a live connection before an inference,
// reconnecting with backoff when there is none. It returns
// ErrWebsocketNotConnected when the connection could not be restored.
func (w *Websocket) EnsureConnected(ctx context.Context) error {
	if w.handler.Connected() {
		return nil
	}
	if w.ws.TryReconnect(ctx, w.reportError) {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return ErrWebsocketNotConnected
}

// Reconnect restores a connection a send or a read found gone, reporting
// whether it is back. The inference that found it gone has to start over: a
// fresh connection holds nothing of the old one's state. A connection closed
// on purpose is not restored.
func (w *Websocket) Reconnect(ctx context.Context) bool {
	if w.ws.Disconnecting() {
		return false
	}
	return w.ws.TryReconnect(ctx, w.reportError)
}

// Disconnecting reports whether a deliberate disconnect is under way.
func (w *Websocket) Disconnecting() bool { return w.ws.Disconnecting() }

// reportError puts a connection error on the pipeline.
func (w *Websocket) reportError(ctx context.Context, ef *frames.ErrorFrame, forceTreatAsPermanent bool) {
	w.llm.PushErrorFrame(ctx, ef, forceTreatAsPermanent)
}

// wsHandler adapts the service's handler to the wsservice base, which also
// asks for a receive loop this kind of service does not run, and for whether
// the service is still usable.
type wsHandler struct{ w *Websocket }

func (h wsHandler) ConnectWebsocket(ctx context.Context) error {
	return h.w.handler.ConnectWebsocket(ctx)
}

func (h wsHandler) DisconnectWebsocket(ctx context.Context) error {
	return h.w.handler.DisconnectWebsocket(ctx)
}

func (h wsHandler) Connected() bool { return h.w.handler.Connected() }

func (wsHandler) ReceiveMessages(context.Context) error { return errInlineReceive }

func (h wsHandler) Usable() bool { return h.w.llm.Usable() }
