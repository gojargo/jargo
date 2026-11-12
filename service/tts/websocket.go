package tts

import (
	"context"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

// receiveTaskStopTimeout bounds the wait for the receive loop to end when the
// service disconnects. A loop that is still handing audio to a context the
// output has stopped draining would otherwise hold the disconnect up.
const receiveTaskStopTimeout = time.Second

// WebsocketHandler is the connection half of a TTS service that speaks over a
// WebSocket held open for the call. The service owns the socket; these are what
// the base needs to open it, close it, read from it and know it is still there.
type WebsocketHandler interface {
	// ConnectWebsocket opens the connection, doing nothing when one is open. It
	// is called for the redial of every reconnect attempt, so it must be safe
	// to call repeatedly.
	ConnectWebsocket(ctx context.Context) error
	// DisconnectWebsocket closes the connection. It is called before every
	// redial, so it must tolerate there being nothing to close.
	DisconnectWebsocket(ctx context.Context) error
	// ReceiveMessages reads from the connection until it ends, handling each
	// message. Returning nil means the peer closed the connection gracefully;
	// returning an error means it failed. Either way the base decides whether
	// to reconnect.
	ReceiveMessages(ctx context.Context) error
	// Connected reports whether there is a live connection.
	Connected() bool
}

// WebsocketService is a Synthesizer that speaks over a WebSocket held open for
// the call, driven by a Websocket.
//
// Connect and Disconnect are the service's whole connect and disconnect. The
// base calls Disconnect when the pipeline ends, is canceled or is torn down,
// and an interruptible service both when an interruption has to reset the
// connection. A service writes them from the base's steps, in this order:
//
//	func (s *synthesizer) Connect(ctx context.Context) {
//		s.ws.Connect()
//		_ = s.ConnectWebsocket(ctx)
//		if s.Connected() {
//			s.ws.StartReceiveTask(ctx)
//		}
//	}
//
//	func (s *synthesizer) Disconnect(ctx context.Context) {
//		s.ws.Disconnect()
//		s.ws.CancelReceiveTask()
//		_ = s.DisconnectWebsocket(ctx)
//	}
//
// adding what its provider needs on top, such as a keepalive.
type WebsocketService interface {
	Synthesizer
	WebsocketHandler
	Connect(ctx context.Context)
	Disconnect(ctx context.Context)
}

// Websocket gives a TTS service the connection lifecycle of one that speaks over
// a WebSocket held open for the call.
//
// The service connects when the pipeline starts, and the base disconnects it
// when the pipeline ends or is canceled, and when the service is torn down. A
// receive loop reads the provider's messages for as long as the service is
// connected. When the connection fails it is restored with an exponential
// backoff, each failed attempt reported without ending the call; running out of
// attempts, or a connection that keeps dying the moment it is made, is reported
// as a permanent error, which makes the service unusable so a service switcher
// moves off it (see service/wsservice). Every error is reported as a connection
// error, which raises the service's on-connection-error event.
//
// A provider keeps its context state per connection, so replacing the
// connection, or closing it, ends every open audio context: the audio already
// received still plays, and the rest of a turn in progress moves to a new
// context ID the next connection opens fresh.
//
// The text is handed off and its audio arrives later on the receive loop, so a
// websocket service reports no processing time: there is no synthesis inside
// the call it would be measured around. Time to first byte and to first audio
// carry the latency.
type Websocket struct {
	tts  *Base
	svc  WebsocketService
	opts wsservice.Options
	ws   *wsservice.Base

	// interruptible says the service resets its connection on an interruption
	// while the bot is speaking. See NewInterruptibleWebsocket.
	interruptible bool

	mu sync.Mutex
	// receiveStop ends the receive loop, and receiveDone is closed once it has
	// ended. Both are nil while no loop runs.
	receiveStop context.CancelFunc
	receiveDone chan struct{}
	// botSpeaking is whether the bot is speaking, as BotStartedSpeakingFrame
	// and BotStoppedSpeakingFrame report it.
	botSpeaking bool
	// ttsStarted is set once a TTSStartedFrame has been pushed for the current
	// turn, before a BotStartedSpeakingFrame confirms playback: the window in
	// which botSpeaking, which only reflects confirmed playback, cannot yet tell
	// that a reconnect is needed. It is kept apart from botSpeaking so this
	// early, unconfirmed marker never makes a turn that produces no audio look
	// like one that played.
	ttsStarted bool

	keepalive Keepalive
}

// NewWebsocket builds the connection lifecycle for the TTS service b, whose
// socket svc owns, configured by opts. It attaches itself to b, which from then
// on disconnects svc when the pipeline ends.
func NewWebsocket(b *Base, svc WebsocketService, opts wsservice.Options) *Websocket {
	w := &Websocket{tts: b, svc: svc, opts: opts}
	w.ws = wsservice.New(wsHandler{w}, opts.Config())
	b.websocket = w
	return w
}

// NewInterruptibleWebsocket builds the connection lifecycle of a websocket TTS
// service whose provider cannot be told to drop the audio of one request: it
// does not tag its audio with the context it belongs to, or offers no way to
// cancel one. Audio still coming for a reply the user cut off could then only
// be told apart from the next reply's by the connection it arrives on, so an
// interruption while the bot is speaking, or about to, resets the connection.
//
// The connection is kept when the only contexts still open are uninterruptible
// speech, which the interruption leaves playing.
func NewInterruptibleWebsocket(b *Base, svc WebsocketService, opts wsservice.Options) *Websocket {
	w := NewWebsocket(b, svc, opts)
	w.interruptible = true
	return w
}

// Options are the websocket options the service was configured with.
func (w *Websocket) Options() wsservice.Options { return w.opts }

// Dial opens a connection with the service's timeouts applied: how long a
// close waits for the peer, and how long a send may wait on one that has
// stopped reading.
func (w *Websocket) Dial(ctx context.Context, url string, header http.Header, readLimit int64) (*wsutil.Conn, error) {
	return w.opts.Dial(ctx, url, header, readLimit)
}

// Connect readies the base for a fresh connection: reconnection is allowed
// again, and the streak of instant failures starts over. A service calls it
// first thing in its own Connect, before opening the socket.
func (w *Websocket) Connect() {
	w.ws.Connect()
}

// Disconnect ends the audio contexts of the connection about to be closed and
// marks the disconnect as deliberate, so the receive loop ending is read as the
// shutdown it is rather than as a connection to restore. A service calls it
// first thing in its own Disconnect, before closing the socket.
func (w *Websocket) Disconnect() {
	w.closeConnectionContexts()
	w.ws.Disconnect()
}

// closeConnectionContexts ends the audio contexts of a connection that is being
// replaced or closed.
//
// A provider keeps context state per connection and does not resume it on a
// new one, so audio still due on the old connection never arrives. Every open
// context is closed (audio already received still plays), and a turn in
// progress moves to a new context ID so its remaining sentences open a fresh
// provider context. Reopening a connection that was already closed, to send
// the text at hand, does not come through here, so the context it sends to
// stays open.
func (w *Websocket) closeConnectionContexts() {
	w.tts.ReplaceConnection()
}

// StartReceiveTask starts the receive loop, unless one is running. The loop
// reads the provider's messages, and when the connection fails it reconnects
// with backoff and goes on reading. It runs until CancelReceiveTask, or until
// reconnecting has stopped being worth trying. ctx is the connection's, so the
// loop is not tied to the call that opened it.
func (w *Websocket) StartReceiveTask(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.receiveStop != nil {
		return
	}
	runCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	w.receiveStop, w.receiveDone = stop, done
	go func() {
		defer close(done)
		w.ws.ReceiveTaskHandler(runCtx, w.ReportError)
		w.mu.Lock()
		if w.receiveDone == done {
			w.receiveStop, w.receiveDone = nil, nil
		}
		w.mu.Unlock()
	}()
}

// CancelReceiveTask stops the receive loop, if one runs, and waits a moment for
// it to end.
func (w *Websocket) CancelReceiveTask() {
	w.mu.Lock()
	stop, done := w.receiveStop, w.receiveDone
	w.receiveStop, w.receiveDone = nil, nil
	w.mu.Unlock()
	if stop == nil {
		return
	}
	stop()
	select {
	case <-done:
	case <-time.After(receiveTaskStopTimeout):
		slog.Debug("tts receive loop did not stop in time", "service", w.tts.Name())
	}
}

// ReceiveTaskRunning reports whether the receive loop is running.
func (w *Websocket) ReceiveTaskRunning() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.receiveStop != nil
}

// CreateKeepaliveTask starts the keepalive loop if it isn't running, for a
// provider that closes an idle connection: every interval it sends h's
// keepalive while the websocket is open. A service calls it from its Connect.
func (w *Websocket) CreateKeepaliveTask(ctx context.Context, interval time.Duration, h KeepaliveHandler) {
	w.keepalive.Start(context.WithoutCancel(ctx), interval, h)
}

// CancelKeepaliveTask stops the keepalive loop, if running. A service calls it
// from its Disconnect.
func (w *Websocket) CancelKeepaliveTask() {
	w.keepalive.Stop()
}

// KeepaliveRunning reports whether the keepalive loop is running.
func (w *Websocket) KeepaliveRunning() bool { return w.keepalive.Running() }

// TryReconnect replaces the connection, retrying with backoff, and reports
// whether there is a live one at the end. The audio contexts of the connection
// replaced are ended first. Every failed attempt is reported, and running out
// of them is reported as permanent.
func (w *Websocket) TryReconnect(ctx context.Context) bool {
	return w.ws.TryReconnect(ctx, w.ReportError)
}

// ReconnectInProgress reports whether the receive loop is redialing. The
// connection is the loop's until it is back.
func (w *Websocket) ReconnectInProgress() bool { return w.ws.ReconnectInProgress() }

// Disconnecting reports whether a deliberate disconnect is under way.
func (w *Websocket) Disconnecting() bool { return w.ws.Disconnecting() }

// ReportError reports a connection error: it raises the service's
// on-connection-error event and pushes the error, as permanent when
// forceTreatAsPermanent is set.
func (w *Websocket) ReportError(ctx context.Context, ef *frames.ErrorFrame, forceTreatAsPermanent bool) {
	w.tts.ReportConnectionError(ctx, ef, forceTreatAsPermanent)
}

// disconnect runs the service's own disconnect, for the end of the pipeline.
func (w *Websocket) disconnect(ctx context.Context) {
	w.svc.Disconnect(ctx)
}

// trackFrame follows the frames an interruptible service needs to tell whether
// an interruption has to reset the connection. It runs once the base has
// processed the frame.
func (w *Websocket) trackFrame(f frames.Frame) {
	if !w.interruptible {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	switch f.(type) {
	case *frames.BotStartedSpeakingFrame:
		w.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		// The turn ended normally, so there is nothing left to reconnect for.
		w.botSpeaking = false
		w.ttsStarted = false
	case *frames.LLMFullResponseStartFrame:
		// A safety net for a previous turn that never produced audio, which
		// BotStoppedSpeakingFrame then never came to clear.
		w.ttsStarted = false
	}
}

// trackPush notes a TTSStartedFrame being pushed, which closes the window
// between the synthesis starting and the BotStartedSpeakingFrame confirming
// playback, in which stale audio could otherwise leak through an interruption.
func (w *Websocket) trackPush(f frames.Frame) {
	if !w.interruptible {
		return
	}
	if _, ok := f.(*frames.TTSStartedFrame); !ok {
		return
	}
	w.mu.Lock()
	w.ttsStarted = true
	w.mu.Unlock()
}

// beginInterruption reports whether the interruption about to be handled has to
// reset the connection, and clears what made it so. It is read before the base
// handles the interruption, which drops the contexts it is decided on.
//
// The connection is reset when the bot was speaking, or about to. Without
// context IDs, audio still coming for an interrupted context would be labeled
// with an uninterruptible one, so the connection is kept only when
// uninterruptible speech is all that is open.
func (w *Websocket) beginInterruption() bool {
	if !w.interruptible {
		return false
	}
	keepConnection := w.tts.onlyUninterruptibleAudioContextsOpen()
	w.mu.Lock()
	shouldReconnect := w.botSpeaking || w.ttsStarted
	w.botSpeaking = false
	w.ttsStarted = false
	w.mu.Unlock()
	return shouldReconnect && !keepConnection
}

// resetConnection disconnects and connects again, for an interruption that cut
// off speech the provider is still generating.
func (w *Websocket) resetConnection(ctx context.Context) {
	w.svc.Disconnect(ctx)
	w.svc.Connect(ctx)
}

// wsHandler adapts the service to the wsservice base. A redial ends the audio
// contexts of the connection it replaces before closing it.
type wsHandler struct{ w *Websocket }

func (h wsHandler) ConnectWebsocket(ctx context.Context) error {
	return h.w.svc.ConnectWebsocket(ctx)
}

func (h wsHandler) DisconnectWebsocket(ctx context.Context) error {
	h.w.closeConnectionContexts()
	return h.w.svc.DisconnectWebsocket(ctx)
}

func (h wsHandler) ReceiveMessages(ctx context.Context) error {
	return h.w.svc.ReceiveMessages(ctx)
}

func (h wsHandler) Connected() bool { return h.w.svc.Connected() }

func (h wsHandler) Usable() bool { return h.w.tts.Usable() }
