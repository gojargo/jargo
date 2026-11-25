package bland

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	ttstext "github.com/gojargo/jargo/text"
)

const (
	// defaultURL is Bland's realtime TTS endpoint.
	defaultURL = "wss://api.bland.ai/v2/tts/ws"
	// readyTimeout bounds the wait for the session to be admitted.
	readyTimeout = 10 * time.Second
	// closeTimeout bounds the wait for the session to settle on a disconnect.
	closeTimeout = 5 * time.Second
)

// Message types of the Bland protocol.
const (
	msgInit          = "init"
	msgReady         = "ready"
	msgSpeak         = "speak"
	msgEndOfTurn     = "end_of_turn"
	msgCancel        = "cancel"
	msgClose         = "close"
	msgDone          = "done"
	msgError         = "error"
	msgUtteranceEnd  = "utterance_end"
	msgUtteranceInit = "utterance_start"
	// keyType names a message's type, and keyContextID the turn it is about.
	keyType      = "type"
	keyContextID = "context_id"
	// codeIdleTimeout is the error Bland reaps an idle session with.
	codeIdleTimeout = "idle_timeout"
)

// sessionFields are the settings the init message fixes for the life of a
// session. Nothing else in the settings reaches Bland, so nothing else earns a
// reconnect.
//
//nolint:gochecknoglobals // the set of fields, fixed
var sessionFields = []string{keyVoice, "expressiveness", "stability", "auto_formatting"}

// errNotConnected is returned when there is no connection to use.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("websocket not connected")

// errSession is returned for a session Bland refused or acknowledged in a way
// the service cannot use.
//
//nolint:gochecknoglobals // sentinel error
var errSession = errors.New("bland session")

// Config configures the Bland realtime TTS service.
type Config struct {
	// APIKey is the Bland API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the WebSocket endpoint; empty uses the hosted one.
	URL string
	// SampleRate is the PCM rate emitted downstream; 0 uses 48 kHz. A rate Bland
	// does not render (it renders 8000, 16000, 24000, 44100 and 48000) is
	// synthesized at 48 kHz and resampled to this one.
	SampleRate int
	// Voice is the voice id, which also sets the model; empty uses Bland's
	// default voice.
	Voice string
	// Expressiveness is from 0.0 to 1.0: higher values produce more varied
	// intonation. Nil keeps Bland's default.
	Expressiveness *float64
	// Stability is from 0.0 to 1.0: higher values produce more consistent
	// delivery. Nil keeps Bland's default.
	Stability *float64
	// AutoFormatting rewrites numbers into the form the voice reads best before
	// synthesis. Nil leaves it off.
	AutoFormatting *bool
	// TextAggregation is how streamed text is grouped before it is sent. Empty
	// or frames.TextTypeToken streams the model's tokens straight to Bland for
	// the lowest latency: Bland buffers them and chooses its own synthesis
	// boundaries. frames.TextTypeSentence sends a sentence at a time instead.
	TextAggregation frames.TextType
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds a Bland realtime TTS service.
//
// It streams speech from Bland's /v2/tts/ws endpoint over a single connection
// for the whole conversation, opened when the pipeline starts. The text of a
// turn is sent on that turn's context as it arrives, and ended once the turn's
// text has all been sent; the audio arrives on the receive loop. An
// interruption cancels the turn on the server rather than tearing down the
// connection.
//
// When the connection drops it is restored with backoff, and when that keeps
// failing the service reports a permanent error and becomes unusable. See
// tts.Websocket.
func NewTTS(cfg Config) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns its synthesizer alongside it.
func newTTS(cfg Config) (*tts.Base, *synthesizer) {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	s := &synthesizer{
		cfg:          cfg,
		blandRate:    resolveSampleRate(cfg.SampleRate),
		live:         newTTSSettings(cfg.Voice, cfg.Expressiveness, cfg.Stability, cfg.AutoFormatting),
		readyTimeout: readyTimeout,
	}
	b := tts.New("BlandTTS", s)
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	if cfg.TextAggregation != frames.TextTypeSentence {
		b.SetTextAggregator(ttstext.NewTokenAggregator())
	}
	s.svc = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	return b, s
}

// ttsService is the part of the TTS service the synthesizer asks about the
// audio contexts, and reports through. tts.Base implements it.
type ttsService interface {
	Name() string
	ActiveAudioContextID() string
	StopAllMetrics(ctx context.Context)
}

// synthesizer speaks over one session held open for the call. Bland's audio
// names no turn, so it is labeled with the context playing, while the messages
// that end a turn name it.
type synthesizer struct {
	cfg Config
	// blandRate is the rate Bland synthesizes at, resolved from the service's.
	blandRate int
	host      tts.AudioContextHost
	svc       ttsService
	ws        *tts.Websocket

	// live is the settings store, which the base updates on the frame
	// goroutine. The init message built from it is read under mu, since a
	// redial sends it from the receive loop.
	live *TTSSettings
	// readyTimeout bounds the wait for the session to be admitted.
	readyTimeout time.Duration

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a turn may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	mu   sync.Mutex
	conn *wsConn
}

// wsConn is one connection, with the goroutine reading it.
//
// The reading runs apart from the receive loop so stopping the loop does not
// close the connection: the session still has to be settled on it.
type wsConn struct {
	conn *wsutil.Conn
	// reads carries each message read, and finally the error that ended the
	// reading.
	reads chan wsRead
	// stop is closed when the connection is being closed, which ends the reading
	// goroutine once nothing takes its messages any more.
	stop     chan struct{}
	stopOnce sync.Once
	// readEnded is set once the reading has failed, which is the connection
	// closing as far as the service is concerned.
	readEnded atomic.Bool
}

// wsRead is one message read from the connection, or the error that ended the
// reading.
type wsRead struct {
	typ  websocket.MessageType
	data []byte
	err  error
}

func newWSConn(conn *wsutil.Conn) *wsConn {
	c := &wsConn{conn: conn, reads: make(chan wsRead), stop: make(chan struct{})}
	go c.read()
	return c
}

// read reads the connection until it ends, handing each message to whoever
// receives from reads.
func (c *wsConn) read() {
	for {
		typ, data, err := c.conn.Read(context.Background())
		if err != nil {
			c.readEnded.Store(true)
		}
		select {
		case c.reads <- wsRead{typ: typ, data: data, err: err}:
		case <-c.stop:
			return
		}
		if err != nil {
			return
		}
	}
}

// halt stops handing messages on, for a connection about to be closed.
func (c *wsConn) halt() { c.stopOnce.Do(func() { close(c.stop) }) }

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *synthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Bland voice synthesis is billed against. The voice sets
// the model, so there is no model of its own.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{VoiceID: s.live.Voice.Or("")}
}

// RequiresTrailingSpace reports that each sentence is sent ending in a space,
// implementing tts.TrailingSpaceRequirer. Bland appends each speak text
// verbatim, so consecutive sentences would otherwise glue together. It applies
// when sentences are sent; streamed tokens keep the model's own whitespace.
func (s *synthesizer) RequiresTrailingSpace() bool { return true }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. The init
// message fixes the voice, the controls and the formatting for the life of a
// session, so a change to any of them opens a new one.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if slices.ContainsFunc(sessionFields, changed.Has) {
		s.Disconnect(ctx)
		s.Connect(ctx)
	}
	return nil
}

// initMessage builds the message that opens a session.
func (s *synthesizer) initMessage() map[string]any {
	init := map[string]any{
		keyType:  msgInit,
		keyVoice: s.live.Voice.AsAny(),
		"audio":  map[string]any{"encoding": encoding, "sample_rate": s.blandRate},
	}
	applyRequestOptions(init, s.live)
	return init
}

// Start opens the session when the pipeline starts, implementing tts.Starter.
func (s *synthesizer) Start(ctx context.Context) { s.Connect(ctx) }

// Connect opens the session, then starts the receive loop, implementing
// tts.WebsocketService.
func (s *synthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect stops the receive loop and settles and closes the session,
// implementing tts.WebsocketService.
func (s *synthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens the socket and holds the session at ready,
// implementing tts.WebsocketHandler. Bland's ready confirms wallet and
// concurrency admission, so a refused session fails here rather than on the
// first turn. A failure is reported here and leaves no connection behind,
// which a redial reads as a failed attempt; one the caller canceled is not
// reported.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to bland")
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	conn, err := s.ws.Dial(ctx, s.cfg.URL, header, wsutil.DefaultReadLimit)
	if err == nil {
		var sessionID any
		if sessionID, err = s.openSession(ctx, conn); err != nil {
			_ = conn.CloseNow()
		} else {
			slog.Debug("bland session ready", "service", s.svc.Name(), "session_id", sessionID)
		}
	}
	if err != nil {
		if ctx.Err() != nil {
			return nil
		}
		ef := frames.NewErrorFrame(fmt.Sprintf("%s error: %v", s.svc.Name(), err))
		ef.Err = err
		s.ws.ReportError(ctx, ef, false)
		return nil
	}
	s.mu.Lock()
	old := s.conn
	s.conn = newWSConn(conn)
	s.mu.Unlock()
	if old != nil {
		// A connection whose reading had already ended, replaced rather than
		// settled: nothing is left to say to the server on it.
		old.halt()
		_ = old.conn.CloseNow()
	}
	return nil
}

// openSession sends the init message and waits for the session to be ready,
// checking it carries the audio format asked for: audio is never labeled with
// a format the server did not acknowledge. It returns the session's id.
func (s *synthesizer) openSession(ctx context.Context, conn *wsutil.Conn) (any, error) {
	if err := writeJSON(ctx, conn, s.initMessage()); err != nil {
		return nil, err
	}
	readCtx, cancel := context.WithTimeout(ctx, s.readyTimeout)
	defer cancel()
	_, data, err := conn.Read(readCtx)
	if err != nil {
		return nil, err
	}
	var msg map[string]any
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, err
	}
	if msg[keyType] != msgReady {
		message, ok := msg["message"]
		if !ok {
			message = msg
		}
		return nil, fmt.Errorf("%w: Bland rejected the session: %v: %v", errSession, msg["code"], message)
	}
	if rate, _ := msg["sample_rate"].(float64); msg["encoding"] != encoding || rate != float64(s.blandRate) {
		return nil, fmt.Errorf("%w: Bland acknowledged an unexpected audio format: %v at %v Hz",
			errSession, msg["encoding"], msg["sample_rate"])
	}
	return msg["session_id"], nil
}

// DisconnectWebsocket settles and closes the session, implementing
// tts.WebsocketHandler. What the service was timing is reported as far as it
// got first: the audio it was waiting for will not come on this connection.
func (s *synthesizer) DisconnectWebsocket(ctx context.Context) error {
	if s.svc != nil {
		s.svc.StopAllMetrics(ctx)
	}
	s.closeSocket(ctx)
	return nil
}

// closeSocket settles and closes the socket. It is apart from
// DisconnectWebsocket so a turn can replace a socket Bland closed while idle
// without stopping the metrics of the turn it is about to send.
//
// Only a live socket can be settled. Bland reaps an idle session after 60s and
// closes it itself, and teardown runs after that: sending close down a dead
// socket fails, and reporting that as a pipeline error would turn routine
// housekeeping into an error the application has to explain. Bland sends done
// only after it settles the session's usage, so it is waited for before the
// socket is closed. A settle that does not complete costs nothing, since the
// server bills on disconnect regardless: it is logged, not reported.
func (s *synthesizer) closeSocket(ctx context.Context) {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	s.mu.Lock()
	c := s.conn
	s.conn = nil
	s.mu.Unlock()
	if c == nil {
		return
	}
	if !c.readEnded.Load() {
		slog.Debug("disconnecting from bland")
		if err := s.settle(ctx, c); err != nil {
			if closedConnection(err) || errors.Is(err, context.DeadlineExceeded) {
				slog.Debug("bland close handshake did not complete", "service", s.svc.Name(), "err", err)
			} else if s.host != nil {
				s.host.PushError(ctx, fmt.Sprintf("%s error: %v", s.svc.Name(), err), err, false)
			}
		}
	}
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil && !closedConnection(err) {
		slog.Debug("failed to close the bland websocket", "service", s.svc.Name(), "err", err)
	}
}

// settle sends close and reads the session's last messages until done. The
// receive loop has already stopped, so they are read here.
func (s *synthesizer) settle(ctx context.Context, c *wsConn) error {
	if err := writeJSON(ctx, c.conn, map[string]any{keyType: msgClose}); err != nil {
		return err
	}
	timeout := time.NewTimer(closeTimeout)
	defer timeout.Stop()
	for {
		select {
		case <-timeout.C:
			return context.DeadlineExceeded
		case r := <-c.reads:
			if r.err != nil {
				return r.err
			}
			if r.typ != websocket.MessageText {
				continue
			}
			var msg map[string]any
			if err := json.Unmarshal(r.data, &msg); err != nil {
				return err
			}
			if msg[keyType] == msgDone {
				return nil
			}
		}
	}
}

// closedConnection reports whether err is the connection being closed already.
func closedConnection(err error) bool {
	return websocket.CloseStatus(err) != -1 || errors.Is(err, net.ErrClosed) || errors.Is(err, errNotConnected)
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *synthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.conn.readEnded.Load()
}

// ReceiveMessages handles the session's messages until it ends, implementing
// tts.WebsocketHandler.
func (s *synthesizer) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errNotConnected
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case r := <-c.reads:
			if r.err != nil {
				return r.err
			}
			s.handleMessage(ctx, r.typ, r.data)
		}
	}
}

// handleMessage handles one server message. Audio arrives as binary messages
// and is labeled with the context playing. A turn is closed once Bland ends
// it, however it ended, and once Bland reports an error on it: a refused turn
// never starts, so no utterance_end arrives to close it.
func (s *synthesizer) handleMessage(ctx context.Context, typ websocket.MessageType, data []byte) {
	if s.host == nil {
		return
	}
	if typ == websocket.MessageBinary {
		contextID := s.svc.ActiveAudioContextID()
		f := frames.NewTTSAudioRawFrame(data, s.blandRate, 1)
		f.ContextID = contextID
		s.host.AppendToAudioContext(contextID, f)
		return
	}
	var msg map[string]any
	if err := json.Unmarshal(data, &msg); err != nil {
		slog.Error("bland tts: invalid JSON message", "message", string(data))
		return
	}
	contextID, _ := msg[keyContextID].(string)
	switch msg[keyType] {
	case msgUtteranceInit:
		slog.Debug("bland tts: turn started", "context", contextID)
	case msgUtteranceEnd:
		reason := msg["reason"]
		if reason == "failed" {
			// The failure's detail is reported from the error message that
			// precedes it.
			slog.Warn("bland tts: turn failed", "service", s.svc.Name(), "context", contextID)
		} else {
			slog.Debug("bland tts: turn ended", "context", contextID, "reason", reason)
		}
		s.closeTurn(contextID)
	case msgError:
		code := msg["code"]
		if code == codeIdleTimeout {
			// Bland reaps a session after 60s without a client message, which
			// any conversational pause reaches. Reconnecting is routine
			// housekeeping, not something the application can act on.
			slog.Debug("bland tts: session reaped after idle timeout", "service", s.svc.Name())
		} else {
			message, ok := msg["message"]
			if !ok {
				message = msg
			}
			s.host.PushError(ctx, fmt.Sprintf("%s error %v: %v", s.svc.Name(), code, message), nil, false)
		}
		s.closeTurn(contextID)
	case msgDone:
		slog.Debug("bland tts: session settled", "session_id", msg["session_id"])
	default:
		slog.Debug("bland tts: received an unknown message type", "message", msg)
	}
}

// closeTurn stops and closes a turn's audio context, if it is still open.
func (s *synthesizer) closeTurn(contextID string) {
	if contextID == "" || !s.host.AudioContextAvailable(contextID) {
		return
	}
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	s.host.AppendToAudioContext(contextID, stopped)
	s.host.RemoveAudioContext(contextID)
}

// OnAudioContextInterrupted cancels the interrupted turn on the server instead
// of dropping the connection, implementing tts.AudioContextInterrupter.
func (s *synthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	ctx = context.WithoutCancel(ctx)
	s.svc.StopAllMetrics(ctx)
	if contextID == "" {
		return
	}
	if err := s.send(ctx, map[string]any{keyType: msgCancel, keyContextID: contextID}); err != nil &&
		!errors.Is(err, errNotConnected) {
		slog.Error("bland tts: error sending the cancel message", "service", s.svc.Name(), "err", err)
	}
}

// FlushAudio ends the turn so Bland synthesizes the text it still buffers for
// it, implementing tts.AudioFlusher. It falls back to the active context when
// none is named.
func (s *synthesizer) FlushAudio(ctx context.Context, contextID string) {
	if contextID == "" {
		contextID = s.svc.ActiveAudioContextID()
	}
	if contextID == "" {
		return
	}
	if err := s.send(ctx, map[string]any{keyType: msgEndOfTurn, keyContextID: contextID}); err != nil &&
		!errors.Is(err, errNotConnected) {
		slog.Error("bland tts: error sending the end_of_turn message", "service", s.svc.Name(), "err", err)
	}
}

// RunTTS appends a text delta to the turn and returns: the audio arrives on the
// receive loop. Bland ends a session after 60s without a client message, which
// a conversational gap reaches easily, so a session that has closed is
// replaced first. A send that fails is yielded as an error in the context's
// stream.
func (s *synthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		// The receive loop has ended, or is about to, with the connection; it is
		// stopped so connecting starts a fresh one.
		s.ws.CancelReceiveTask()
		s.closeSocket(ctx)
		s.Connect(ctx)
	}
	err := s.send(ctx, map[string]any{keyType: msgSpeak, keyContextID: contextID, "text": text})
	if err != nil {
		if yield != nil {
			ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
			ef.Err = err
			return yield(ef)
		}
		return err
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// send writes one JSON message on the open connection.
func (s *synthesizer) send(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errNotConnected
	}
	return writeJSON(ctx, c.conn, msg)
}

func writeJSON(ctx context.Context, conn *wsutil.Conn, msg map[string]any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, payload)
}

// Compile-time checks that the synthesizer is driven the way the base drives
// it.
var (
	_ tts.ContextSynthesizer      = (*synthesizer)(nil)
	_ tts.WebsocketService        = (*synthesizer)(nil)
	_ tts.Starter                 = (*synthesizer)(nil)
	_ tts.AudioFlusher            = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*synthesizer)(nil)
	_ tts.SettingsUpdater         = (*synthesizer)(nil)
	_ tts.Describer               = (*synthesizer)(nil)
	_ tts.TrailingSpaceRequirer   = (*synthesizer)(nil)
)
