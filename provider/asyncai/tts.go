package asyncai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

// Message keys of the Async protocol.
const (
	keyTranscript = "transcript"
	keyContextID  = "context_id"
)

// keepaliveInterval is how often an idle connection is sent a blank
// transcript, so the server does not close it between turns.
const keepaliveInterval = 10 * time.Second

// errNotConnected is returned when there is no connection to use.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("asyncai tts: websocket not connected")

// sessionInitFields are the settings sent in the init message when the
// connection opens, and never again: only a new session applies a change to
// them.
//
//nolint:gochecknoglobals // the set of fields, fixed
var sessionInitFields = []string{"model", "voice", "language"}

// NewTTS builds an Async TTS service.
//
// It holds one connection open for the call, opened when the pipeline starts.
// When it drops it is restored with backoff, and when that keeps failing the
// service reports a permanent error and becomes unusable. See tts.Websocket.
func NewTTS(cfg Config) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns its synthesizer alongside it.
func newTTS(cfg Config) (*tts.Base, *synthesizer) {
	s := newSynthesizer(cfg)
	b := tts.New("AsyncAITTS", s)
	// The audio of a request is labeled by the context it names, but the next
	// request's frames are still held until this one's audio has played.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	s.svc = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	return b, s
}

// newSynthesizer builds the synthesizer, with the defaults applied and its
// settings taken from cfg.
func newSynthesizer(cfg Config) *synthesizer {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	s := &synthesizer{cfg: cfg, live: &settings.TTS{}}
	s.live.Model = settings.Set(cfg.Model)
	s.live.Voice = settings.Set(cfg.Voice)
	s.live.Language = settings.Set(cfg.Language)
	s.refreshInit()
	return s
}

// ttsService is the part of the TTS service the synthesizer asks about the audio
// contexts, and reports through. tts.Base implements it.
type ttsService interface {
	ActiveAudioContextID() string
	HasActiveAudioContext() bool
	StopAllMetrics(ctx context.Context)
	WarnUnhandledUpdatedSettings(ctx context.Context, unhandled []string)
}

// synthesizer speaks over one connection held open for the call. A sentence is
// sent on the context it belongs to and the call returns; the audio arrives on
// the receive loop, labeled with that context.
type synthesizer struct {
	cfg  Config
	host tts.AudioContextHost
	svc  ttsService
	ws   *tts.Websocket

	// live is the settings store, which the base updates on the frame
	// goroutine. It is read only there; the init message built from it is
	// copied into init under mu, since a redial sends it from the receive loop.
	live *settings.TTS

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	mu   sync.Mutex
	conn *wsConn
	// init is the message that opens a session.
	init map[string]any
}

// wsConn is one connection, with the goroutine reading it.
//
// The reading runs apart from the receive loop so stopping the loop does not
// close the connection: the service still has to tell the server it is done
// with it.
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
		_, data, err := c.conn.Read(context.Background())
		if err != nil {
			c.readEnded.Store(true)
		}
		select {
		case c.reads <- wsRead{data: data, err: err}:
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

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Async model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.Voice}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Async does, implementing
// tts.LanguageNamer.
func (s *synthesizer) ServiceLanguage(l language.Language) string { return asyncLanguage(l) }

// asyncLanguage maps a language to the code Async takes: its base code. A
// language Async is not known to speak still gets its base code, with a
// warning.
func asyncLanguage(l language.Language) string {
	base := l.BaseCode()
	switch base {
	case "en", "fr", "es", "de", "it", "pt", "nl", "ar", "ru", "ro", "ja", "he", "hy", "tr", "hi", "zh":
	default:
		slog.Warn("asyncai tts: language not verified, using its base code", "language", string(l), "code", base)
	}
	return base
}

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. The model,
// the voice and the language are sent once, in the init message that opens a
// session, so a change to any of them starts a new session. Any other field
// cannot reach a live session and is reported as not taking effect.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.refreshInit()
	if slices.ContainsFunc(sessionInitFields, changed.Has) {
		s.Disconnect(ctx)
		s.Connect(ctx)
	}
	s.svc.WarnUnhandledUpdatedSettings(ctx, changed.Except(sessionInitFields...))
	return nil
}

// refreshInit builds the init message from the settings store. It runs on the
// frame goroutine, which is where the store is written.
func (s *synthesizer) refreshInit() {
	init := map[string]any{
		"model_id": s.live.Model.AsAny(),
		"voice":    map[string]any{"mode": "id", "id": s.live.Voice.AsAny()},
		"output_format": map[string]any{
			"container":   defaultContainer,
			"encoding":    defaultEncoding,
			"sample_rate": s.cfg.SampleRate,
		},
		"language": s.live.Language.AsAny(),
	}
	s.mu.Lock()
	s.init = init
	s.mu.Unlock()
}

// buildMsg is a message carrying text on a context.
func buildMsg(text, contextID string, force bool) map[string]any {
	return map[string]any{keyTranscript: text, keyContextID: contextID, "force": force}
}

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *synthesizer) Start(ctx context.Context) {
	s.refreshInit()
	s.Connect(ctx)
}

// Connect opens the connection, then starts the receive loop and the keepalive,
// implementing tts.WebsocketService.
func (s *synthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
		s.ws.CreateKeepaliveTask(ctx, keepaliveInterval, s)
	}
}

// Disconnect stops the receive loop and the keepalive and closes the
// connection, implementing tts.WebsocketService.
func (s *synthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	s.ws.CancelKeepaliveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens a session unless one is open, implementing
// tts.WebsocketHandler: it dials, then sends the init message that fixes the
// model, the voice, the output format and the language. A failure is reported
// here and leaves no connection behind, which a redial reads as a failed
// attempt.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to asyncai tts")
	endpoint := s.cfg.URL + "?api_key=" + url.QueryEscape(s.cfg.APIKey) + "&version=" + url.QueryEscape(s.cfg.Version)
	conn, err := s.ws.Dial(ctx, endpoint, http.Header{}, wsutil.DefaultReadLimit)
	if err == nil {
		s.mu.Lock()
		init := s.init
		s.mu.Unlock()
		if err = writeJSON(ctx, conn, init); err != nil {
			_ = conn.CloseNow()
		}
	}
	if err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
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
		// closed: nothing is left to say to the server on it.
		old.halt()
		_ = old.conn.CloseNow()
	}
	return nil
}

// DisconnectWebsocket closes the connection, implementing
// tts.WebsocketHandler. With a context still playing, the server is first told
// to terminate every context on it.
func (s *synthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.svc != nil {
		s.svc.StopAllMetrics(ctx)
	}
	s.mu.Lock()
	c := s.conn
	s.conn = nil
	s.mu.Unlock()
	if c == nil {
		return nil
	}
	slog.Debug("disconnecting from asyncai tts")
	c.halt()
	var err error
	if s.svc != nil && s.svc.HasActiveAudioContext() {
		err = writeJSON(ctx, c.conn, map[string]any{"terminate": true})
	}
	if closeErr := c.conn.Close(websocket.StatusNormalClosure, ""); err == nil {
		err = closeErr
	}
	if err != nil && s.host != nil {
		s.host.PushError(ctx, "Unknown error occurred", err, false)
	}
	slog.Debug("disconnected from asyncai tts")
	return nil
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *synthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.conn.readEnded.Load()
}

// ReceiveMessages handles the connection's messages until it ends, implementing
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
			if err := s.handleMessage(r.data); err != nil {
				return err
			}
		}
	}
}

// wsMessage is the subset of an Async WebSocket message read.
type wsMessage struct {
	ContextID string `json:"context_id"`
	Final     bool   `json:"final"`
	Audio     string `json:"audio"`
}

// handleMessage handles one server message.
//
// The final message arrives after the context has been closed, so it carries
// nothing to act on. Audio for a context that is no longer open is dropped,
// unless it is the active one, which the base opens again: audio the server
// was still generating when the user cut in arrives after the interruption
// closed its context.
func (s *synthesizer) handleMessage(data []byte) error {
	var m *wsMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("asyncai tts: decode message: %w", err)
	}
	if m == nil {
		return nil
	}
	id := m.ContextID
	if m.Final {
		slog.Debug("asyncai tts: final message", "context", id)
		return nil
	}
	if s.host == nil {
		return nil
	}
	if !s.host.AudioContextAvailable(id) {
		if s.svc.ActiveAudioContextID() != id {
			slog.Debug("asyncai tts: ignoring a message from an unavailable context", "context", id)
			return nil
		}
		slog.Debug("asyncai tts: received a delayed message, recreating the context", "context", id)
	}
	if m.Audio == "" {
		return nil
	}
	pcm, err := base64.StdEncoding.DecodeString(m.Audio)
	if err != nil {
		return fmt.Errorf("asyncai tts: decode audio: %w", err)
	}
	f := frames.NewTTSAudioRawFrame(pcm, s.cfg.SampleRate, 1)
	f.ContextID = id
	s.host.AppendToAudioContext(id, f)
	return nil
}

// KeepaliveReady reports whether the connection is open, implementing
// tts.KeepaliveHandler.
func (s *synthesizer) KeepaliveReady() bool { return s.Connected() }

// SendKeepalive sends a blank transcript, on the active context if there is
// one, implementing tts.KeepaliveHandler. An interruption can clear the context
// without a new reply following, and a blank transcript naming no context
// keeps the connection open then.
func (s *synthesizer) SendKeepalive(ctx context.Context) error {
	msg := map[string]any{keyTranscript: " "}
	if id := s.svc.ActiveAudioContextID(); id != "" {
		msg[keyContextID] = id
	}
	return s.send(ctx, msg)
}

// RunTTS sends one sentence on its context and returns: the audio arrives on
// the receive loop. A connection that has closed is opened again first. A send
// that fails ends the context's synthesis with a stop frame and is reported.
func (s *synthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	if err := s.send(ctx, buildMsg(text, contextID, true)); err != nil {
		if yield != nil {
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = contextID
			_ = yield(stopped)
		}
		return fmt.Errorf("asyncai tts: %w", err)
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// FlushAudio has the server generate what it holds for the context, or for the
// active one when none is named, implementing tts.AudioFlusher.
func (s *synthesizer) FlushAudio(ctx context.Context, contextID string) {
	if contextID == "" {
		contextID = s.svc.ActiveAudioContextID()
	}
	if contextID == "" || !s.Connected() {
		return
	}
	_ = s.send(ctx, buildMsg(" ", contextID, true))
}

// OnAudioContextInterrupted closes the server-side context the user cut off,
// implementing tts.AudioContextInterrupter.
func (s *synthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	s.closeContext(context.WithoutCancel(ctx), contextID)
}

// OnAudioContextCompleted closes the server-side context once its audio has
// played, implementing tts.AudioContextCompleter. Async sends no signal when a
// context is exhausted, so the client closes it to free the server's resources.
func (s *synthesizer) OnAudioContextCompleted(ctx context.Context, contextID string) {
	s.closeContext(ctx, contextID)
}

// closeContext closes a server-side context. Async leaves closing contexts to
// the client, both on an interruption and once a context is done.
func (s *synthesizer) closeContext(ctx context.Context, contextID string) {
	if contextID == "" {
		return
	}
	msg := map[string]any{keyContextID: contextID, "close_context": true, keyTranscript: ""}
	if err := s.send(ctx, msg); err != nil && !errors.Is(err, errNotConnected) {
		slog.Error("asyncai tts: closing a context failed", "context", contextID, "err", err)
	}
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

var (
	_ tts.Synthesizer             = (*synthesizer)(nil)
	_ tts.ContextSynthesizer      = (*synthesizer)(nil)
	_ tts.WebsocketService        = (*synthesizer)(nil)
	_ tts.KeepaliveHandler        = (*synthesizer)(nil)
	_ tts.Starter                 = (*synthesizer)(nil)
	_ tts.AudioFlusher            = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*synthesizer)(nil)
	_ tts.AudioContextCompleter   = (*synthesizer)(nil)
	_ tts.SettingsHolder          = (*synthesizer)(nil)
	_ tts.SettingsUpdater         = (*synthesizer)(nil)
	_ tts.LanguageNamer           = (*synthesizer)(nil)
)
