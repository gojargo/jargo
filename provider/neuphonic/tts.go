package neuphonic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

// stopMarker ends a piece of text: Neuphonic generates what it holds once it
// sees it.
const stopMarker = "<STOP>"

// keyText is the message key Neuphonic reads text from.
const keyText = "text"

// keepaliveInterval is how often an idle connection is sent an empty text, so
// the server keeps it open through silences.
const keepaliveInterval = 10 * time.Second

// errNotConnected is returned when there is no connection to read.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("neuphonic tts: websocket not connected")

// NewTTS builds a Neuphonic TTS service.
//
// It holds one connection open for the call, opened when the pipeline starts.
// When it drops it is restored with backoff, and when that keeps failing the
// service reports a permanent error and becomes unusable. Neuphonic's audio
// names no request, so an interruption while the bot speaks resets the
// connection, and the next request's frames are held until this one's audio
// has played. See tts.NewInterruptibleWebsocket.
func NewTTS(cfg Config) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns its synthesizer alongside it.
func newTTS(cfg Config) (*tts.Base, *synthesizer) {
	s := newSynthesizer(cfg)
	b := tts.New("NeuphonicTTS", s)
	// The service names the language the settings start with, so the URL is
	// built again from the store as it now stands.
	s.refreshEndpoint()
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	s.svc = b
	s.ws = tts.NewInterruptibleWebsocket(b, s, cfg.Websocket)
	return b, s
}

// newSynthesizer builds the synthesizer, with the defaults applied and its
// settings taken from cfg.
func newSynthesizer(cfg Config) *synthesizer {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Speed == 0 {
		cfg.Speed = defaultSpeed
	}
	s := &synthesizer{cfg: cfg, live: newTTSSettings(cfg)}
	s.refreshEndpoint()
	return s
}

// ttsService is the part of the TTS service the synthesizer asks about the
// audio contexts, and reports through. tts.Base implements it.
type ttsService interface {
	ActiveAudioContextID() string
	StopAllMetrics(ctx context.Context)
}

// synthesizer speaks over one connection held open for the call. A sentence is
// sent and the call returns; the audio arrives on the receive loop.
type synthesizer struct {
	cfg  Config
	host tts.AudioContextHost
	svc  ttsService
	ws   *tts.Websocket

	// live is the settings store, which the base updates on the frame
	// goroutine. It is read only there; the URL built from it is copied into
	// endpoint under mu, since a redial dials it from the receive loop.
	live *TTSSettings

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	mu       sync.Mutex
	conn     *wsConn
	endpoint string
}

// wsConn is one connection, with the goroutine reading it.
//
// The reading runs apart from the receive loop so stopping the loop does not
// close the connection: closing it still waits for the server to answer.
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

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// refreshEndpoint builds the URL a connection is dialed on from the settings
// store: the language in the path, and the synthesis options as query
// parameters, leaving out those that have no value. It runs on the frame
// goroutine, which is where the store is written.
func (s *synthesizer) refreshEndpoint() {
	lang := s.live.Language.Or("")
	q := url.Values{}
	q.Set("lang_code", lang)
	if speed, ok := s.live.Speed.Value(); ok {
		q.Set("speed", strconv.FormatFloat(speed, 'g', -1, 64))
	}
	if temperature, ok := s.live.Temperature.Value(); ok {
		q.Set("temperature", strconv.FormatFloat(temperature, 'g', -1, 64))
	}
	q.Set("encoding", s.cfg.Encoding)
	q.Set("sampling_rate", strconv.Itoa(s.cfg.SampleRate))
	if voice, ok := s.live.Voice.Value(); ok && voice != "" {
		q.Set("voice_id", voice)
	}
	if ipaKey, ok := s.live.IPAKey.Value(); ok && ipaKey != "" {
		q.Set("ipa_key", ipaKey)
	}
	endpoint := fmt.Sprintf("%s/speak/%s?%s", s.cfg.URL, lang, q.Encode())
	s.mu.Lock()
	s.endpoint = endpoint
	s.mu.Unlock()
}

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *synthesizer) Start(ctx context.Context) {
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

// ConnectWebsocket opens a connection unless one is open, implementing
// tts.WebsocketHandler. Everything that configures the synthesis travels in the
// URL, so there is nothing to send once it is open. A failure is reported here
// and leaves no connection behind, which a redial reads as a failed attempt.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to neuphonic tts")
	s.mu.Lock()
	endpoint := s.endpoint
	s.mu.Unlock()
	header := http.Header{}
	header.Set("x-api-key", s.cfg.APIKey)
	conn, err := s.ws.Dial(ctx, endpoint, header, wsutil.DefaultReadLimit)
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
// tts.WebsocketHandler. What the service was timing is reported as far as it
// got first: the audio it was waiting for will not come on this connection.
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
	slog.Debug("disconnecting from neuphonic tts")
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
	}
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
			if r.typ != websocket.MessageText {
				continue
			}
			if err := s.handleMessage(r.data); err != nil {
				return err
			}
		}
	}
}

// wsMessage is the subset of a Neuphonic WebSocket message read.
type wsMessage struct {
	Data *struct {
		Audio string `json:"audio"`
	} `json:"data"`
}

// handleMessage handles one server message: audio is appended to the context
// playing, since Neuphonic does not say which request it belongs to. Anything
// else is ignored.
func (s *synthesizer) handleMessage(data []byte) error {
	var m wsMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("neuphonic tts: decode message: %w", err)
	}
	if m.Data == nil || m.Data.Audio == "" || s.host == nil {
		return nil
	}
	pcm, err := base64.StdEncoding.DecodeString(m.Data.Audio)
	if err != nil {
		return fmt.Errorf("neuphonic tts: decode audio: %w", err)
	}
	contextID := s.svc.ActiveAudioContextID()
	f := frames.NewTTSAudioRawFrame(pcm, s.cfg.SampleRate, 1)
	f.ContextID = contextID
	s.host.AppendToAudioContext(contextID, f)
	return nil
}

// KeepaliveReady reports whether the connection is open, implementing
// tts.KeepaliveHandler.
func (s *synthesizer) KeepaliveReady() bool { return s.Connected() }

// SendKeepalive sends an empty text, which keeps the connection open through
// silences, implementing tts.KeepaliveHandler.
func (s *synthesizer) SendKeepalive(ctx context.Context) error {
	return s.send(ctx, map[string]any{keyText: ""})
}

// RunTTS sends one sentence and returns: the audio arrives on the receive loop.
// A connection that has closed is opened again first. A send that fails ends
// the context's synthesis with a stop frame, is reported, and resets the
// connection, whose state on the server is unknown from here.
func (s *synthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	if err := s.sendText(ctx, text); err != nil {
		if yield != nil {
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = contextID
			_ = yield(stopped)
		}
		s.Disconnect(ctx)
		s.Connect(ctx)
		return fmt.Errorf("neuphonic tts: %w", err)
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// sendText sends a sentence with the marker that has Neuphonic generate it at
// once.
func (s *synthesizer) sendText(ctx context.Context, text string) error {
	msg := map[string]any{keyText: text + " " + stopMarker}
	slog.Debug("neuphonic tts: sending text", "msg", msg)
	return s.send(ctx, msg)
}

// FlushAudio has Neuphonic generate what it holds, implementing
// tts.AudioFlusher.
func (s *synthesizer) FlushAudio(ctx context.Context, _ string) {
	if err := s.send(ctx, map[string]any{keyText: stopMarker}); err != nil {
		slog.Warn("neuphonic tts: flushing failed", "err", err)
	}
}

// send writes one JSON message on the open connection. With no connection
// there is nothing to send on, and nothing is sent: reopening it is the
// caller's business.
func (s *synthesizer) send(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return nil
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.conn.Write(ctx, websocket.MessageText, payload)
}

// neuphonicLanguage maps a Language to Neuphonic's language code. The zero value
// falls back to English; unmapped languages fall back to their base code.
func neuphonicLanguage(l language.Language) string {
	switch base := l.BaseCode(); base {
	case "":
		return "en"
	case "hi":
		return "HI"
	default:
		return base
	}
}

var (
	_ tts.Synthesizer        = (*synthesizer)(nil)
	_ tts.ContextSynthesizer = (*synthesizer)(nil)
	_ tts.WebsocketService   = (*synthesizer)(nil)
	_ tts.KeepaliveHandler   = (*synthesizer)(nil)
	_ tts.Starter            = (*synthesizer)(nil)
	_ tts.AudioFlusher       = (*synthesizer)(nil)
)
