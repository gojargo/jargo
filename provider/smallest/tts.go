package smallest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

// keepaliveInterval is how often an idle connection is sent a blank text, so
// the server keeps it open through silences.
const keepaliveInterval = 30 * time.Second

// errNotConnected is returned when there is no connection to use.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("websocket not connected")

// NewTTS builds a Smallest AI TTS service.
//
// It holds one connection open for the call, opened when the pipeline starts.
// When it drops it is restored with backoff, and when that keeps failing the
// service reports a permanent error and becomes unusable. Smallest's audio
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
	b := tts.New("SmallestTTS", s.driven())
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	s.svc = b
	s.ws = tts.NewInterruptibleWebsocket(b, s, cfg.Websocket)
	return b, s
}

// newSynthesizer builds the synthesizer, with the defaults applied and its
// settings taken from cfg.
func newSynthesizer(cfg Config) *synthesizer {
	cfg = withDefaults(cfg)
	s := &synthesizer{
		cfg:            cfg,
		wordTimestamps: cfg.WordTimestamps == nil || *cfg.WordTimestamps,
		live:           newTTSSettings(cfg),
	}
	s.refreshKeepalive()
	return s
}

// driven is what the base drives. Only the timestamp-aware type implements
// tts.WordTimestamps, so the base pushes the text of each word as its audio
// plays solely when the timings were asked for, and the whole text otherwise.
func (s *synthesizer) driven() tts.Synthesizer {
	if s.wordTimestamps {
		return &timedSynthesizer{synthesizer: s}
	}
	return s
}

// withDefaults fills in what the caller left unset. The voice is resolved from
// the model rather than fixed, since the two models serve different voice
// catalogs and one default for both would name a voice the other does not
// serve. A model Smallest is not known to offer gets the default model's voice.
func withDefaults(cfg Config) Config {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		voice, ok := modelDefaultVoices[cfg.Model]
		if !ok {
			voice = modelDefaultVoices[defaultModel]
		}
		cfg.Voice = voice
	}
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = defaultOutputFormat
	}
	return cfg
}

// ttsService is the part of the TTS service the synthesizer asks about the
// audio contexts, and reports through. tts.Base implements it.
type ttsService interface {
	ActiveAudioContextID() string
	StopAllMetrics(ctx context.Context)
}

// synthesizer speaks over one connection held open for the call. The sentences
// of a turn are sent on the turn's context, each marked as continuing it, and
// the call returns; the audio arrives on the receive loop.
type synthesizer struct {
	cfg            Config
	wordTimestamps bool

	host tts.AudioContextHost
	svc  ttsService
	ws   *tts.Websocket

	// live is the settings store, which the base updates on the frame
	// goroutine. It is read only there; the keepalive built from it is copied
	// under mu, since the keepalive goes out from a goroutine of its own.
	live *TTSSettings

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	mu        sync.Mutex
	conn      *wsConn
	keepalive map[string]any

	// Smallest times the words of each request from that request's own audio,
	// starting again near zero, while every request of a turn plays on one
	// audio context. The duration of the requests already done is added to
	// the next one's timings to put them on the turn's timeline. A request
	// ends where the request ID changes, since Smallest says a turn is
	// complete but not a request. The receive loop updates these, and a new
	// turn resets them.
	wtMu sync.Mutex
	// cumulativeTime is the length, in seconds, of the turn's requests before
	// the one in flight.
	cumulativeTime float64
	// requestEndTime is the latest end of a word of the request in flight.
	requestEndTime float64
	// wtRequestID is the request in flight, or nil before the turn's first
	// word.
	wtRequestID *string
}

// timedSynthesizer adds word timing on top of synthesizer. It implements
// tts.WordTimestamps.
type timedSynthesizer struct {
	*synthesizer
}

// RunTTSTimed sends the text like RunTTS, implementing tts.WordTimestamps. The
// timings arrive on the receive loop with the audio, so the word callback is
// not used here; implementing the interface is what has the base push the text
// of each word as its audio plays.
func (s *timedSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	yield func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, yield)
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

// Metadata reports the Smallest model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.Voice}
}

// buildMsg builds a request carrying text on a context. Every fragment sharing
// the context is joined into one generation instead of each starting its
// prosody afresh; continueTranscript says more fragments are coming for it,
// and false marks the last, which closes the context once it has been spoken.
// It reads the settings store, so it runs on the frame goroutine.
func (s *synthesizer) buildMsg(text, contextID string, continueTranscript bool) map[string]any {
	msg := map[string]any{
		"text":        text,
		"voice_id":    s.live.Voice.AsAny(),
		"model":       s.live.Model.AsAny(),
		"language":    s.live.Language.AsAny(),
		"sample_rate": s.cfg.SampleRate,
		"context_id":  contextID,
		"continue":    continueTranscript,
	}
	if v, ok := s.live.Speed.Value(); ok {
		msg["speed"] = v
	}
	if v, ok := s.live.MathNotation.Value(); ok {
		msg["math_notation"] = v
	}
	if v, ok := s.live.ContentFilter.Value(); ok {
		msg["content_filter"] = map[string]any{"enabled": true, "action": v}
	}
	if v, ok := s.live.DateFormat.Value(); ok {
		msg["date_format"] = v
	}
	if v, ok := s.live.NumberPronunciationLanguage.Value(); ok {
		msg["number_pronunciation_language"] = v
	}
	if s.wordTimestamps {
		msg["word_timestamps"] = true
	}
	if s.cfg.MaxBufferDelayMs != nil {
		msg["max_buffer_delay_ms"] = *s.cfg.MaxBufferDelayMs
	}
	msg["output_format"] = s.cfg.OutputFormat
	return msg
}

// refreshKeepalive builds the keepalive from the settings store: a blank text
// in the voice, the model and the language the service speaks. It runs on the
// frame goroutine, which is where the store is written.
func (s *synthesizer) refreshKeepalive() {
	msg := map[string]any{
		"text":     " ",
		"voice_id": s.live.Voice.AsAny(),
		"model":    s.live.Model.AsAny(),
		"language": s.live.Language.AsAny(),
	}
	s.mu.Lock()
	s.keepalive = msg
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
// tts.WebsocketHandler. Every request carries its own settings, so there is
// nothing to send once it is open. A failure is reported here and leaves no
// connection behind, which a redial reads as a failed attempt.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to smallest tts")
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	header.Set("X-Source", "jargo")
	conn, err := s.ws.Dial(ctx, s.cfg.URL, header, wsutil.DefaultReadLimit)
	if err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Smallest TTS connection error: %v", err))
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
	slog.Debug("disconnecting from smallest tts")
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Smallest TTS error closing websocket: %v", err), err, false)
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
			if err := s.handleMessage(ctx, r.data); err != nil {
				return err
			}
		}
	}
}

// wsMessage is the subset of a Smallest WebSocket message read.
type wsMessage struct {
	Status    string  `json:"status"`
	RequestID *string `json:"request_id"`
	Data      struct {
		Audio string   `json:"audio"`
		Word  *string  `json:"word"`
		Start *float64 `json:"start"`
		End   *float64 `json:"end"`
	} `json:"data"`
	Error json.RawMessage `json:"error"`
}

// handleMessage handles one server message. Smallest does not say which
// request audio or a word belongs to, so it is taken to be the context
// playing.
func (s *synthesizer) handleMessage(ctx context.Context, data []byte) error {
	var m wsMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("smallest tts: decode message: %w", err)
	}
	if s.host == nil {
		return nil
	}
	switch m.Status {
	case "complete":
		s.svc.StopAllMetrics(ctx)
	case "chunk":
		pcm, err := base64.StdEncoding.DecodeString(m.Data.Audio)
		if err != nil {
			return fmt.Errorf("smallest tts: decode audio: %w", err)
		}
		contextID := s.svc.ActiveAudioContextID()
		f := frames.NewTTSAudioRawFrame(pcm, s.cfg.SampleRate, 1)
		f.ContextID = contextID
		s.host.AppendToAudioContext(contextID, f)
	case "word_timestamp":
		s.handleWordTimestamp(m)
	case "error":
		contextID := s.svc.ActiveAudioContextID()
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		_ = s.host.PushFrame(ctx, stopped, processor.Downstream)
		s.svc.StopAllMetrics(ctx)
		detail := string(data)
		if len(m.Error) > 0 {
			var text string
			if json.Unmarshal(m.Error, &text) == nil {
				detail = text
			} else {
				detail = string(m.Error)
			}
		}
		s.host.PushError(ctx, "Smallest TTS error: "+detail, nil, false)
	default:
		slog.Warn("smallest tts: unknown message status", "msg", string(data))
	}
	return nil
}

// handleWordTimestamp puts a word on the turn's timeline and hands it to the
// context playing. A word that names no start is not timed.
func (s *synthesizer) handleWordTimestamp(m wsMessage) {
	s.wtMu.Lock()
	s.advanceWordTimestampRequest(m.RequestID)
	if m.Data.Word == nil || m.Data.Start == nil {
		s.wtMu.Unlock()
		return
	}
	// Only the word and its start are timed; its end sizes the offset of the
	// next request.
	word := uctx.WordTiming{Word: *m.Data.Word, Offset: *m.Data.Start + s.cumulativeTime}
	if m.Data.End != nil {
		s.requestEndTime = max(s.requestEndTime, *m.Data.End)
	}
	s.wtMu.Unlock()
	s.host.AddWordTimestamps(s.svc.ActiveAudioContextID(), []uctx.WordTiming{word}, tts.WordTimingOptions{})
}

// advanceWordTimestampRequest moves the turn's offset on when the words cross
// into a new request: the span of the request just finished, up to the end of
// its last word, is added to the offset of the ones after it. Called with wtMu
// held.
func (s *synthesizer) advanceWordTimestampRequest(requestID *string) {
	if equalIDs(requestID, s.wtRequestID) {
		return
	}
	if s.wtRequestID != nil {
		s.cumulativeTime += s.requestEndTime
		s.requestEndTime = 0
	}
	s.wtRequestID = requestID
}

// equalIDs reports whether two request IDs, either of which may be missing,
// are the same.
func equalIDs(a, b *string) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// OnTurnContextCreated resets the word timing at the start of each turn,
// implementing tts.TurnContextCreator. A turn plays on a fresh audio context,
// so the offset built up over the last turn must not carry over.
func (s *synthesizer) OnTurnContextCreated(context.Context, string) {
	s.wtMu.Lock()
	defer s.wtMu.Unlock()
	s.cumulativeTime = 0
	s.requestEndTime = 0
	s.wtRequestID = nil
}

// KeepaliveReady reports whether the connection is open, implementing
// tts.KeepaliveHandler.
func (s *synthesizer) KeepaliveReady() bool { return s.Connected() }

// SendKeepalive sends a blank text, which keeps the connection open through
// silences, implementing tts.KeepaliveHandler.
func (s *synthesizer) SendKeepalive(ctx context.Context) error {
	s.mu.Lock()
	msg := s.keepalive
	s.mu.Unlock()
	return s.send(ctx, msg)
}

// RunTTS sends one fragment on its context, marked as continuing it, and
// returns: the audio arrives on the receive loop. A connection that has closed
// is opened again first. A send that fails ends the context's synthesis with a
// stop frame, is reported, and resets the connection, whose state on the
// server is unknown from here.
func (s *synthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	if err := s.send(ctx, s.buildMsg(text, contextID, true)); err != nil {
		if yield != nil {
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = contextID
			_ = yield(stopped)
		}
		s.Disconnect(ctx)
		s.Connect(ctx)
		return fmt.Errorf("smallest tts send error: %w", err)
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// FlushAudio closes the context, or the active one when none is named, with an
// empty last fragment, which has Smallest speak what it is still buffering
// for it. It implements tts.AudioFlusher.
func (s *synthesizer) FlushAudio(ctx context.Context, contextID string) {
	if contextID == "" {
		contextID = s.svc.ActiveAudioContextID()
	}
	s.mu.Lock()
	connected := s.conn != nil
	s.mu.Unlock()
	if contextID == "" || !connected {
		return
	}
	if err := s.send(ctx, s.buildMsg("", contextID, false)); err != nil {
		slog.Warn("smallest tts: flushing failed", "context", contextID, "err", err)
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
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.conn.Write(ctx, websocket.MessageText, payload)
}

var (
	_ tts.Synthesizer        = (*synthesizer)(nil)
	_ tts.ContextSynthesizer = (*synthesizer)(nil)
	_ tts.Describer          = (*synthesizer)(nil)
	_ tts.WebsocketService   = (*synthesizer)(nil)
	_ tts.KeepaliveHandler   = (*synthesizer)(nil)
	_ tts.Starter            = (*synthesizer)(nil)
	_ tts.AudioFlusher       = (*synthesizer)(nil)
	_ tts.TurnContextCreator = (*synthesizer)(nil)
	_ tts.WordTimestamps     = (*timedSynthesizer)(nil)
	_ tts.ContextSynthesizer = (*timedSynthesizer)(nil)
)
