package together

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
	"strings"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	ttsURL          = "wss://api.together.ai/v1/audio/speech/websocket"
	defaultTTSModel = "hexgrad/Kokoro-82M"
	defaultTTSVoice = "af_heart"
	// ttsSampleRate is Together's fixed TTS output rate; it streams 24 kHz PCM.
	ttsSampleRate = 24000
)

// errTTSNotConnected is returned when there is no connection to use.
//
//nolint:gochecknoglobals // sentinel error
var errTTSNotConnected = errors.New("together tts: websocket not connected")

// errTTS wraps an error the Together TTS session reports, which ends the
// connection so it is opened again.
//
//nolint:gochecknoglobals // sentinel error
var errTTS = errors.New("together: tts error")

// TTSConfig configures the Together AI streaming TTS service, which streams 24 kHz
// mono PCM over an OpenAI-compatible realtime WebSocket.
type TTSConfig struct {
	// APIKey is the Together AI API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the TTS model; empty uses a default.
	Model string
	// Voice is the voice id; empty uses a default.
	Voice string
	// Language for synthesis; the zero value uses English. Sent as the ISO 639-1
	// or lowercase locale code Together accepts.
	Language language.Language
	// MaxPartialLength caps the partial text length for streaming; nil omits it.
	MaxPartialLength *int
	// ExtraParams are model-specific synthesis options sent with the session
	// setup, such as "pronunciation_dict", a list of "<source>/<replacement>"
	// entries; nil sends none.
	ExtraParams map[string]any
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Together AI streaming TTS service.
//
// It holds one connection open for the call, opened when the pipeline starts.
// When it drops it is restored with backoff, and when that keeps failing the
// service reports a permanent error and becomes unusable. Together's audio
// names no request, so it is labeled with the context playing, and the next
// request's frames are held until this one's audio has played.
func NewTTS(cfg TTSConfig) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns its synthesizer alongside it.
func newTTS(cfg TTSConfig) (*tts.Base, *synthesizer) {
	s := newSynthesizer(cfg)
	b := tts.New("TogetherTTS", s)
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	s.svc = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	return b, s
}

// newSynthesizer builds the synthesizer NewTTS drives, with the defaults
// applied and its settings taken from cfg.
func newSynthesizer(cfg TTSConfig) *synthesizer {
	cfg = cfg.withTTSDefaults()
	s := &synthesizer{cfg: cfg, live: newTTSSettings(cfg)}
	s.refreshSession()
	return s
}

// withTTSDefaults fills unset fields with their defaults.
func (c TTSConfig) withTTSDefaults() TTSConfig {
	if c.URL == "" {
		c.URL = ttsURL
	}
	if c.Model == "" {
		c.Model = defaultTTSModel
	}
	if c.Voice == "" {
		c.Voice = defaultTTSVoice
	}
	if c.Language == "" {
		c.Language = language.English
	}
	return c
}

// togetherLanguages are the languages this provider has been verified against,
// under the code Together takes for each.
//
//nolint:gochecknoglobals // read-only lookup table
var togetherLanguages = map[language.Language]string{
	language.English:    "en",
	language.Spanish:    "es",
	language.French:     "fr",
	language.Hindi:      "hi",
	language.Italian:    "it",
	language.Japanese:   "ja",
	language.Portuguese: "pt",
	language.Chinese:    "zh",
	language.ChineseHK:  "zh-hk",
}

// togetherLanguage maps a Language to a Together language code. Together accepts
// ISO 639-1 codes and lowercase locale codes ("zh-hk"); a regional variant
// without a verified locale code falls back to its base language code. The zero
// value maps to nothing, which sends no language.
func togetherLanguage(l language.Language) string {
	if l == "" {
		return ""
	}
	return language.Resolve(l, togetherLanguages, true)
}

// ttsService is the part of the TTS service the synthesizer asks about the
// audio contexts, and reports through. tts.Base implements it.
type ttsService interface {
	ActiveAudioContextID() string
	StopAllMetrics(ctx context.Context)
}

// synthesizer speaks over one connection held open for the call. A sentence is
// appended to the session's text buffer and committed, and the call returns;
// the audio arrives on the receive loop.
type synthesizer struct {
	cfg TTSConfig

	host tts.AudioContextHost
	svc  ttsService
	ws   *tts.Websocket

	// live is the settings store, updated on the frame goroutine and read only
	// there. The URL and the session update built from it are copied under
	// mu, since a redial uses them from the receive loop.
	live *TTSSettings

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	mu       sync.Mutex
	conn     *ttsConn
	endpoint string
	session  []byte
	// sessionID is the id Together gave the session on the open connection.
	sessionID string
}

// ttsConn is one connection, with the goroutine reading it.
//
// The reading runs apart from the receive loop so stopping the loop does not
// close the connection: closing it still waits for the server to answer.
type ttsConn struct {
	conn *wsutil.Conn
	// reads carries each message read, and finally the error that ended the
	// reading.
	reads chan ttsRead
	// stop is closed when the connection is being closed, which ends the reading
	// goroutine once nothing takes its messages any more.
	stop     chan struct{}
	stopOnce sync.Once
	// readEnded is set once the reading has failed, which is the connection
	// closing as far as the service is concerned.
	readEnded atomic.Bool
}

// ttsRead is one message read from the connection, or the error that ended
// the reading.
type ttsRead struct {
	typ  websocket.MessageType
	data []byte
	err  error
}

func newTTSConn(conn *wsutil.Conn) *ttsConn {
	c := &ttsConn{conn: conn, reads: make(chan ttsRead), stop: make(chan struct{})}
	go c.read()
	return c
}

// read reads the connection until it ends, handing each message to whoever
// receives from reads.
func (c *ttsConn) read() {
	for {
		typ, data, err := c.conn.Read(context.Background())
		if err != nil {
			c.readEnded.Store(true)
		}
		select {
		case c.reads <- ttsRead{typ: typ, data: data, err: err}:
		case <-c.stop:
			return
		}
		if err != nil {
			return
		}
	}
}

// halt stops handing messages on, for a connection about to be closed.
func (c *ttsConn) halt() { c.stopOnce.Do(func() { close(c.stop) }) }

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *synthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports Together's fixed PCM output rate.
func (s *synthesizer) SampleRate() int { return ttsSampleRate }

// Metadata reports the Together model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	model, _ := s.live.Model.Value()
	voice, _ := s.live.Voice.Value()
	return tts.Metadata{Model: model, VoiceID: voice}
}

// refreshSession builds the URL a connection is dialed on and the session
// update sent once it is open, from the settings store. It runs on the frame
// goroutine, which is where the store is written.
func (s *synthesizer) refreshSession() {
	endpoint := s.buildWebsocketURL()
	session := s.sessionUpdate()
	s.mu.Lock()
	s.endpoint, s.session = endpoint, session
	s.mu.Unlock()
}

// buildWebsocketURL builds the TTS WebSocket URL with the model, voice,
// language and partial length query parameters. A setting without a value is
// left off.
func (s *synthesizer) buildWebsocketURL() string {
	params := [][2]string{{"model", s.live.Model.Or("")}, {"voice", s.live.Voice.Or("")}}
	if lang := s.live.Language.Or(""); lang != "" {
		params = append(params, [2]string{"language", lang})
	}
	if n, ok := s.live.MaxPartialLength.Value(); ok {
		params = append(params, [2]string{"max_partial_length", strconv.Itoa(n)})
	}
	// Kokoro blends voices with a `+`-separated name such as
	// "af_bella(2)+af_heart(1)", which has to be escaped or the server reads
	// the `+` as a space and rejects the voice.
	query := make([]string, 0, len(params))
	for _, p := range params {
		query = append(query, p[0]+"="+queryEscape(p[1]))
	}
	return s.cfg.URL + "?" + strings.Join(query, "&")
}

// queryEscape percent-encodes a query value, leaving only unreserved characters
// and `/` as they are, so a space is sent as %20 rather than `+` and a model id
// keeps its slash.
func queryEscape(v string) string {
	escaped := strings.ReplaceAll(url.QueryEscape(v), "+", "%20")
	return strings.ReplaceAll(escaped, "%2F", "/")
}

// sessionUpdate builds the message that pins the voice on the session, along
// with the model-specific options when there are any.
func (s *synthesizer) sessionUpdate() []byte {
	session := map[string]any{"voice": s.live.Voice.AsAny()}
	if extra, ok := s.live.ExtraParams.Value(); ok && len(extra) > 0 {
		session["extra_params"] = extra
	}
	b, err := json.Marshal(map[string]any{msgType: "tts_session.updated", "session": session})
	if err != nil {
		slog.Warn("together tts: encoding the session update failed", "err", err)
		return nil
	}
	return b
}

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *synthesizer) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect opens the connection, then starts the receive loop, implementing
// tts.WebsocketService.
func (s *synthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect stops the receive loop and closes the connection, implementing
// tts.WebsocketService.
func (s *synthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens a connection unless one is open, implementing
// tts.WebsocketHandler, and pins the voice on its session. A dial that fails is
// reported here and leaves no connection behind, which a redial reads as a
// failed attempt. The session update failing is only warned about: the URL
// already names the voice.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	s.mu.Lock()
	endpoint, session := s.endpoint, s.session
	s.mu.Unlock()
	slog.Debug("connecting to together tts", "url", endpoint)
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	conn, err := s.ws.Dial(ctx, endpoint, header, wsutil.DefaultReadLimit)
	if err != nil {
		if s.host != nil {
			s.host.PushError(ctx, fmt.Sprintf("Error connecting to Together AI TTS: %v", err), err, false)
		}
		return nil
	}
	s.mu.Lock()
	old := s.conn
	s.conn = newTTSConn(conn)
	s.mu.Unlock()
	if old != nil {
		// A connection whose reading had already ended, replaced rather than
		// closed: nothing is left to say to the server on it.
		old.halt()
		_ = old.conn.CloseNow()
	}
	if err := conn.Write(ctx, websocket.MessageText, session); err != nil {
		slog.Warn("together tts: sending the voice setting failed", "err", err)
	} else {
		slog.Debug("together tts: sent the voice setting")
	}
	slog.Debug("connected to together tts")
	return nil
}

// DisconnectWebsocket closes the connection and forgets its session,
// implementing tts.WebsocketHandler.
func (s *synthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	s.mu.Lock()
	c := s.conn
	s.conn, s.sessionID = nil, ""
	s.mu.Unlock()
	if c == nil {
		return nil
	}
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Error disconnecting: %v", err), err, false)
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
// tts.WebsocketHandler. An error event ends it with that error, so the
// connection is opened again.
func (s *synthesizer) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errTTSNotConnected
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
			if err := s.handleMessage(ctx, r.data); err != nil {
				return err
			}
		}
	}
}

// ttsEvent is the subset of a Together TTS event read.
type ttsEvent struct {
	Type    string `json:"type"`
	Delta   string `json:"delta"`
	ItemID  string `json:"item_id"`
	Text    string `json:"text"`
	Session struct {
		ID    string          `json:"id"`
		Voice json.RawMessage `json:"voice"`
	} `json:"session"`
	Error json.RawMessage `json:"error"`
}

// ttsEventError is the error a failure event carries.
type ttsEventError struct {
	Message string `json:"message"`
	Code    string `json:"code"`
}

// handleMessage handles one server event. Together does not say which request
// an event belongs to, so it is taken to be the context playing. Only an error
// event fails the connection; a message that is not JSON is warned about and
// skipped.
func (s *synthesizer) handleMessage(ctx context.Context, data []byte) error {
	var evt ttsEvent
	if err := json.Unmarshal(data, &evt); err != nil {
		slog.Warn("together tts: failed to parse a websocket message", "err", err)
		return nil
	}
	switch evt.Type {
	case "session.created":
		s.mu.Lock()
		s.sessionID = evt.Session.ID
		s.mu.Unlock()
		slog.Debug("together tts: session created", "session", evt.Session.ID)
	case "session.updated":
		if len(evt.Session.Voice) > 0 {
			slog.Debug("together tts: voice updated", "voice", string(evt.Session.Voice))
		}
	case "conversation.item.input_text.received":
		slog.Debug("together tts: text received", "text", truncate(evt.Text, 50))
	case "conversation.item.audio_output.delta":
		s.handleAudioDelta(ctx, evt)
	case "conversation.item.audio_output.done":
		s.handleAudioDone(ctx, evt)
	case "conversation.item.tts.failed":
		s.handleTTSFailed(ctx, evt)
	case "error":
		return s.handleError(ctx, evt)
	default:
		slog.Debug("together tts: unhandled event", "type", evt.Type)
	}
	return nil
}

// handleAudioDelta appends a chunk of synthesized audio to the context
// playing. Audio that cannot be decoded is reported and dropped.
func (s *synthesizer) handleAudioDelta(ctx context.Context, evt ttsEvent) {
	if evt.Delta == "" || s.host == nil {
		return
	}
	pcm, err := base64.StdEncoding.DecodeString(evt.Delta)
	if err != nil {
		s.host.PushError(ctx, fmt.Sprintf("Error processing audio delta: %v", err), err, false)
		return
	}
	contextID := s.svc.ActiveAudioContextID()
	f := frames.NewTTSAudioRawFrame(pcm, ttsSampleRate, 1)
	f.ContextID = contextID
	s.host.AppendToAudioContext(contextID, f)
}

// handleAudioDone ends the context playing once Together has generated all of
// a request's audio, which ends it at once rather than after the stop-frame
// timeout.
func (s *synthesizer) handleAudioDone(ctx context.Context, evt ttsEvent) {
	slog.Debug("together tts: audio generation complete", "item", evt.ItemID)
	s.svc.StopAllMetrics(ctx)
	contextID := s.svc.ActiveAudioContextID()
	if contextID == "" || s.host == nil {
		return
	}
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	s.host.AppendToAudioContext(contextID, stopped)
	s.host.RemoveAudioContext(contextID)
}

// handleTTSFailed reports a synthesis Together could not complete, and stops
// the context playing.
func (s *synthesizer) handleTTSFailed(ctx context.Context, evt ttsEvent) {
	contextID := s.svc.ActiveAudioContextID()
	if s.host != nil {
		s.host.PushError(ctx, "TTS error: "+errorDetail(evt.Error), nil, false)
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		_ = s.host.PushFrame(ctx, stopped, processor.Downstream)
	}
	s.svc.StopAllMetrics(ctx)
}

// handleError reports an error that ends the session, and returns it so the
// connection is opened again.
func (s *synthesizer) handleError(ctx context.Context, evt ttsEvent) error {
	var e ttsEventError
	_ = json.Unmarshal(evt.Error, &e)
	if e.Message == "" {
		e.Message = "Unknown error"
	}
	msg := fmt.Sprintf("Together AI TTS error [%s]: %s", e.Code, e.Message)
	if s.host != nil {
		s.host.PushError(ctx, msg, nil, false)
	}
	return fmt.Errorf("%w: [%s] %s", errTTS, e.Code, e.Message)
}

// errorDetail is what a failure event says went wrong, as Together sent it.
func errorDetail(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return "{}"
	}
	return string(raw)
}

// truncate shortens text for a log line.
func truncate(text string, n int) string {
	r := []rune(text)
	if len(r) <= n {
		return text
	}
	return string(r[:n]) + "..."
}

// OnAudioContextInterrupted clears the text Together is still holding when the
// user cuts in, implementing tts.AudioContextInterrupter.
func (s *synthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	s.svc.StopAllMetrics(ctx)
	if contextID != "" {
		s.wsSend(context.WithoutCancel(ctx), map[string]any{msgType: "input_text_buffer.clear"})
	}
}

// FlushAudio commits the text buffer, which has Together generate what it is
// still holding, implementing tts.AudioFlusher.
func (s *synthesizer) FlushAudio(ctx context.Context, _ string) {
	if !s.Connected() {
		return
	}
	slog.Debug("together tts: flushing audio")
	s.wsSend(ctx, map[string]any{msgType: "input_text_buffer.commit"})
}

// RunTTS appends one sentence to the session's text buffer and commits it, and
// returns: the audio arrives on the receive loop. A connection that has closed
// is opened again first. A send that fails is reported by wsSend and does not
// fail the sentence.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, _ func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	s.wsSend(ctx, map[string]any{msgType: "input_text_buffer.append", "text": text})
	s.wsSend(ctx, map[string]any{msgType: "input_text_buffer.commit"})
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// wsSend writes one JSON message on the open connection. Nothing is sent while
// the service disconnects or without a connection, and a send that fails then
// is not reported either: the connection was going away. Any other failure is
// reported.
func (s *synthesizer) wsSend(ctx context.Context, msg map[string]any) {
	if s.ws.Disconnecting() {
		return
	}
	err := s.send(ctx, msg)
	if err == nil || errors.Is(err, errTTSNotConnected) {
		return
	}
	s.mu.Lock()
	gone := s.conn == nil
	s.mu.Unlock()
	if s.ws.Disconnecting() || gone || s.host == nil {
		return
	}
	s.host.PushError(ctx, fmt.Sprintf("Error sending message: %v", err), err, false)
}

// send writes one JSON message on the open connection.
func (s *synthesizer) send(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errTTSNotConnected
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.conn.Write(ctx, websocket.MessageText, payload)
}

var (
	_ tts.Synthesizer             = (*synthesizer)(nil)
	_ tts.ContextSynthesizer      = (*synthesizer)(nil)
	_ tts.Describer               = (*synthesizer)(nil)
	_ tts.WebsocketService        = (*synthesizer)(nil)
	_ tts.Starter                 = (*synthesizer)(nil)
	_ tts.AudioFlusher            = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*synthesizer)(nil)
)
