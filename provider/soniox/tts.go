package soniox

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttssoniox "github.com/gojargo/jargo/adapter/tts/soniox"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
	errs "github.com/gojargo/jargo/utils/errors"
	ttstext "github.com/gojargo/jargo/utils/text"
)

const (
	defaultTTSURL   = "wss://tts-rt.soniox.com/tts-websocket"
	defaultTTSModel = "tts-rt-v2"
	// defaultVoice is one of Soniox's stock voices.
	defaultVoice = "Bryce"
	// defaultTTSSampleRate is the PCM rate requested from Soniox.
	defaultTTSSampleRate = 24000
	// defaultTTSAudioFormat is raw 16-bit little-endian mono PCM, the format
	// the pipeline carries.
	defaultTTSAudioFormat = "pcm_s16le"
	// keyStreamID is the field naming the stream a message belongs to.
	keyStreamID = "stream_id"
	// keyModel is the field and the setting naming the synthesis model.
	keyModel = "model"
	// ttsKeepaliveInterval is how often an idle connection is told to stay
	// open. Soniox closes one after 20 to 30 seconds of silence, so this stays
	// well inside that.
	ttsKeepaliveInterval = 20 * time.Second
	// errorTypeInvalidStreamState is what Soniox answers a message for a
	// stream it no longer has.
	errorTypeInvalidStreamState = "invalid_stream_state"
)

// errTTSNotConnected is returned by a send that found no connection to send on,
// the dial having failed and been reported.
//
//nolint:gochecknoglobals // sentinel error
var errTTSNotConnected = errors.New("soniox tts: websocket not connected")

// TTSConfig configures the Soniox streaming TTS service.
type TTSConfig struct {
	// APIKey is the Soniox API key, sent on the WebSocket handshake. Required.
	APIKey string `validate:"required"`
	// URL overrides the synthesis WebSocket endpoint; empty uses the hosted
	// endpoint.
	URL string
	// Model is the synthesis model; empty uses a current default.
	Model string
	// Voice is a stock voice name (for example "Adrian") or the UUID of a cloned
	// voice belonging to the API key's project; empty uses a default.
	Voice string
	// Language for synthesis; the zero value uses English. Mapped to Soniox's
	// language code, which is the base code: a regional language is sent as
	// its base.
	Language language.Language
	// SampleRate is the PCM rate requested from Soniox and emitted downstream;
	// 0 uses 24 kHz. Soniox accepts 8000, 16000, 24000, 44100 and 48000.
	SampleRate int `validate:"omitempty,oneof=8000 16000 24000 44100 48000"`
	// AudioFormat is the audio format Soniox sends; empty uses "pcm_s16le",
	// the raw 16-bit PCM the pipeline carries. The audio is passed on as it
	// arrives, so any other format needs decoding downstream.
	AudioFormat string
	// Speed multiplies the speaking rate from 0.7 to 1.3; nil leaves it unset
	// and uses Soniox's default of 1.0.
	Speed *float64 `validate:"omitempty,min=0.7,max=1.3"`
	// ReduceSilence shortens the pauses between words so the speech flows more
	// tightly, without changing how fast the words themselves are spoken. Only
	// the models that support silence reduction accept it. Nil leaves it unset
	// and uses Soniox's default, off.
	ReduceSilence *bool
	// ClientReferenceID labels the request in Soniox's own usage records, so the
	// audio generated can be attributed to a customer or a session. Empty omits
	// it, and Soniox ignores it for a temporary API key.
	ClientReferenceID string
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.TextTypeSentence (the zero value) sends the turn's text to its
	// stream a sentence at a time, frames.TextTypeToken streams it as it is
	// generated.
	TextAggregation frames.TextType
	// WordTimestamps requests per-character timings and drives the word-aligned
	// text path: the TTS base emits a TTSTextFrame for each spoken word as its
	// audio plays, mapped back to its original written form, so the assistant
	// context records only what was actually spoken before an interruption. It
	// is on by default; set it to false to fall back to aggregated text frames.
	WordTimestamps *bool
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// wordTimestamps reports whether per-word timing is requested.
func (c TTSConfig) wordTimestamps() bool {
	return c.WordTimestamps == nil || *c.WordTimestamps
}

// withTTSDefaults fills the fields whose zero value stands for "use a default".
func (c TTSConfig) withTTSDefaults() TTSConfig {
	if c.URL == "" {
		c.URL = defaultTTSURL
	}
	if c.Model == "" {
		c.Model = defaultTTSModel
	}
	if c.Voice == "" {
		c.Voice = defaultVoice
	}
	if c.Language == "" {
		c.Language = language.English
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSSampleRate
	}
	if c.AudioFormat == "" {
		c.AudioFormat = defaultTTSAudioFormat
	}
	return c
}

// NewTTS builds a Soniox streaming TTS service.
//
// It holds one connection open for the call, opened when the pipeline starts,
// and speaks each turn on a stream of its own on it, named by the turn's audio
// context: the stream is opened as the turn starts, each sentence is sent to it
// as it is ready, and it is closed once the turn's text is complete. Soniox
// tags its audio with the stream, so the audio still arriving for a turn the
// user cut off is dropped rather than spoken over the next one, and the stream
// is canceled. When the connection drops it is restored with backoff, and when
// that keeps failing the service reports a permanent error and becomes
// unusable.
func NewTTS(cfg TTSConfig) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns its synthesizer alongside it.
func newTTS(cfg TTSConfig) (*tts.Base, *ttsSynthesizer) {
	s := newTTSSynthesizer(cfg)
	var syn tts.WebsocketService = s
	if s.cfg.wordTimestamps() {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the base
		// takes the word-aligned path solely when timings were requested.
		syn = &timedTTSSynthesizer{ttsSynthesizer: s}
	}
	b := tts.New("SonioxTTS", syn)
	s.svc = b
	s.ws = tts.NewWebsocket(b, syn, s.cfg.Websocket)
	if s.cfg.TextAggregation == frames.TextTypeToken {
		b.SetTextAggregator(ttstext.NewTokenAggregator())
	}
	return b, s
}

// newTTSSynthesizer builds the synthesizer, with the defaults applied and its
// settings taken from cfg.
func newTTSSynthesizer(cfg TTSConfig) *ttsSynthesizer {
	cfg = cfg.withTTSDefaults()
	s := &ttsSynthesizer{
		cfg:        cfg,
		live:       newTTSSettings(cfg),
		configured: map[string]bool{},
		canceled:   map[string]bool{},
		partials:   map[string]partialWord{},
	}
	s.refreshSpaceless()
	return s
}

// ttsService is the part of the TTS service the synthesizer asks about the
// audio contexts, and reports through. tts.Base implements it.
type ttsService interface {
	ActiveAudioContextID() string
	StopAllMetrics(ctx context.Context)
}

// ttsSynthesizer speaks over one connection held open for the call, with each
// turn on a stream of its own, named by the turn's audio context.
type ttsSynthesizer struct {
	cfg  TTSConfig
	host tts.AudioContextHost
	svc  ttsService
	ws   *tts.Websocket

	// live is the settings store, which the base updates on the frame
	// goroutine, where every stream's configuration is built from it.
	live *TTSSettings
	// spaceless says the language is written without spaces between words. It
	// is taken from live on the frame goroutine, and read by the receive loop.
	spaceless atomic.Bool

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex
	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	mu   sync.Mutex
	conn *ttsConn

	// stateMu guards the streams' bookkeeping, which the frame goroutine and
	// the receive loop both keep. It is never held across a call into the host:
	// appending blocks while the audio plays out.
	stateMu sync.Mutex
	// configured holds the streams whose configuration has been sent on this
	// connection. Soniox rejects a second configuration for a stream.
	configured map[string]bool
	// canceled holds the streams canceled on this connection. A stream can
	// end on its own as its cancel goes out, and Soniox answers that cancel
	// with invalid_stream_state.
	canceled map[string]bool
	// partials holds each stream's word still being assembled across timing
	// messages, since a word's characters can arrive split across them.
	partials map[string]partialWord
}

// partialWord is a word still being assembled, and when its first character
// starts, in seconds from the start of the stream.
type partialWord struct {
	word  string
	start float64
}

// timedTTSSynthesizer adds word-timestamp streaming on top of ttsSynthesizer.
// It implements tts.WordTimestamps.
type timedTTSSynthesizer struct {
	*ttsSynthesizer
}

// RunTTSTimed sends the sentence like RunTTS, implementing tts.WordTimestamps.
// The timings arrive on the receive loop with the audio, so the word callback
// is not used here; implementing the interface is what puts the base on the
// word-aligned path.
func (s *timedTTSSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	yield func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, yield)
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

// ttsRead is one message read from the connection, or the error that ended the
// reading.
type ttsRead struct {
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
		_, data, err := c.conn.Read(context.Background())
		if err != nil {
			c.readEnded.Store(true)
		}
		select {
		case c.reads <- ttsRead{data: data, err: err}:
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
func (s *ttsSynthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// ExpressionAdapter implements tts.ExpressionAdapterProvider: audio tags on
// tts-rt-v2, and none on the other models.
func (s *ttsSynthesizer) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	if strings.HasPrefix(s.live.Model.Or(""), "tts-rt-v2") {
		return ttssoniox.SonioxExpressionAdapter{}
	}
	return ttsadapter.BaseExpressionAdapter{}
}

// Metadata reports the Soniox model and voice synthesis is billed against.
func (s *ttsSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(""), VoiceID: s.live.Voice.Or("")}
}

// refreshSpaceless records whether the language in the settings is written
// without spaces between words. Chinese and Japanese are, so their character
// timings cannot be assembled into words by splitting on spaces.
func (s *ttsSynthesizer) refreshSpaceless() {
	base, _, _ := strings.Cut(s.live.Language.Or(""), "-")
	switch strings.ToLower(base) {
	case "zh", "ja":
		s.spaceless.Store(true)
	default:
		s.spaceless.Store(false)
	}
}

// spacelessLanguage reports whether the language is written without spaces
// between words.
func (s *ttsSynthesizer) spacelessLanguage() bool { return s.spaceless.Load() }

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *ttsSynthesizer) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect opens the connection, then starts the receive loop and the keepalive,
// implementing tts.WebsocketService.
func (s *ttsSynthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
		s.ws.CreateKeepaliveTask(ctx, ttsKeepaliveInterval, s)
	}
}

// Disconnect ends the audio contexts of the connection, stops the receive loop
// and the keepalive, and closes the connection, implementing
// tts.WebsocketService.
func (s *ttsSynthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	s.ws.CancelKeepaliveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens a connection unless one is open, implementing
// tts.WebsocketHandler. A failure is reported here and leaves no connection
// behind, which a redial reads as a failed attempt.
func (s *ttsSynthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to soniox tts")
	conn, err := s.ws.Dial(ctx, s.cfg.URL, authHeader(s.cfg.APIKey), readLimit)
	if err != nil {
		s.mu.Lock()
		old := s.conn
		s.conn = nil
		s.mu.Unlock()
		if old != nil {
			old.halt()
			_ = old.conn.CloseNow()
		}
		ef := frames.NewErrorFrame(fmt.Sprintf("Unable to connect to Soniox TTS: %v", err))
		ef.Err = err
		s.ws.ReportError(ctx, ef, false)
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
	return nil
}

// DisconnectWebsocket closes the connection, implementing
// tts.WebsocketHandler. What the service was timing is reported as far as it
// got first, and the bookkeeping of the streams opened on the connection is
// dropped: Soniox does not carry a stream over to another.
func (s *ttsSynthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	s.svc.StopAllMetrics(ctx)
	s.mu.Lock()
	c := s.conn
	s.conn = nil
	s.mu.Unlock()
	defer s.clearStreams()
	if c == nil {
		return nil
	}
	slog.Debug("disconnecting from soniox tts")
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Error closing Soniox websocket: %v", err), err, false)
	}
	return nil
}

// clearStreams forgets every stream of the connection.
func (s *ttsSynthesizer) clearStreams() {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	clear(s.configured)
	clear(s.canceled)
	clear(s.partials)
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *ttsSynthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.conn.readEnded.Load()
}

// send writes one JSON message on the open connection, serialized against the
// other writers, failing with errTTSNotConnected when there is none.
func (s *ttsSynthesizer) send(ctx context.Context, msg map[string]any) error {
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
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return c.conn.Write(ctx, websocket.MessageText, payload)
}

// KeepaliveReady reports whether the connection is open, implementing
// tts.KeepaliveHandler.
func (s *ttsSynthesizer) KeepaliveReady() bool { return s.Connected() }

// SendKeepalive resets Soniox's idle timeout without starting any synthesis,
// implementing tts.KeepaliveHandler.
func (s *ttsSynthesizer) SendKeepalive(ctx context.Context) error {
	return s.send(ctx, map[string]any{"keep_alive": true})
}

// buildConfigMsg is the configuration that opens the stream for contextID.
// What has no value is left out, and left to Soniox.
func (s *ttsSynthesizer) buildConfigMsg(contextID string) map[string]any {
	config := map[string]any{
		keyStreamID:    contextID,
		keyModel:       s.live.Model.Or(""),
		"voice":        s.live.Voice.Or(""),
		"audio_format": s.cfg.AudioFormat,
	}
	if lang, ok := s.live.Language.Value(); ok {
		config["language"] = lang
	}
	if speed, ok := s.live.Speed.Value(); ok {
		config["speed"] = speed
	}
	if reduce, ok := s.live.ReduceSilence.Value(); ok {
		config["reduce_silence"] = reduce
	}
	if ref, ok := s.live.ClientReferenceID.Value(); ok {
		config["client_reference_id"] = ref
	}
	// Character timings drive the word-aligned text frames.
	config["return_timestamps"] = s.cfg.wordTimestamps()
	if strings.HasPrefix(s.cfg.AudioFormat, "pcm_") {
		config["sample_rate"] = s.cfg.SampleRate
	}
	return config
}

// sendConfig opens the stream for contextID, unless it is open already:
// Soniox rejects a second configuration for a stream.
func (s *ttsSynthesizer) sendConfig(ctx context.Context, contextID string) error {
	s.stateMu.Lock()
	configured := s.configured[contextID]
	s.stateMu.Unlock()
	if configured {
		return nil
	}
	if err := s.send(ctx, s.buildConfigMsg(contextID)); err != nil {
		return err
	}
	s.stateMu.Lock()
	s.configured[contextID] = true
	s.stateMu.Unlock()
	return nil
}

// closeStream cancels a stream and forgets it. A cancel works on a stream in
// any state, but only on one Soniox has seen, so only a configured stream is
// canceled. A stream that ends on its own as the cancel goes out has Soniox
// answer invalid_stream_state, which the receive loop expects for a canceled
// stream.
func (s *ttsSynthesizer) closeStream(ctx context.Context, contextID string) {
	s.stateMu.Lock()
	configured := s.configured[contextID]
	s.stateMu.Unlock()
	if configured && s.Connected() {
		if err := s.send(ctx, map[string]any{keyStreamID: contextID, "cancel": true}); err != nil {
			slog.Warn("soniox tts: failed to cancel stream", "stream", contextID, "err", err)
		} else {
			s.stateMu.Lock()
			s.canceled[contextID] = true
			s.stateMu.Unlock()
		}
	}
	s.stateMu.Lock()
	delete(s.configured, contextID)
	s.stateMu.Unlock()
}

// OnTurnContextCreated opens the turn's stream as soon as the turn starts,
// implementing tts.TurnContextCreator, so Soniox sets it up while the first
// sentence is still being aggregated.
func (s *ttsSynthesizer) OnTurnContextCreated(ctx context.Context, contextID string) {
	if err := s.sendConfig(ctx, contextID); err != nil {
		slog.Warn("soniox tts: failed to open the turn's stream early", "stream", contextID, "err", err)
	}
}

// OnTurnContextCompleted cancels the turn's stream when no text ever reached
// it, implementing tts.TurnContextCompleter. A turn that produced no speech
// (one that only called tools, say) would otherwise leave the stream opened
// early to Soniox's idle timer. A stream that did receive text was closed by
// the flush that ends the turn.
func (s *ttsSynthesizer) OnTurnContextCompleted(ctx context.Context, contextID string) {
	if s.host.AudioContextAvailable(contextID) {
		return
	}
	s.closeStream(ctx, contextID)
}

// OnAudioContextInterrupted cancels the stream of a context the user cut off,
// implementing tts.AudioContextInterrupter, so Soniox stops generating audio
// nobody is listening to.
func (s *ttsSynthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	s.svc.StopAllMetrics(ctx)
	s.closeStream(ctx, contextID)
	s.stateMu.Lock()
	delete(s.partials, contextID)
	s.stateMu.Unlock()
}

// FlushAudio ends the stream's text, implementing tts.AudioFlusher, so Soniox
// generates what it holds and then terminates the stream. With no context
// named it ends the one playing.
func (s *ttsSynthesizer) FlushAudio(ctx context.Context, contextID string) {
	flushID := contextID
	if flushID == "" {
		flushID = s.svc.ActiveAudioContextID()
	}
	s.mu.Lock()
	connected := s.conn != nil
	s.mu.Unlock()
	if flushID == "" || !connected {
		return
	}
	if err := s.send(ctx, map[string]any{"text": "", "text_end": true, keyStreamID: flushID}); err != nil {
		slog.Warn("soniox tts: flushing the stream failed", "stream", flushID, "err", err)
	}
}

// RunTTS sends one sentence to the context's stream and returns: the audio
// arrives on the receive loop. The stream is opened first if this connection
// has not opened it yet, which is the case after a reconnect. A connection
// that has closed is opened again first. A send that fails ends the context's
// synthesis with a stop frame, is reported, and resets the connection, whose
// state on the server is unknown from here.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	err := s.sendConfig(ctx, contextID)
	if err == nil {
		err = s.send(ctx, map[string]any{"text": text, "text_end": false, keyStreamID: contextID})
	}
	if err == nil {
		tts.StartUsageMetrics(ctx, text)
		return nil
	}

	if yield != nil {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		_ = yield(stopped)
	}
	s.Disconnect(ctx)
	s.Connect(ctx)
	return fmt.Errorf("soniox tts: %w", err)
}

// ReceiveMessages handles the connection's messages until it ends, implementing
// tts.WebsocketHandler. A stream that fails does not end the connection: the
// other streams on it carry on.
func (s *ttsSynthesizer) ReceiveMessages(ctx context.Context) error {
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
			if err := s.handleMessage(ctx, r.data); err != nil {
				return err
			}
		}
	}
}

// ttsMessage is the subset of a Soniox synthesis message read.
type ttsMessage struct {
	StreamID string `json:"stream_id"`
	// Audio is a base64-encoded chunk of audio.
	Audio string `json:"audio"`
	// Timestamps carries per-character timing; it can arrive without audio.
	Timestamps *ttsTimings `json:"timestamps"`
	// Terminated marks the end of the stream, the point at which the last word
	// is complete.
	Terminated   bool            `json:"terminated"`
	ErrorCode    json.RawMessage `json:"error_code"`
	ErrorType    string          `json:"error_type"`
	ErrorMessage string          `json:"error_message"`
}

// ttsTimings is Soniox's per-character timing payload: one entry per character
// (spaces included) and the offset at which it starts, in seconds from the
// beginning of the stream.
type ttsTimings struct {
	Characters []string  `json:"characters"`
	Starts     []float64 `json:"character_start_times_seconds"`
}

// hasErrorCode reports whether the message is an error: it carries an error
// code, of whatever type.
func (m *ttsMessage) hasErrorCode() bool {
	return len(m.ErrorCode) > 0 && string(m.ErrorCode) != "null"
}

// errorCodeText is the error code as Soniox sent it, a string unquoted.
func (m *ttsMessage) errorCodeText() string {
	var text string
	if json.Unmarshal(m.ErrorCode, &text) == nil {
		return text
	}
	return string(m.ErrorCode)
}

// classifyErrorCode is the category of an error Soniox reports in a stream.
// Its error code is an HTTP status (402 when the account is out of credit, 401
// for a rejected key, and so on); a code that is missing or not a whole number
// leaves the category unset, which reports it as unknown.
func classifyErrorCode(code json.RawMessage) errs.Category {
	status, err := strconv.Atoi(string(code))
	if err != nil {
		return errs.Unset
	}
	return errs.ClassifyHTTPStatusCode(status)
}

// handleMessage handles one server message, dispatching it to the stream it
// names. A message that is not JSON is logged and stepped over; audio that is
// not base64 fails the connection.
func (s *ttsSynthesizer) handleMessage(ctx context.Context, data []byte) error {
	var m ttsMessage
	if err := json.Unmarshal(data, &m); err != nil {
		slog.Warn("soniox tts: received a message that is not JSON", "message", string(data), "err", err)
		return nil
	}
	streamID := m.StreamID

	switch {
	case m.hasErrorCode():
		s.handleError(ctx, &m)
		return nil
	case m.Terminated:
		s.handleTerminated(streamID)
		return nil
	}

	if m.Audio != "" && streamID != "" && s.host.AudioContextAvailable(streamID) {
		audio, err := base64.StdEncoding.DecodeString(m.Audio)
		if err != nil {
			return fmt.Errorf("soniox tts: decode audio: %w", err)
		}
		f := frames.NewTTSAudioRawFrame(audio, s.SampleRate(), 1)
		f.ContextID = streamID
		s.host.AppendToAudioContext(streamID, f)
	}

	if m.Timestamps != nil && streamID != "" && s.host.AudioContextAvailable(streamID) {
		if words := s.toWordTimes(streamID, m.Timestamps); len(words) > 0 {
			// A language written without spaces has its timings reported per
			// character, so each token already reads as continuous text: a
			// consumer joining them must add no spacing of its own.
			opts := tts.WordTimingOptions{IncludesInterFrameSpaces: s.spacelessLanguage()}
			s.host.AddWordTimestamps(streamID, words, opts)
		}
	}
	// An audio_end is informational; the stream ends with terminated.
	return nil
}

// handleError reports a stream's failure, ends its speech, and forgets it. A
// stream that had already ended when its cancel arrived is gone, as asked, and
// is not reported.
func (s *ttsSynthesizer) handleError(ctx context.Context, m *ttsMessage) {
	streamID := m.StreamID
	s.stateMu.Lock()
	canceled := s.canceled[streamID]
	s.stateMu.Unlock()
	if m.ErrorType == errorTypeInvalidStreamState && canceled {
		slog.Debug("soniox tts: stream had already ended when canceled", "stream", streamID)
	} else {
		msg := fmt.Sprintf("Soniox TTS error %s %s (stream %s): %s",
			m.errorCodeText(), m.ErrorType, streamID, m.ErrorMessage)
		var opts []processor.ErrorOption
		if category := classifyErrorCode(m.ErrorCode); category != errs.Unset {
			opts = append(opts, processor.WithErrorCategory(category))
		}
		s.host.PushError(ctx, msg, nil, false, opts...)
	}
	if streamID != "" && s.host.AudioContextAvailable(streamID) {
		s.finishContext(streamID)
	}
	s.stateMu.Lock()
	delete(s.partials, streamID)
	delete(s.configured, streamID)
	delete(s.canceled, streamID)
	s.stateMu.Unlock()
}

// handleTerminated ends a stream: the word it was still assembling has no
// space to close it and is reported now, and its stop frame rides the queue
// behind the audio.
func (s *ttsSynthesizer) handleTerminated(streamID string) {
	s.stateMu.Lock()
	final := s.partials[streamID]
	delete(s.partials, streamID)
	delete(s.configured, streamID)
	s.stateMu.Unlock()
	if streamID == "" || !s.host.AudioContextAvailable(streamID) {
		return
	}
	if final.word != "" {
		s.host.AddWordTimestamps(streamID,
			[]uctx.WordTiming{{Word: final.word, Offset: final.start}}, tts.WordTimingOptions{})
	}
	s.finishContext(streamID)
}

// finishContext closes a context out: the stop frame rides the queue behind the
// audio, so it is pushed only once the last of that audio has been.
func (s *ttsSynthesizer) finishContext(contextID string) {
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	s.host.AppendToAudioContext(contextID, stopped)
	s.host.RemoveAudioContext(contextID)
}

// toWordTimes turns one timing message into words, each with when it starts.
//
// Soniox sends two parallel lists, one character and one start time each, and
// a word's characters can arrive split across messages, so the word still
// being assembled is carried per stream into the next. A word ends at a space.
// An audio tag is one word, spaces and all, such as [long pause], and the
// asterisks that stress a word, as in *really*, are left out of it. Chinese and
// Japanese never use spaces, so each of their characters is a word, punctuation
// left out. Lists that do not line up are reported and yield nothing.
func (s *ttsSynthesizer) toWordTimes(streamID string, t *ttsTimings) []uctx.WordTiming {
	chars, starts := t.Characters, t.Starts
	if len(chars) != len(starts) {
		slog.Error("soniox tts: timestamp mismatch", "characters", len(chars), "starts", len(starts))
		return nil
	}

	if s.spacelessLanguage() {
		words, _ := uctx.CharsAsWords(chars, starts)
		return words
	}

	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	current := s.partials[streamID]
	var words []uctx.WordTiming
	for i, char := range chars {
		if char == "*" {
			continue
		}
		inTag := strings.LastIndex(current.word, "[") > strings.LastIndex(current.word, "]")
		if char == " " && !inTag {
			if current.word != "" {
				words = append(words, uctx.WordTiming{Word: current.word, Offset: current.start})
				current.word = ""
			}
			continue
		}
		if current.word == "" {
			current.start = starts[i]
		}
		current.word += char
	}
	s.partials[streamID] = current
	return words
}

// sonioxTTSLanguages are the languages Soniox speaks, each under the code it
// takes for it. A regional variant resolves through its base language.
//
//nolint:gochecknoglobals // read-only lookup table
var sonioxTTSLanguages = map[language.Language]string{
	language.Afrikaans:       "af",
	language.Arabic:          "ar",
	language.Azerbaijani:     "az",
	language.Belarusian:      "be",
	language.Bulgarian:       "bg",
	language.Bengali:         "bn",
	language.Bosnian:         "bs",
	language.Catalan:         "ca",
	language.Czech:           "cs",
	language.Welsh:           "cy",
	language.Danish:          "da",
	language.German:          "de",
	language.Greek:           "el",
	language.English:         "en",
	language.Spanish:         "es",
	language.Estonian:        "et",
	language.Basque:          "eu",
	language.Persian:         "fa",
	language.Finnish:         "fi",
	language.French:          "fr",
	language.Galician:        "gl",
	language.Gujarati:        "gu",
	language.Hebrew:          "he",
	language.Hindi:           "hi",
	language.Croatian:        "hr",
	language.Hungarian:       "hu",
	language.Indonesian:      "id",
	language.Icelandic:       "is",
	language.Italian:         "it",
	language.Japanese:        "ja",
	language.Kazakh:          "kk",
	language.Kannada:         "kn",
	language.Korean:          "ko",
	language.Lithuanian:      "lt",
	language.Latvian:         "lv",
	language.Macedonian:      "mk",
	language.Malayalam:       "ml",
	language.Marathi:         "mr",
	language.Malay:           "ms",
	language.NorwegianBokmal: "no",
	language.Dutch:           "nl",
	language.Norwegian:       "no",
	language.Punjabi:         "pa",
	language.Polish:          "pl",
	language.Portuguese:      "pt",
	language.Romanian:        "ro",
	language.Russian:         "ru",
	language.Slovak:          "sk",
	language.Slovenian:       "sl",
	language.Albanian:        "sq",
	language.Serbian:         "sr",
	language.Sundanese:       "su",
	language.Swedish:         "sv",
	language.Swahili:         "sw",
	language.Tamil:           "ta",
	language.Telugu:          "te",
	language.Thai:            "th",
	language.Tagalog:         "tl",
	language.Turkish:         "tr",
	language.Ukrainian:       "uk",
	language.Urdu:            "ur",
	language.Uzbek:           "uz",
	language.Vietnamese:      "vi",
	language.Chinese:         "zh",
}

// sonioxTTSLanguage maps a language to Soniox's code for it, the base code.
func sonioxTTSLanguage(l language.Language) string {
	return language.Resolve(l, sonioxTTSLanguages, true)
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*ttsSynthesizer)(nil)
	_ tts.WebsocketService        = (*ttsSynthesizer)(nil)
	_ tts.KeepaliveHandler        = (*ttsSynthesizer)(nil)
	_ tts.Starter                 = (*ttsSynthesizer)(nil)
	_ tts.AudioFlusher            = (*ttsSynthesizer)(nil)
	_ tts.AudioContextInterrupter = (*ttsSynthesizer)(nil)
	_ tts.TurnContextCreator      = (*ttsSynthesizer)(nil)
	_ tts.TurnContextCompleter    = (*ttsSynthesizer)(nil)
	_ tts.Describer               = (*ttsSynthesizer)(nil)
	_ tts.WordTimestamps          = (*timedTTSSynthesizer)(nil)
	_ tts.WebsocketService        = (*timedTTSSynthesizer)(nil)
)
