package grok

import (
	"bytes"
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

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

const (
	// defaultTTSWSURL is the hosted streaming synthesis WebSocket.
	defaultTTSWSURL = "wss://api.x.ai/v1/tts"
	// defaultTTSHTTPURL is the hosted batch synthesis endpoint.
	defaultTTSHTTPURL = "https://api.x.ai/v1/tts"
	// defaultVoice is a current xAI voice.
	defaultVoice = "eve"
	// defaultTTSSampleRate is the PCM rate jargo requests from xAI.
	defaultTTSSampleRate = 24000
	// ttsCodec is raw 16-bit little-endian mono PCM. xAI also offers wav, mulaw
	// and alaw, but the pipeline carries linear PCM, so the codec is fixed.
	ttsCodec = "pcm"
	// ttsLanguageAuto lets xAI detect the language.
	ttsLanguageAuto = "auto"
	// ttsReadLimit bounds a single inbound message. Audio arrives base64-encoded
	// and timing payloads can cover several seconds of speech at once.
	ttsReadLimit = 1 << 20
)

// keyType is the key naming the type of every message, sent and received.
const keyType = "type"

// xAI TTS message types.
const (
	// ttsClientDelta appends text to the utterance being synthesized.
	ttsClientDelta = "text.delta"
	// ttsClientDone ends the utterance and starts synthesis.
	ttsClientDone = "text.done"
	// ttsClientClear drops the utterance being synthesized.
	ttsClientClear = "text.clear"
	// ttsEventDelta carries a chunk of audio, timing, or both.
	ttsEventDelta = "audio.delta"
	// ttsEventDone ends the utterance's audio.
	ttsEventDone = "audio.done"
	// ttsEventClear acknowledges a text.clear.
	ttsEventClear = "audio.clear"
	// ttsEventError reports a server-side failure.
	ttsEventError = "error"
)

// errTTSNotConnected is returned when there is no connection to use.
//
//nolint:gochecknoglobals // sentinel error
var errTTSNotConnected = errors.New("xai tts: websocket not connected")

// TTSConfig configures the xAI text-to-speech services. Both the streaming
// WebSocket service (NewTTS) and the batch HTTP one (NewHTTPTTS) read it;
// WordTimestamps and Websocket apply only to the streaming service.
type TTSConfig struct {
	// APIKey is the xAI API key, sent as a Bearer token. Required.
	APIKey string `validate:"required"`
	// URL overrides the synthesis endpoint; empty uses the hosted endpoint for
	// the transport being built.
	URL string
	// Voice is the xAI voice identifier; empty uses a default.
	Voice string
	// Language for synthesis; the zero value lets xAI detect it. xAI names a
	// few languages by region, so a regional language maps to its xAI code.
	Language language.Language
	// SampleRate is the PCM rate requested from xAI and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Speed multiplies the speaking rate from 0.7 to 1.5; nil leaves it unset.
	Speed *float64 `validate:"omitempty,min=0.7,max=1.5"`
	// OptimizeStreamingLatency picks a latency optimization level (0, 1 or 2);
	// nil leaves it unset.
	OptimizeStreamingLatency *int `validate:"omitempty,min=0,max=2"`
	// TextNormalization expands numbers, dates and abbreviations before
	// synthesis; nil leaves it unset.
	TextNormalization *bool
	// WordTimestamps requests per-character timings and drives the word-aligned
	// text path: the TTS base emits a TTSTextFrame for each spoken word as its
	// audio plays, mapped back to its original written form, so the assistant
	// context records only what was actually spoken before an interruption. It
	// is on by default; set it to false to fall back to aggregated text frames.
	// The batch HTTP service ignores it.
	WordTimestamps *bool
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering. The batch HTTP
	// service ignores it.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// wordTimestamps reports whether per-word timing is requested.
func (c TTSConfig) wordTimestamps() bool {
	return c.WordTimestamps == nil || *c.WordTimestamps
}

// applyTTSDefaults fills the fields whose zero value stands for "use a default".
func applyTTSDefaults(cfg TTSConfig, defaultURL string) TTSConfig {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSSampleRate
	}
	return cfg
}

// xaiLanguages are the languages xAI has been verified against, under the code
// it takes for each. It names a few languages by region.
//
//nolint:gochecknoglobals // read-only lookup table
var xaiLanguages = map[language.Language]string{
	language.Arabic:       "ar-EG",
	language.ArabicEG:     "ar-EG",
	language.ArabicSA:     "ar-SA",
	language.ArabicAE:     "ar-AE",
	language.Bengali:      "bn",
	language.German:       "de",
	language.English:      "en",
	language.Spanish:      "es-ES",
	language.SpanishES:    "es-ES",
	language.SpanishMX:    "es-MX",
	language.French:       "fr",
	language.Hindi:        "hi",
	language.Indonesian:   "id",
	language.Italian:      "it",
	language.Japanese:     "ja",
	language.Korean:       "ko",
	language.Portuguese:   "pt-PT",
	language.PortugueseBR: "pt-BR",
	language.PortuguesePT: "pt-PT",
	language.Russian:      "ru",
	language.Turkish:      "tr",
	language.Vietnamese:   "vi",
	language.Chinese:      "zh",
}

// xaiLanguage maps a Language to xAI's language code. A language xAI has not
// been verified against resolves through its base language; the zero value
// leaves detection to the server.
func xaiLanguage(l language.Language) string {
	if l == "" {
		return ttsLanguageAuto
	}
	return language.Resolve(l, xaiLanguages, true)
}

// NewTTS builds an xAI streaming TTS service.
//
// It holds one connection open for the call, opened when the pipeline starts.
// The sentences of a turn are streamed into one utterance, closed when the
// turn's text is complete, and the audio is labeled with the context playing,
// since xAI's messages name no request. When the connection drops it is
// restored with backoff, and when that keeps failing the service reports a
// permanent error and becomes unusable.
func NewTTS(cfg TTSConfig) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the streaming service and returns its synthesizer alongside it.
func newTTS(cfg TTSConfig) (*tts.Base, *ttsSynthesizer) {
	s := newTTSSynthesizer(cfg)
	b := tts.New("XAITTS", s.driven())
	// The service names the language the settings start with, so the URL is
	// built again from the store as it now stands.
	s.refreshEndpoint()
	s.svc = b
	s.ws = tts.NewWebsocket(b, s, s.cfg.Websocket)
	return b, s
}

// newTTSSynthesizer builds the streaming synthesizer, with the defaults applied
// and its settings taken from cfg.
func newTTSSynthesizer(cfg TTSConfig) *ttsSynthesizer {
	cfg = applyTTSDefaults(cfg, defaultTTSWSURL)
	s := &ttsSynthesizer{cfg: cfg, live: newWebsocketTTSSettings(cfg)}
	s.refreshEndpoint()
	return s
}

// driven is what the base drives. Only the timestamp-aware type implements
// tts.WordTimestamps, so the base pushes the text of each word as its audio
// plays solely when the timings were asked for, and the whole text otherwise.
func (s *ttsSynthesizer) driven() tts.Synthesizer {
	if s.cfg.wordTimestamps() {
		return &timedTTSSynthesizer{ttsSynthesizer: s}
	}
	return s
}

// NewHTTPTTS builds an xAI batch TTS service. It synthesizes each sentence in
// one request and streams the response body downstream. Prefer NewTTS: the
// streaming service starts speaking sooner and reports word timings.
func NewHTTPTTS(cfg TTSConfig) *tts.Base {
	cfg = applyTTSDefaults(cfg, defaultTTSHTTPURL)
	return tts.New("XAIHTTPTTS", &httpTTSSynthesizer{cfg: cfg, http: &http.Client{}})
}

// ttsService is the part of the TTS service the synthesizer asks about the
// audio contexts, and reports through. tts.Base implements it.
type ttsService interface {
	ActiveAudioContextID() string
	StopAllMetrics(ctx context.Context)
}

// ttsSynthesizer speaks over one xAI WebSocket held open for the call. A
// sentence is streamed into the utterance and the call returns; the audio
// arrives on the receive loop.
type ttsSynthesizer struct {
	cfg TTSConfig

	host tts.AudioContextHost
	svc  ttsService
	ws   *tts.Websocket

	// live is the settings store, updated on the frame goroutine and read only
	// there. The URL built from it is copied under mu, since a redial uses it
	// from the receive loop.
	live *WebsocketTTSSettings

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	mu       sync.Mutex
	conn     *ttsConn
	endpoint string

	// The character timings of an utterance are assembled into words across
	// the messages that carry them. tsMu guards that state, which the receive
	// loop builds and an interruption resets.
	tsMu sync.Mutex
	// tsContextID is the context the words being assembled belong to.
	tsContextID string
	words       uctx.CharAccumulator
}

// timedTTSSynthesizer adds word timing on top of ttsSynthesizer. It implements
// tts.WordTimestamps.
type timedTTSSynthesizer struct {
	*ttsSynthesizer
}

// RunTTSTimed streams the text like RunTTS, implementing tts.WordTimestamps.
// The timings arrive on the receive loop with the audio, so the word callback
// is not used here; implementing the interface is what has the base push the
// text of each word as its audio plays.
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
func (s *ttsSynthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the xAI voice synthesis is billed against. xAI selects the
// model by voice, so there is no separate model identifier.
func (s *ttsSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{VoiceID: s.live.Voice.Or(s.cfg.Voice)}
}

// RequiresTrailingSpace reports that each sentence is sent ending in a space,
// implementing tts.TrailingSpaceRequirer: the sentences of a turn are streamed
// into one utterance, so without it the last word of one would run into the
// first of the next.
func (s *ttsSynthesizer) RequiresTrailingSpace() bool { return true }

// refreshEndpoint builds the URL a connection is dialed on from the settings
// store. It runs on the frame goroutine, which is where the store is written.
func (s *ttsSynthesizer) refreshEndpoint() {
	endpoint := s.buildURL()
	s.mu.Lock()
	s.endpoint = endpoint
	s.mu.Unlock()
}

// buildURL builds the session URL. xAI bakes the audio parameters into the
// handshake, so a connection serves one voice, language and rate. A setting
// without a value is left off, and a language without one lets xAI detect it.
func (s *ttsSynthesizer) buildURL() string {
	q := url.Values{}
	q.Set("voice", s.live.Voice.Or(""))
	lang := s.live.Language.Or("")
	if lang == "" {
		lang = ttsLanguageAuto
	}
	q.Set("language", lang)
	q.Set("codec", ttsCodec)
	q.Set("sample_rate", strconv.Itoa(s.cfg.SampleRate))
	if v, ok := s.live.Speed.Value(); ok {
		q.Set("speed", strconv.FormatFloat(v, 'g', -1, 64))
	}
	if v, ok := s.live.OptimizeStreamingLatency.Value(); ok {
		q.Set("optimize_streaming_latency", strconv.Itoa(v))
	}
	if v, ok := s.live.TextNormalization.Value(); ok {
		q.Set("text_normalization", strconv.FormatBool(v))
	}
	if v, ok := s.live.WithTimestamps.Value(); ok {
		q.Set("with_timestamps", strconv.FormatBool(v))
	}
	return s.cfg.URL + "?" + q.Encode()
}

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *ttsSynthesizer) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect opens the connection, then starts the receive loop, implementing
// tts.WebsocketService.
func (s *ttsSynthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect stops the receive loop and closes the connection, implementing
// tts.WebsocketService.
func (s *ttsSynthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
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
	slog.Debug("connecting to xai tts")
	s.mu.Lock()
	endpoint := s.endpoint
	s.mu.Unlock()
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	conn, err := s.ws.Dial(ctx, endpoint, header, ttsReadLimit)
	if err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
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
// got first: the audio it was waiting for will not come on this connection.
func (s *ttsSynthesizer) DisconnectWebsocket(ctx context.Context) error {
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
	slog.Debug("disconnecting from xai tts")
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Error disconnecting from xAI TTS: %v", err), err, false)
	}
	return nil
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *ttsSynthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.conn.readEnded.Load()
}

// ReceiveMessages handles the connection's messages until it ends, implementing
// tts.WebsocketHandler.
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
			if r.typ != websocket.MessageText {
				slog.Warn("xai tts: unexpected binary frame")
				continue
			}
			if err := s.handleMessage(ctx, r.data); err != nil {
				return err
			}
		}
	}
}

// ttsMessage is the subset of an xAI TTS event read.
type ttsMessage struct {
	Type string `json:"type"`
	// Delta is a base64-encoded chunk of PCM.
	Delta string `json:"delta"`
	// Timestamps carries per-character timing; it can arrive without audio.
	Timestamps *ttsTimings `json:"audio_timestamps"`
	Message    string      `json:"message"`
	Error      string      `json:"error"`
}

// ttsTimings is xAI's per-character timing payload: one entry per character
// (spaces and punctuation included) and the matching [start, end] second pair,
// absolute from the start of the utterance.
type ttsTimings struct {
	Chars []string    `json:"graph_chars"`
	Times [][]float64 `json:"graph_times"`
}

// handleMessage handles one server message. xAI does not say which request a
// message belongs to, so it is taken to be the context playing. Audio that
// cannot be decoded fails the connection; a message that is not JSON is
// logged and skipped.
func (s *ttsSynthesizer) handleMessage(ctx context.Context, data []byte) error {
	var m ttsMessage
	if err := json.Unmarshal(data, &m); err != nil {
		slog.Error("xai tts: invalid JSON message", "msg", string(data), "err", err)
		return nil
	}
	contextID := s.svc.ActiveAudioContextID()
	switch m.Type {
	case ttsEventDelta:
		return s.handleAudioDelta(m, contextID)
	case ttsEventDone:
		s.handleAudioDone(ctx, contextID)
	case ttsEventError:
		s.handleError(ctx, m, data, contextID)
	case ttsEventClear:
		slog.Debug("xai tts: audio clear acknowledged")
	default:
		slog.Debug("xai tts: unhandled message type", "type", m.Type)
	}
	return nil
}

// handleAudioDone ends the context playing once xAI has generated all of the
// utterance's audio, after the trailing word, which has no terminating space
// to close it.
func (s *ttsSynthesizer) handleAudioDone(ctx context.Context, contextID string) {
	s.svc.StopAllMetrics(ctx)
	if contextID == "" {
		return
	}
	s.tsMu.Lock()
	var last []uctx.WordTiming
	if s.tsContextID == contextID {
		if w, ok := s.words.Flush(); ok {
			last = []uctx.WordTiming{w}
		}
	}
	s.resetTimestampStateLocked()
	s.tsMu.Unlock()
	if len(last) > 0 && s.host != nil {
		s.host.AddWordTimestamps(contextID, last, tts.WordTimingOptions{})
	}
	s.closeAudioContext(contextID)
}

// handleError ends the context playing and reports what xAI said went wrong.
// The connection stays up.
func (s *ttsSynthesizer) handleError(ctx context.Context, m ttsMessage, data []byte, contextID string) {
	s.svc.StopAllMetrics(ctx)
	s.resetTimestampState()
	detail := m.Message
	if detail == "" {
		detail = m.Error
	}
	if detail == "" {
		detail = string(data)
	}
	if contextID != "" {
		s.closeAudioContext(contextID)
	}
	if s.host != nil {
		s.host.PushError(ctx, "xAI TTS error: "+detail, nil, false)
	}
}

// handleAudioDelta appends a chunk of audio to the context playing, and the
// words its timings complete. Timings arrive in deltas of their own, with no
// audio, as well as alongside it. Nothing is appended with no context playing.
func (s *ttsSynthesizer) handleAudioDelta(m ttsMessage, contextID string) error {
	if contextID == "" || s.host == nil {
		return nil
	}
	if m.Delta != "" {
		pcm, err := base64.StdEncoding.DecodeString(m.Delta)
		if err != nil {
			return fmt.Errorf("xai tts: decode audio: %w", err)
		}
		f := frames.NewTTSAudioRawFrame(pcm, s.cfg.SampleRate, 1)
		f.ContextID = contextID
		s.host.AppendToAudioContext(contextID, f)
	}
	s.handleWordTimestamps(m.Timestamps, contextID)
	return nil
}

// handleWordTimestamps assembles a timing payload into words and hands the
// ones it completes to the context. A payload whose characters and times do
// not pair up is logged and dropped, keeping the word being assembled.
func (s *ttsSynthesizer) handleWordTimestamps(t *ttsTimings, contextID string) {
	if t == nil || len(t.Chars) == 0 || len(t.Times) == 0 {
		return
	}
	s.tsMu.Lock()
	// A new utterance clears any word carried over from the previous one.
	if contextID != s.tsContextID {
		s.resetTimestampStateLocked()
		s.tsContextID = contextID
	}
	words, err := s.words.Add(t.Chars, t.startOffsets())
	s.tsMu.Unlock()
	if err != nil {
		slog.Error("xai tts: timestamp length mismatch", "chars", len(t.Chars), "times", len(t.Times))
		return
	}
	if len(words) > 0 {
		s.host.AddWordTimestamps(contextID, words, tts.WordTimingOptions{})
	}
}

// startOffsets flattens xAI's [start, end] pairs to the start of each character,
// which is what the word accumulator aligns on.
func (t *ttsTimings) startOffsets() []float64 {
	starts := make([]float64, 0, len(t.Times))
	for _, pair := range t.Times {
		if len(pair) == 0 {
			starts = append(starts, 0)
			continue
		}
		starts = append(starts, pair[0])
	}
	return starts
}

// resetTimestampState forgets the words being assembled.
func (s *ttsSynthesizer) resetTimestampState() {
	s.tsMu.Lock()
	defer s.tsMu.Unlock()
	s.resetTimestampStateLocked()
}

// resetTimestampStateLocked forgets the words being assembled. Called with tsMu
// held.
func (s *ttsSynthesizer) resetTimestampStateLocked() {
	s.tsContextID = ""
	s.words.Reset()
}

// closeAudioContext marks a context finished: its stop frame goes out after
// its audio, and it is dropped.
func (s *ttsSynthesizer) closeAudioContext(contextID string) {
	if s.host == nil {
		return
	}
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	s.host.AppendToAudioContext(contextID, stopped)
	s.host.RemoveAudioContext(contextID)
}

// FlushAudio closes the utterance, which has xAI synthesize what it has
// buffered, implementing tts.AudioFlusher.
func (s *ttsSynthesizer) FlushAudio(ctx context.Context, _ string) {
	if !s.Connected() {
		return
	}
	if err := s.send(ctx, map[string]any{keyType: ttsClientDone}); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
	}
}

// OnAudioContextInterrupted drops the utterance xAI is synthesizing when the
// user cuts in, without reconnecting, implementing
// tts.AudioContextInterrupter.
func (s *ttsSynthesizer) OnAudioContextInterrupted(ctx context.Context, _ string) {
	s.svc.StopAllMetrics(ctx)
	if s.Connected() {
		if err := s.send(context.WithoutCancel(ctx), map[string]any{keyType: ttsClientClear}); err != nil {
			slog.Warn("xai tts: clearing the utterance failed", "err", err)
		}
	}
	s.resetTimestampState()
}

// RunTTS streams one sentence into the utterance and returns: the audio
// arrives on the receive loop. A connection that has closed is opened again
// first. A send that fails ends the context's synthesis with a stop frame, is
// reported, and resets the connection, whose state on the server is unknown
// from here.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	if err := s.send(ctx, map[string]any{keyType: ttsClientDelta, "delta": text}); err != nil {
		if yield != nil {
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = contextID
			_ = yield(stopped)
		}
		s.Disconnect(ctx)
		s.Connect(ctx)
		return fmt.Errorf("xai tts: unknown error occurred: %w", err)
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// send writes one JSON message on the open connection.
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
	return c.conn.Write(ctx, websocket.MessageText, payload)
}

// httpTTSSynthesizer synthesizes each sentence in a single HTTP request.
type httpTTSSynthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *httpTTSSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the xAI voice synthesis is billed against.
func (s *httpTTSSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{VoiceID: s.cfg.Voice}
}

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *httpTTSSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	payload := map[string]any{
		"text":     text,
		"voice_id": s.cfg.Voice,
		"language": xaiLanguage(s.cfg.Language),
		"output_format": map[string]any{
			"codec":       ttsCodec,
			"sample_rate": s.cfg.SampleRate,
		},
	}
	if s.cfg.Speed != nil {
		payload["speed"] = *s.cfg.Speed
	}
	if s.cfg.OptimizeStreamingLatency != nil {
		payload["optimize_streaming_latency"] = *s.cfg.OptimizeStreamingLatency
	}
	if s.cfg.TextNormalization != nil {
		payload["text_normalization"] = *s.cfg.TextNormalization
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.URL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamSpeech(s.http, req, text, emit)
}
