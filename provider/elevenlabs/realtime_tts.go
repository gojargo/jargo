package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

// realtimeTTSPath is the multi-stream-input endpoint. Multi-stream is what
// carries a context id on every message, which is how audio is attributed to the
// synthesis that asked for it.
const realtimeTTSPath = "/v1/text-to-speech/%s/multi-stream-input"

// defaultRealtimeTTSHost is the ElevenLabs WebSocket host.
const defaultRealtimeTTSHost = "wss://api.elevenlabs.io"

// keepaliveInterval is how often an idle connection is sent empty text. The
// server drops a stream that has gone quiet, and between turns nothing is sent
// for as long as the user talks.
const keepaliveInterval = 10 * time.Second

// Message keys on the multi-stream-input protocol.
const (
	keyText         = "text"
	keyContextID    = "context_id"
	keyCloseContext = "close_context"
	keyCloseSocket  = "close_socket"
	keyFlush        = "flush"
)

// errRealtimeTTS wraps an error reading or writing the stream.
//
//nolint:gochecknoglobals // sentinel error
var errRealtimeTTS = errors.New("elevenlabs realtime tts")

// errWebsocketUnavailable is returned for a sentence when there is still no
// connection after connecting again.
//
//nolint:gochecknoglobals // sentinel error
var errWebsocketUnavailable = errors.New("elevenlabs realtime tts: websocket unavailable")

// errNotConnected is returned when there is no connection to use.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("elevenlabs realtime tts: websocket not connected")

// RealtimeTTSConfig configures the streaming WebSocket TTS service.
type RealtimeTTSConfig struct {
	// APIKey is the ElevenLabs API key, sent as the xi-api-key header. Required.
	APIKey string `validate:"required"`
	// Host overrides the WebSocket host; empty uses the hosted API.
	Host string
	// VoiceID is the ElevenLabs voice; empty uses a default public voice.
	VoiceID string
	// Model is the ElevenLabs model; empty uses the low-latency flash model.
	Model string
	// SampleRate is the PCM rate requested and emitted downstream. Empty uses
	// 48 kHz. Must be a rate ElevenLabs supports.
	SampleRate int
	// Language for multilingual models; the zero value leaves it unset.
	Language language.Language
	// VoiceSettings overrides the voice's default settings when non-nil.
	VoiceSettings *VoiceSettings
	// AutoMode lets ElevenLabs decide when it has enough text to start
	// generating, rather than waiting to be told. nil leaves it on, which is
	// what suits a sentence at a time.
	AutoMode *bool
	// ApplyTextNormalization controls spoken-form normalization ("auto", "on",
	// "off"); empty leaves it unset.
	ApplyTextNormalization string
	// EnableLogging toggles server-side logging; nil leaves it unset.
	EnableLogging *bool
	// EnableSSMLParsing has ElevenLabs parse SSML tags in the text, which is
	// what lets it read <phoneme> tags; nil leaves it unset.
	EnableSSMLParsing *bool
	// PronunciationDictionaryLocators applies the given dictionaries.
	PronunciationDictionaryLocators []PronunciationDictionaryLocator
	// WordTimestamps reports per-word timing alongside the audio, which lets the
	// assistant context record what was actually spoken before an interruption.
	// ElevenLabs times every character, so the words are assembled from those.
	WordTimestamps bool
	// PauseFrameProcessing holds incoming frames after a turn until its audio
	// has played; nil leaves it on. With it off, contexts synthesize
	// concurrently on the one connection (two speak frames pushed at once, or
	// back-to-back turns) and still play in order.
	PauseFrameProcessing *bool
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c RealtimeTTSConfig) Validate() error { return validate.Struct(c) }

func (c RealtimeTTSConfig) withDefaults() RealtimeTTSConfig {
	if c.Host == "" {
		c.Host = defaultRealtimeTTSHost
	}
	if c.VoiceID == "" {
		c.VoiceID = defaultVoiceID
	}
	if c.Model == "" {
		c.Model = defaultModel
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultSampleRate
	}
	if c.AutoMode == nil {
		// Without auto mode the server holds text until it is explicitly
		// flushed, and closing a context discards whatever was never flushed:
		// the synthesis then ends with a final marker and no audio at all.
		// Sentence-at-a-time synthesis has nothing to gain from that buffering,
		// so it is on unless the caller turns it off.
		c.AutoMode = new(true)
	}
	if c.PauseFrameProcessing == nil {
		c.PauseFrameProcessing = new(true)
	}
	return c
}

// NewRealtimeTTS builds an ElevenLabs TTS service over the streaming WebSocket
// API. Unlike NewTTS, which issues one HTTP request per sentence, this holds a
// single connection open for the call and one synthesis context open for each
// turn, so neither a sentence nor a turn pays for setup before its audio starts.
//
// The connection is opened when the pipeline starts. When it drops it is
// restored with backoff, and when that keeps failing the service reports a
// permanent error and becomes unusable. See tts.Websocket.
func NewRealtimeTTS(cfg RealtimeTTSConfig) *tts.Base {
	b, _ := newRealtimeTTS(cfg)
	return b
}

// newRealtimeTTS builds the service and returns its synthesizer alongside it.
func newRealtimeTTS(cfg RealtimeTTSConfig) (*tts.Base, *realtimeSynthesizer) {
	s := newRealtimeSynthesizer(cfg)
	// The multi-stream endpoint serves no Eleven v3 or v4 model: the connection
	// is rejected with a 400 at the handshake, which surfaces as an opaque error
	// that leaves the service unusable. Say so up front, with a pointer to the
	// service that does serve them.
	if model := s.cfg.Model; strings.HasPrefix(model, "eleven_v3") || strings.HasPrefix(model, "eleven_v4") {
		slog.Warn("elevenlabs: this is an Eleven v3/v4 model, which the text-to-speech WebSocket "+
			"endpoint does not serve; use NewTTS for HTTP streaming", "model", model)
	}
	var b *tts.Base
	if s.cfg.WordTimestamps {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the
		// base takes the word-aligned path solely when timings were requested.
		b = tts.New("ElevenLabsRealtimeTTS", &timedRealtimeSynthesizer{realtimeSynthesizer: s})
	} else {
		b = tts.New("ElevenLabsRealtimeTTS", s)
	}
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: *s.cfg.PauseFrameProcessing})
	s.svc = b
	s.ws = tts.NewWebsocket(b, s, s.cfg.Websocket)
	return b, s
}

// newRealtimeSynthesizer builds the synthesizer, with its settings taken from
// cfg.
func newRealtimeSynthesizer(cfg RealtimeTTSConfig) *realtimeSynthesizer {
	s := &realtimeSynthesizer{cfg: cfg.withDefaults()}
	s.live = newRealtimeTTSSettings(s.cfg)
	s.refreshConnectionSettings()
	return s
}

// realtimeService is the part of the TTS service the synthesizer asks about
// the audio contexts, and reports through. tts.Base implements it.
type realtimeService interface {
	ActiveAudioContextID() string
	AudioContextIDs() []string
	StopAllMetrics(ctx context.Context)
	WarnUnhandledUpdatedSettings(ctx context.Context, unhandled []string)
}

// realtimeSynthesizer streams over one connection held open for the call, with
// one synthesis context per turn.
//
// The base's receive loop reads the connection and attributes each message to
// the context id it carries. Sending a sentence is therefore just a write: it
// does not wait for the audio of the sentence before it, so the server
// generates the next while the current is still playing. The audio goes into
// the base's audio context for that id, which is what pushes it downstream in
// order.
type realtimeSynthesizer struct {
	cfg  RealtimeTTSConfig
	host tts.AudioContextHost
	svc  realtimeService
	ws   *tts.Websocket

	// live is the settings store, which the base updates on the frame
	// goroutine. It is read only there; what the connection needs of it is
	// copied into conn* below under mu, since a redial reads those from the
	// receive loop.
	live *RealtimeTTSSettings

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	// mu guards everything below. It is never held across a call into the
	// host or a write.
	mu   sync.Mutex
	conn *realtimeConn
	// connURL is the endpoint the next connection dials, and spaceless whether
	// the language set is written without spaces between words.
	connURL   string
	spaceless bool
	// contextInitSent holds the contexts whose opening message has gone out on
	// this connection, which are the only ones a keepalive may name.
	contextInitSent map[string]bool
	// alignmentStarted holds the contexts that have been given an alignment
	// yet, which is what makes the next one utterance-leading or not.
	alignmentStarted map[string]bool
	// wordClocks holds each context's word timing state. The server
	// interleaves the alignment of concurrent contexts on one connection.
	wordClocks map[string]*wordClock
}

// timedRealtimeSynthesizer adds word-timestamp streaming on top of
// realtimeSynthesizer. It implements tts.WordTimestamps.
type timedRealtimeSynthesizer struct {
	*realtimeSynthesizer
}

// wordClock is the word timing state of one context's alignment stream.
type wordClock struct {
	// acc carries a word split across alignment messages until it ends.
	acc uctx.CharAccumulator
	// cumulative is how much of this context's audio earlier messages accounted
	// for, in seconds. ElevenLabs times each message from the start of the audio
	// it carries, so the offsets have to be rebased onto the context's timeline;
	// without that every message after the first reports its words as starting
	// near zero.
	cumulative float64
}

// realtimeConn is one connection, with the goroutine reading it.
//
// The reading runs apart from the receive loop so stopping the loop does not
// close the connection: the service still has to ask the server to close it,
// and wait for it to, which a read cut short would make impossible.
type realtimeConn struct {
	conn *wsutil.Conn
	// reads carries each message read, and finally the error that ended the
	// reading.
	reads chan realtimeRead
	// stop is closed when the connection is being closed, which ends the reading
	// goroutine once nothing takes its messages any more.
	stop     chan struct{}
	stopOnce sync.Once
	// readEnded is set once the reading has failed, which is the connection
	// closing as far as the service is concerned.
	readEnded atomic.Bool
}

// realtimeRead is one message read from the connection, or the error that ended
// the reading.
type realtimeRead struct {
	data []byte
	err  error
}

func newRealtimeConn(conn *wsutil.Conn) *realtimeConn {
	c := &realtimeConn{conn: conn, reads: make(chan realtimeRead), stop: make(chan struct{})}
	go c.read()
	return c
}

// read reads the connection until it ends, handing each message to whoever
// receives from reads.
func (c *realtimeConn) read() {
	for {
		_, data, err := c.conn.Read(context.Background())
		if err != nil {
			c.readEnded.Store(true)
		}
		select {
		case c.reads <- realtimeRead{data: data, err: err}:
		case <-c.stop:
			return
		}
		if err != nil {
			return
		}
	}
}

// halt stops handing messages on, for a connection about to be closed.
func (c *realtimeConn) halt() { c.stopOnce.Do(func() { close(c.stop) }) }

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *realtimeSynthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *realtimeSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the ElevenLabs model and voice synthesis is billed against.
func (s *realtimeSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.VoiceID}
}

// model is the model the settings name.
func (s *realtimeSynthesizer) model() string { return s.live.Model.Or(s.cfg.Model) }

// refreshConnectionSettings copies what the connection needs from the settings
// store. It runs on the frame goroutine, which is where the store is written.
func (s *realtimeSynthesizer) refreshConnectionSettings() {
	endpoint := s.endpoint()
	spaceless := wordTimestampsIncludeInterFrameSpaces(s.live.Language.Or(""))
	s.mu.Lock()
	s.connURL, s.spaceless = endpoint, spaceless
	s.mu.Unlock()
}

// wordTimestampsIncludeInterFrameSpaces reports whether the language is written
// without spaces between words. Chinese (Cantonese included) and Japanese are,
// so the tokens this provider times already read as continuous text and a
// consumer joining them must add no spacing of its own.
func wordTimestampsIncludeInterFrameSpaces(lang string) bool {
	base, _, _ := strings.Cut(lang, "-")
	switch strings.ToLower(base) {
	case "zh", "yue", "ja":
		return true
	default:
		return false
	}
}

// endpoint builds the multi-stream-input URL. The voice, model and output format
// are fixed for the connection; only the text varies per message.
func (s *realtimeSynthesizer) endpoint() string {
	q := url.Values{}
	q.Set("model_id", s.model())
	q.Set("output_format", outputFormat(s.cfg.SampleRate))
	if s.cfg.AutoMode != nil {
		q.Set("auto_mode", strconv.FormatBool(*s.cfg.AutoMode))
	}
	if s.cfg.EnableSSMLParsing != nil {
		q.Set("enable_ssml_parsing", strconv.FormatBool(*s.cfg.EnableSSMLParsing))
	}
	if s.cfg.EnableLogging != nil {
		q.Set("enable_logging", strconv.FormatBool(*s.cfg.EnableLogging))
	}
	if normalization, ok := s.live.ApplyTextNormalization.Value(); ok && normalization != "" {
		q.Set("apply_text_normalization", normalization)
	}
	if code := elevenlabsLanguage(language.Language(s.live.Language.Or(""))); code != "" {
		q.Set("language_code", code)
	}
	path := fmt.Sprintf(realtimeTTSPath, s.live.Voice.Or(s.cfg.VoiceID))
	return s.cfg.Host + path + "?" + q.Encode()
}

// Start connects when the pipeline starts, implementing tts.Starter. The
// handshake to the vendor is the slowest part of a call's first sentence, so
// leaving it to the first sentence puts it in front of the bot's opening words.
func (s *realtimeSynthesizer) Start(ctx context.Context) {
	s.refreshConnectionSettings()
	s.Connect(ctx)
}

// Connect opens the connection, then starts the receive loop and the keepalive,
// implementing tts.WebsocketService.
func (s *realtimeSynthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
		s.ws.CreateKeepaliveTask(ctx, keepaliveInterval, s)
	}
}

// Disconnect stops the receive loop and the keepalive and closes the
// connection, implementing tts.WebsocketService.
func (s *realtimeSynthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	s.ws.CancelKeepaliveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens the connection unless one is open, implementing
// tts.WebsocketHandler. A failure is reported here and leaves no connection
// behind, which a redial reads as a failed attempt.
func (s *realtimeSynthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to elevenlabs realtime tts")
	s.mu.Lock()
	endpoint := s.connURL
	s.mu.Unlock()
	header := http.Header{}
	header.Set("xi-api-key", s.cfg.APIKey)
	conn, err := s.ws.Dial(ctx, endpoint, header, wsutil.DefaultReadLimit)
	if err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
		ef.Err = err
		s.ws.ReportError(ctx, ef, false)
		return nil
	}
	s.mu.Lock()
	old := s.conn
	s.conn = newRealtimeConn(conn)
	s.mu.Unlock()
	if old != nil {
		// A connection whose reading had already ended, replaced rather than
		// closed: nothing is left to say to the server on it.
		old.halt()
		_ = old.conn.CloseNow()
	}
	return nil
}

// DisconnectWebsocket closes the connection, implementing tts.WebsocketHandler.
//
// The protocol tears down in two steps: the service asks ElevenLabs to close,
// then ElevenLabs closes. The close is sent after that request and waits for
// the server's own close, up to the close timeout, so the closing handshake is
// not raced. The server having closed first is the ordinary end of a session
// and is not an error.
func (s *realtimeSynthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.svc != nil {
		s.svc.StopAllMetrics(ctx)
	}
	s.mu.Lock()
	c := s.conn
	s.conn = nil
	s.mu.Unlock()
	if c != nil {
		slog.Debug("disconnecting from elevenlabs realtime tts")
		c.halt()
		err := s.writeTo(ctx, c, map[string]any{keyCloseSocket: true})
		switch {
		case err == nil, closedConnection(err):
			if err != nil {
				slog.Debug("elevenlabs realtime tts websocket already closed during disconnect", "err", err)
			}
		case s.host != nil:
			s.host.PushError(ctx, "Unknown error occurred", err, false)
		}
		_ = c.conn.Close(websocket.StatusNormalClosure, "")
		slog.Debug("disconnected from elevenlabs realtime tts")
	}
	s.clearConnectionState()
	return nil
}

// closedConnection reports whether err is a send failing because the
// connection was already closed.
func closedConnection(err error) bool {
	return websocket.CloseStatus(err) != -1 || errors.Is(err, net.ErrClosed)
}

// clearConnectionState drops what the service knew of the contexts on a
// connection that has closed.
func (s *realtimeSynthesizer) clearConnectionState() {
	s.mu.Lock()
	defer s.mu.Unlock()
	clear(s.contextInitSent)
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *realtimeSynthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.conn.readEnded.Load()
}

// ReceiveMessages handles the connection's messages until it ends, implementing
// tts.WebsocketHandler.
func (s *realtimeSynthesizer) ReceiveMessages(ctx context.Context) error {
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

// handleMessage handles one server message.
//
// A message for a context that is no longer open is dropped: an interruption
// abandons a turn mid-flight, and audio the server had already generated for it
// arrives afterwards. Attributing by context id keeps that audio out of the next
// turn, which is what the ids are for.
func (s *realtimeSynthesizer) handleMessage(data []byte) error {
	var msg realtimeTTSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return fmt.Errorf("%w: decode message: %w", errRealtimeTTS, err)
	}
	if msg.Error != "" {
		slog.Warn("elevenlabs realtime tts error", "context", msg.ContextID, "error", msg.Error)
	}
	id := msg.ContextID

	// The final message is handled first, whatever else it carries. An
	// interrupted context is no longer open, and there is nothing to do for it.
	if msg.IsFinal {
		if s.host != nil && s.host.AudioContextAvailable(id) {
			s.finishContext(id)
		}
		return nil
	}

	if msg.Audio != "" {
		pcm, err := base64.StdEncoding.DecodeString(msg.Audio)
		if err != nil {
			return fmt.Errorf("%w: decode audio: %w", errRealtimeTTS, err)
		}
		if s.host != nil {
			f := frames.NewTTSAudioRawFrame(pcm, s.cfg.SampleRate, 1)
			f.ContextID = id
			s.host.AppendToAudioContext(id, f)
		}
	}

	// The alignment of an interrupted context keeps arriving after its clock was
	// dropped; handling it would start a clock nothing removes.
	if s.host == nil || !s.host.AudioContextAvailable(id) || !s.cfg.WordTimestamps {
		return nil
	}
	alignment := selectRealtimeAlignment(&msg, len(s.cfg.PronunciationDictionaryLocators) > 0)
	if alignment == nil {
		return nil
	}
	s.mu.Lock()
	started := s.alignmentStarted[id]
	if s.alignmentStarted == nil {
		s.alignmentStarted = map[string]bool{}
	}
	s.alignmentStarted[id] = true
	spaceless := s.spaceless
	s.mu.Unlock()
	words := s.wordTimesFromAlignment(id, alignment.stripLeadingSpaces(!started))
	if len(words) > 0 {
		s.host.AddWordTimestamps(id, words, tts.WordTimingOptions{IncludesInterFrameSpaces: spaceless})
	}
	return nil
}

// finishContext closes a context out once the server has marked it final.
func (s *realtimeSynthesizer) finishContext(id string) {
	if s.cfg.WordTimestamps {
		// The closing word ends on the utterance, not on a space.
		s.mu.Lock()
		var last uctx.WordTiming
		var ok bool
		if clock := s.wordClocks[id]; clock != nil {
			last, ok = clock.acc.Flush()
		}
		spaceless := s.spaceless
		s.mu.Unlock()
		if ok {
			s.host.AddWordTimestamps(id, []uctx.WordTiming{last}, tts.WordTimingOptions{IncludesInterFrameSpaces: spaceless})
		}
	}
	// The stop frame rides the queue behind the audio, so it is pushed only once
	// the last of that audio has been.
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = id
	s.host.AppendToAudioContext(id, stopped)
	s.host.RemoveAudioContext(id)
}

// wordTimesFromAlignment folds one alignment message into the words it
// completes, on its context's own clock.
func (s *realtimeSynthesizer) wordTimesFromAlignment(id string, a *charAlignment) []uctx.WordTiming {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.wordClocks == nil {
		s.wordClocks = map[string]*wordClock{}
	}
	clock := s.wordClocks[id]
	if clock == nil {
		clock = &wordClock{}
		s.wordClocks[id] = clock
	}
	starts := make([]float64, len(a.CharStartTimesMs))
	for i, ms := range a.CharStartTimesMs {
		starts[i] = clock.cumulative + ms/1000
	}
	words, err := clock.acc.Add(a.Chars, starts)
	if err != nil {
		// The message cannot be timed, but its span still counts below.
		slog.Error("elevenlabs realtime tts: character timings do not line up", "err", err)
	}
	// Alignment times restart at zero in every message, so the clock advances by
	// each message's span. It has to advance for messages that complete no word
	// too (trailing punctuation arrives on its own), or every later word is
	// timestamped early by the shortfall.
	starts, durations := a.CharStartTimesMs, a.CharDurationsMs
	switch {
	case len(starts) > 0 && len(durations) > 0:
		clock.cumulative += (starts[len(starts)-1] + durations[len(durations)-1]) / 1000
	case len(words) > 0:
		clock.cumulative = words[len(words)-1].Offset
	}
	return words
}

// resetWordClock drops a context's word timing state.
func (s *realtimeSynthesizer) resetWordClock(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.wordClocks, id)
}

// resetAlignmentState drops everything the service tracks for a context.
func (s *realtimeSynthesizer) resetAlignmentState(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.wordClocks, id)
	delete(s.alignmentStarted, id)
	delete(s.contextInitSent, id)
}

// KeepaliveReady reports whether the connection is open, implementing
// tts.KeepaliveHandler.
func (s *realtimeSynthesizer) KeepaliveReady() bool { return s.Connected() }

// SendKeepalive sends empty text, implementing tts.KeepaliveHandler.
//
// It names the active context only once that context's opening message, which
// carries the voice settings, has gone out. Otherwise the keepalive would be
// the context's first message, with no voice settings, and ElevenLabs would
// reject the opening message that follows. A keepalive naming no context keeps
// the connection open just as well until then.
func (s *realtimeSynthesizer) SendKeepalive(ctx context.Context) error {
	if !s.Connected() {
		return nil
	}
	msg := map[string]any{keyText: ""}
	if id := s.svc.ActiveAudioContextID(); id != "" {
		s.mu.Lock()
		sent := s.contextInitSent[id]
		s.mu.Unlock()
		if sent {
			msg[keyContextID] = id
		}
	}
	return s.send(ctx, msg)
}

// send writes one JSON message on the open connection.
func (s *realtimeSynthesizer) send(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errNotConnected
	}
	return s.writeTo(ctx, c, msg)
}

// writeTo writes one JSON message on c.
func (s *realtimeSynthesizer) writeTo(ctx context.Context, c *realtimeConn, msg map[string]any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.conn.Write(ctx, websocket.MessageText, payload)
}

// RunTTS sends one sentence on the turn's context, opening that context on the
// wire the first time it is used. It yields nothing: the call returns once the
// text is sent, and the audio arrives on the receive loop, which appends it to
// the context it belongs to.
//
// A connection that has closed is opened again first. A send that fails ends
// the context's synthesis with a stop frame and is reported.
func (s *realtimeSynthesizer) RunTTS(
	ctx context.Context, text, contextID string, yield func(f frames.Frame) error,
) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	if !s.Connected() {
		slog.Warn("elevenlabs realtime tts: websocket unavailable after reconnect, skipping TTS")
		return errWebsocketUnavailable
	}
	if err := s.sendText(ctx, text, contextID); err != nil {
		if yield != nil {
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = contextID
			_ = yield(stopped)
		}
		return fmt.Errorf("%w: %w", errRealtimeTTS, err)
	}
	return nil
}

// sendText sends text on the context, opening the context first if its opening
// message has not gone out on this connection.
func (s *realtimeSynthesizer) sendText(ctx context.Context, text, contextID string) error {
	s.mu.Lock()
	opened := s.contextInitSent[contextID]
	s.mu.Unlock()
	if !opened {
		s.resetWordClock(contextID)
		if err := s.sendContextInit(ctx, contextID); err != nil {
			return err
		}
	}
	if contextID != "" {
		if err := s.send(ctx, map[string]any{keyText: text, keyContextID: contextID}); err != nil {
			return err
		}
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// sendContextInit opens a context. The opening message carries the voice
// settings and the pronunciation dictionaries, and a single space rather than
// the first sentence: it initializes the context without committing text.
func (s *realtimeSynthesizer) sendContextInit(ctx context.Context, contextID string) error {
	// Recorded before the send, so the keepalive may name the context from now
	// on.
	s.mu.Lock()
	if s.contextInitSent == nil {
		s.contextInitSent = map[string]bool{}
	}
	s.contextInitSent[contextID] = true
	s.mu.Unlock()
	msg := map[string]any{keyText: " ", keyContextID: contextID}
	if v := s.live.voiceSettings(); v != nil {
		msg["voice_settings"] = v
	}
	if len(s.cfg.PronunciationDictionaryLocators) > 0 {
		msg["pronunciation_dictionary_locators"] = s.cfg.PronunciationDictionaryLocators
	}
	return s.send(ctx, msg)
}

// FlushAudio tells the server to generate whatever text it is still holding for
// the context, or for the active one when none is named, implementing
// tts.AudioFlusher.
func (s *realtimeSynthesizer) FlushAudio(ctx context.Context, contextID string) {
	if contextID == "" {
		contextID = s.svc.ActiveAudioContextID()
	}
	if contextID == "" || !s.Connected() {
		return
	}
	_ = s.send(ctx, map[string]any{keyContextID: contextID, keyFlush: true})
}

// OnTurnContextCompleted closes the server-side context once the turn's text has
// all been sent, implementing tts.TurnContextCompleter. Closing is what makes the
// final marker arrive immediately after the last audio byte rather than after the
// server waits to see whether more text is coming.
func (s *realtimeSynthesizer) OnTurnContextCompleted(ctx context.Context, contextID string) {
	s.closeContext(ctx, contextID)
}

// OnAudioContextInterrupted stops the server generating into a context nobody is
// listening to, implementing tts.AudioContextInterrupter.
func (s *realtimeSynthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	s.closeContext(context.WithoutCancel(ctx), contextID)
	s.resetAlignmentState(contextID)
}

// OnAudioContextCompleted releases the state kept for a context that has
// finished playing, implementing tts.AudioContextCompleter.
func (s *realtimeSynthesizer) OnAudioContextCompleted(_ context.Context, contextID string) {
	s.resetAlignmentState(contextID)
}

// closeContext closes a server-side context to free it. ElevenLabs leaves
// closing contexts to the client, both on an interruption and once a turn is
// done.
func (s *realtimeSynthesizer) closeContext(ctx context.Context, contextID string) {
	if contextID == "" {
		return
	}
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return
	}
	err := s.writeTo(ctx, c, map[string]any{keyContextID: contextID, keyCloseContext: true})
	if err != nil && s.host != nil {
		s.host.PushError(ctx, "Unknown error occurred", err, false)
	}
}

// charAlignment times each character of a message's audio, relative to the start
// of that audio rather than of the context.
type charAlignment struct {
	Chars            []string  `json:"chars"`
	CharStartTimesMs []float64 `json:"charStartTimesMs"` //nolint:tagliatelle // ElevenLabs wire keys are camelCase
	CharDurationsMs  []float64 `json:"charDurationsMs"`  //nolint:tagliatelle // ElevenLabs wire keys are camelCase
}

// stripLeadingSpaces drops the spaces this alignment opens with, but only when
// it is the first of an utterance. It is the WebSocket schema's counterpart to
// the same trim on the HTTP one, which names its three parallel arrays
// differently.
func (a *charAlignment) stripLeadingSpaces(shouldStrip bool) *charAlignment {
	if !shouldStrip || len(a.Chars) == 0 || a.Chars[0] != " " {
		return a
	}
	n := 0
	for n < len(a.Chars) && a.Chars[n] == " " {
		n++
	}
	return &charAlignment{
		Chars:            a.Chars[n:],
		CharStartTimesMs: cutFrom(a.CharStartTimesMs, n),
		CharDurationsMs:  cutFrom(a.CharDurationsMs, n),
	}
}

// selectRealtimeAlignment picks which of a message's two alignments to time the
// words with, the WebSocket schema's counterpart to selectAlignment. The
// normalized one is preferred only with a pronunciation dictionary, and either
// falls back to the other when it is missing.
func selectRealtimeAlignment(m *realtimeTTSMessage, preferNormalized bool) *charAlignment {
	if preferNormalized {
		if m.NormalizedAlignment != nil {
			return m.NormalizedAlignment
		}
		return m.Alignment
	}
	if m.Alignment != nil {
		return m.Alignment
	}
	return m.NormalizedAlignment
}

// realtimeTTSMessage is one server message. Audio arrives base64-encoded, and
// alignment times every character rather than every word.
//
//nolint:tagliatelle // ElevenLabs wire keys are camelCase
type realtimeTTSMessage struct {
	ContextID           string         `json:"contextId"`
	Audio               string         `json:"audio"`
	IsFinal             bool           `json:"isFinal"`
	Alignment           *charAlignment `json:"alignment"`
	NormalizedAlignment *charAlignment `json:"normalizedAlignment"`
	Error               string         `json:"error"`
}

var (
	_ tts.Synthesizer             = (*realtimeSynthesizer)(nil)
	_ tts.ContextSynthesizer      = (*realtimeSynthesizer)(nil)
	_ tts.WebsocketService        = (*realtimeSynthesizer)(nil)
	_ tts.KeepaliveHandler        = (*realtimeSynthesizer)(nil)
	_ tts.Starter                 = (*realtimeSynthesizer)(nil)
	_ tts.AudioFlusher            = (*realtimeSynthesizer)(nil)
	_ tts.TurnContextCompleter    = (*realtimeSynthesizer)(nil)
	_ tts.AudioContextInterrupter = (*realtimeSynthesizer)(nil)
	_ tts.AudioContextCompleter   = (*realtimeSynthesizer)(nil)
	_ tts.WordTimestamps          = (*timedRealtimeSynthesizer)(nil)
	_ tts.ContextSynthesizer      = (*timedRealtimeSynthesizer)(nil)
)

// RunTTSTimed sends the sentence like RunTTS, implementing tts.WordTimestamps.
// The timings arrive on the receive loop with the audio, so neither callback is
// used here.
func (s *timedRealtimeSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	yield func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, yield)
}
