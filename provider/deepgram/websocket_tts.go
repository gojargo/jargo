package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// defaultWebsocketTTSBaseURL is the host the Aura WebSocket is opened on.
	defaultWebsocketTTSBaseURL = "wss://" + defaultHost
	// defaultWebsocketTTSVoice is the Aura voice spoken when none is set.
	defaultWebsocketTTSVoice = "aura-2-helena-en"
	// websocketTTSReadLimit bounds a single incoming message. Audio frames can
	// be larger than the library's default, so the ceiling is raised.
	websocketTTSReadLimit = 1 << 24
)

// The Aura WebSocket messages the service sends and reads.
const (
	auraMsgSpeak    = "Speak"
	auraMsgFlush    = "Flush"
	auraMsgClear    = "Clear"
	auraMsgClose    = "Close"
	auraMsgMetadata = "Metadata"
	auraMsgFlushed  = "Flushed"
	auraMsgCleared  = "Cleared"
	auraMsgWarning  = "Warning"
)

// websocketTTSEncodings lists the audio encodings the Aura WebSocket accepts.
func websocketTTSEncodings() []string { return []string{"linear16", "mulaw", "alaw"} }

// errUnsupportedEncoding is returned when the configured encoding is not one
// the Aura WebSocket accepts.
//
//nolint:gochecknoglobals // sentinel error
var errUnsupportedEncoding = errors.New("deepgram tts: unsupported encoding")

// errWebsocketNotConnected is returned when there is no connection to use.
//
//nolint:gochecknoglobals // sentinel error
var errWebsocketNotConnected = errors.New("websocket not connected")

// WebsocketTTSConfig configures the Aura TTS service over the WebSocket API.
type WebsocketTTSConfig struct {
	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the WebSocket base, scheme included; empty uses
	// "wss://api.deepgram.com".
	BaseURL string
	// Voice is the Aura voice model (e.g. "aura-2-helena-en"); empty uses
	// "aura-2-helena-en". Deepgram takes it as its "model" query parameter, so
	// it is also the model the synthesis is reported against.
	Voice string
	// SampleRate is the audio rate requested from Aura and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Encoding is the audio encoding: "linear16", "mulaw" or "alaw". Empty uses
	// "linear16".
	Encoding string
	// Speed multiplies the speaking rate, from 0.7 to 1.5. Nil leaves Aura at
	// its default rate. Supported by the Aura-2 English and Spanish voices;
	// Deepgram recommends staying at or above 0.9 for Spanish.
	Speed *float64
	// MipOptOut opts out of the Deepgram Model Improvement Program. See
	// https://dpgr.am/deepgram-mip for pricing impacts before setting it to
	// true. Nil leaves it unset.
	MipOptOut *bool
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c WebsocketTTSConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.Encoding == "" || slices.Contains(websocketTTSEncodings(), strings.ToLower(c.Encoding)) {
		return nil
	}
	return fmt.Errorf("%w %q: must be one of %s for WebSocket TTS",
		errUnsupportedEncoding, c.Encoding, strings.Join(websocketTTSEncodings(), ", "))
}

// withDefaults returns a copy of the config with empty fields filled in.
func (c WebsocketTTSConfig) withDefaults() WebsocketTTSConfig {
	if c.BaseURL == "" {
		c.BaseURL = defaultWebsocketTTSBaseURL
	}
	if c.Voice == "" {
		c.Voice = defaultWebsocketTTSVoice
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSSampleRate
	}
	if c.Encoding == "" {
		c.Encoding = defaultTTSEncoding
	}
	return c
}

// TTSSettings is the part of the Aura synthesis configuration that can change
// while the pipeline runs.
//
// Voice is the Aura voice model. Deepgram takes it as its "model" query
// parameter, so Model follows Voice: it is what the synthesis is reported and
// priced against. Speed multiplies the speaking rate; no value leaves Aura at
// its default rate.
//
// Every setting is a query parameter of the connection, so a change to any of
// them reconnects.
type TTSSettings struct {
	settings.TTS

	// Speed multiplies the speaking rate, from 0.7 to 1.5.
	Speed settings.Opt[float64] `settings:"speed"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with, once its defaults are applied.
func newTTSSettings(cfg WebsocketTTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Voice = settings.Set(cfg.Voice)
	s.Model = settings.Set(cfg.Voice)
	s.Language = settings.Cleared[string]()
	s.Speed = settings.Cleared[float64]()
	if cfg.Speed != nil {
		s.Speed = settings.Set(*cfg.Speed)
	}
	return s
}

// ApplyUpdate merges an update and keeps the model in step with the voice,
// implementing settings.Updater. Deepgram takes the voice as the model, so a
// voice change is also reported as a model change, which relabels the
// synthesis the service reports from then on.
func (s *TTSSettings) ApplyUpdate(delta any) (settings.Changed, error) {
	changed, err := settings.ApplyFields(s, delta)
	if err != nil {
		return nil, err
	}
	if !changed.Has("voice") {
		return changed, nil
	}
	old := s.Model.AsAny()
	s.Model = s.Voice
	if _, had := changed["model"]; !had && old != s.Model.AsAny() {
		changed["model"] = old
	}
	return changed, nil
}

// NewWebsocketTTS builds a Deepgram Aura TTS service over the WebSocket API.
// Unlike NewTTS, which issues one HTTP request per sentence, it holds one
// connection open for the call and streams each sentence on it, so no sentence
// pays for a connection before its audio starts.
//
// The connection is opened when the pipeline starts. When it drops it is
// restored with backoff, and when that keeps failing the service reports a
// permanent error and becomes unusable. See tts.NewWebsocket.
//
// Aura's audio names no request: it is labeled with the context playing, and
// frame handling pauses while a turn's audio is generated so turns never
// overlap. The text of a turn is generated once the turn has all been sent,
// and an interruption tells Aura to clear what it holds.
func NewWebsocketTTS(cfg WebsocketTTSConfig) *tts.Base {
	b, _ := newWebsocketTTS(cfg)
	return b
}

// newWebsocketTTS builds the service and returns its synthesizer alongside it.
func newWebsocketTTS(cfg WebsocketTTSConfig) (*tts.Base, *wsSynth) {
	cfg = cfg.withDefaults()
	s := &wsSynth{cfg: cfg, live: newTTSSettings(cfg)}
	s.refreshEndpoint()
	b := tts.New("DeepgramWebsocketTTS", s)
	s.base = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s
}

// wsSynth speaks over one Aura WebSocket held open for the call. A sentence is
// sent and the call returns; the audio arrives on the receive loop.
type wsSynth struct {
	cfg WebsocketTTSConfig
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every connection is opened with.
	live *TTSSettings

	// base is the service this synthesizes for, and host the audio contexts
	// its audio is appended to.
	base *tts.Base
	host tts.AudioContextHost
	// ws is the connection lifecycle: it runs the receive loop, restores a lost
	// connection with backoff, and reports what fails.
	ws *tts.Websocket

	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex
	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	// mu guards the connection. It is never held across a call into the host:
	// appending blocks while the audio plays out.
	mu   sync.Mutex
	conn *auraConn
	// endpoint is the URL a connection is dialed on, built from live on the
	// frame goroutine, where the settings are written, and read under mu,
	// since a redial dials it from the receive loop.
	endpoint string
}

// auraConn is one connection, with the goroutine reading it.
//
// The reading runs apart from the receive loop so stopping the loop does not
// close the connection: closing it still sends Aura a Close and waits for the
// server to answer.
type auraConn struct {
	conn *wsutil.Conn
	// reads carries each message read, and finally the error that ended the
	// reading.
	reads chan auraRead
	// stop is closed when the connection is being closed, which ends the reading
	// goroutine once nothing takes its messages any more.
	stop     chan struct{}
	stopOnce sync.Once
	// readEnded is set once the reading has failed, which is the connection
	// closing as far as the service is concerned.
	readEnded atomic.Bool
}

// auraRead is one message read from the connection, or the error that ended
// the reading.
type auraRead struct {
	typ  websocket.MessageType
	data []byte
	err  error
}

func newAuraConn(conn *wsutil.Conn) *auraConn {
	c := &auraConn{conn: conn, reads: make(chan auraRead), stop: make(chan struct{})}
	go c.read()
	return c
}

// read reads the connection until it ends, handing each message to whoever
// receives from reads.
func (c *auraConn) read() {
	for {
		typ, data, err := c.conn.Read(context.Background())
		if err != nil {
			c.readEnded.Store(true)
		}
		select {
		case c.reads <- auraRead{typ: typ, data: data, err: err}:
		case <-c.stop:
			return
		}
		if err != nil {
			return
		}
	}
}

// halt stops handing messages on, for a connection about to be closed.
func (c *auraConn) halt() { c.stopOnce.Do(func() { close(c.stop) }) }

// drop closes the connection outright, for one with nothing left to say.
func (c *auraConn) drop() {
	c.halt()
	_ = c.conn.CloseNow()
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *wsSynth) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested output rate.
func (s *wsSynth) SampleRate() int { return s.cfg.SampleRate }

// RequiresTrailingSpace reports that Aura reads the punctuation ending a
// sentence aloud, saying "dot" for a full stop, unless the text ends in a space.
func (s *wsSynth) RequiresTrailingSpace() bool { return true }

// Metadata reports the voice synthesis is billed against, which Deepgram takes
// as the model.
func (s *wsSynth) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(s.cfg.Voice), VoiceID: s.live.Voice.Or(s.cfg.Voice)}
}

// FormatPronunciation implements tts.PronunciationFormatter. See
// FormatPronunciation.
func (s *wsSynth) FormatPronunciation(word, ipa string) (string, bool) {
	return FormatPronunciation(word, ipa)
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *wsSynth) Settings() any { return s.live }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. Every
// setting is a query parameter of the connection, so any change reconnects.
func (s *wsSynth) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.refreshEndpoint()
	s.Disconnect(ctx)
	s.Connect(ctx)
	return nil
}

// refreshEndpoint builds the URL a connection is dialed on from the settings as
// they stand. It runs on the frame goroutine, which is where they are written.
func (s *wsSynth) refreshEndpoint() {
	endpoint := s.cfg.BaseURL + speakPath + "?" + websocketTTSQuery(s.cfg, s.live).Encode()
	s.mu.Lock()
	s.endpoint = endpoint
	s.mu.Unlock()
}

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *wsSynth) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect readies a fresh connection, opens it, and starts the receive loop on
// it, implementing tts.WebsocketService.
func (s *wsSynth) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect stops the receive loop and closes the connection, implementing
// tts.WebsocketService.
func (s *wsSynth) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// websocketTTSQuery builds the synthesis query string, from the settings as
// they stand rather than as the service was built: a connection opened after a
// change is where that change takes effect.
func websocketTTSQuery(cfg WebsocketTTSConfig, live *TTSSettings) url.Values {
	q := url.Values{}
	q.Set("model", live.Voice.Or(cfg.Voice))
	q.Set("encoding", cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(cfg.SampleRate))
	if v, ok := live.Speed.Value(); ok {
		q.Set("speed", strconv.FormatFloat(v, 'f', -1, 64))
	}
	query.SetBoolOpt(q, "mip_opt_out", cfg.MipOptOut)
	return q
}

// ConnectWebsocket opens the connection with the settings as they stand,
// unless one is open, implementing tts.WebsocketHandler. A failure is reported
// here, with Deepgram's reason when it refused the handshake, and leaves no
// connection behind, which the receive loop reads as a failed attempt.
func (s *wsSynth) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to deepgram websocket tts")

	header := http.Header{}
	header.Set("Authorization", authToken(s.cfg.APIKey))
	s.mu.Lock()
	endpoint := s.endpoint
	s.mu.Unlock()
	conn, err := s.ws.Dial(ctx, endpoint, header, websocketTTSReadLimit)
	if err != nil {
		s.mu.Lock()
		old := s.conn
		s.conn = nil
		s.mu.Unlock()
		if old != nil {
			old.drop()
		}
		msg := describeRefusal(err)
		if msg == "" {
			msg = fmt.Sprintf("%s error: %v", s.base.Name(), err)
		}
		ef := frames.NewErrorFrame(msg)
		ef.Err = err
		s.ws.ReportError(ctx, ef, false)
		return nil
	}
	s.mu.Lock()
	old := s.conn
	s.conn = newAuraConn(conn)
	s.mu.Unlock()
	if old != nil {
		// A connection whose reading had already ended, replaced rather than
		// closed: nothing is left to say to the server on it.
		old.drop()
	}
	return nil
}

// DisconnectWebsocket closes the connection, implementing
// tts.WebsocketHandler. What the service was timing is reported as far as it
// got first: the audio it was waiting for will not come on this connection.
// Aura is sent a Close message first, which ends the session gracefully.
func (s *wsSynth) DisconnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	s.base.StopAllMetrics(ctx)
	s.mu.Lock()
	c := s.conn
	s.conn = nil
	s.mu.Unlock()
	if c == nil {
		return nil
	}
	if c.readEnded.Load() {
		// The connection is already gone: there is no one left to send the
		// Close to.
		c.drop()
		return nil
	}
	slog.Debug("disconnecting from deepgram websocket tts", "service", s.base.Name())
	c.halt()
	err := s.write(ctx, c.conn, wsMessage{Type: auraMsgClose})
	if err == nil {
		err = c.conn.Close(websocket.StatusNormalClosure, "")
	} else {
		_ = c.conn.CloseNow()
	}
	if err != nil {
		slog.Error("deepgram websocket tts: disconnecting failed", "service", s.base.Name(), "err", err)
		s.host.PushError(ctx, fmt.Sprintf("%s error: %v", s.base.Name(), err), err, false)
	}
	return nil
}

// Connected reports whether there is a live connection, implementing
// tts.WebsocketHandler.
func (s *wsSynth) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.conn.readEnded.Load()
}

// wsMessage is a message sent to Aura: a Speak carrying text, or a bare Flush,
// Clear or Close.
type wsMessage struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// send sends one message on the open connection, failing with
// errWebsocketNotConnected when there is none.
func (s *wsSynth) send(ctx context.Context, msg wsMessage) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errWebsocketNotConnected
	}
	return s.write(ctx, c.conn, msg)
}

// write writes one JSON message on conn, serialized against the other writers.
func (s *wsSynth) write(ctx context.Context, conn *wsutil.Conn, msg wsMessage) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(ctx, websocket.MessageText, payload)
}

// auraMessage is the subset of an Aura control message read.
type auraMessage struct {
	Type        string `json:"type"`
	WarnMsg     string `json:"warn_msg"`
	Description string `json:"description"`
}

// ReceiveMessages reads and handles messages until the connection ends,
// implementing tts.WebsocketHandler: binary frames are audio, text frames
// metadata and control messages.
func (s *wsSynth) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errWebsocketNotConnected
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case r := <-c.reads:
			if r.err != nil {
				return r.err
			}
			if r.typ == websocket.MessageBinary {
				s.handleAudio(r.data)
				continue
			}
			s.handleMessage(r.data)
		}
	}
}

// handleAudio appends a chunk of audio to the context playing: Aura's audio
// does not say which request it belongs to.
func (s *wsSynth) handleAudio(audio []byte) {
	contextID := s.base.ActiveAudioContextID()
	f := frames.NewTTSAudioRawFrame(audio, s.SampleRate(), 1)
	f.ContextID = contextID
	s.host.AppendToAudioContext(contextID, f)
}

// handleMessage acts on one text message from Aura.
func (s *wsSynth) handleMessage(data []byte) {
	var m auraMessage
	if err := json.Unmarshal(data, &m); err != nil {
		slog.Error("deepgram websocket tts: invalid JSON message", "message", string(data))
		return
	}
	name := s.base.Name()
	switch m.Type {
	case auraMsgMetadata:
		slog.Debug("deepgram websocket tts: received metadata", "service", name, "message", string(data))
	case auraMsgFlushed:
		// Everything sent before the Flush has been generated, so the context
		// playing has all its audio.
		slog.Debug("deepgram websocket tts: received Flushed", "service", name, "message", string(data))
		contextID := s.base.ActiveAudioContextID()
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		s.host.AppendToAudioContext(contextID, stopped)
		s.host.RemoveAudioContext(contextID)
	case auraMsgCleared:
		// The buffer has been cleared after an interruption, which already
		// cleaned up after itself.
		slog.Debug("deepgram websocket tts: received Cleared", "service", name, "message", string(data))
	case auraMsgWarning:
		description := m.WarnMsg
		if description == "" {
			description = m.Description
		}
		if description == "" {
			description = "Unknown warning"
		}
		slog.Warn(fmt.Sprintf("%s warning: %s", name, description))
	default:
		slog.Debug("deepgram websocket tts: received an unknown message type", "message", string(data))
	}
}

// FlushAudio sends a Flush, implementing tts.AudioFlusher. It is sent once the
// turn's text has all been sent, and has Aura generate the audio of what it
// still holds.
func (s *wsSynth) FlushAudio(ctx context.Context, _ string) {
	if !s.Connected() {
		return
	}
	if err := s.send(ctx, wsMessage{Type: auraMsgFlush}); err != nil {
		slog.Error("deepgram websocket tts: sending the Flush message failed", "service", s.base.Name(), "err", err)
	}
}

// OnAudioContextInterrupted sends a Clear when the user cuts in, implementing
// tts.AudioContextInterrupter. Clear empties Aura's text buffer and stops the
// audio it was sending, so a new response can be generated.
func (s *wsSynth) OnAudioContextInterrupted(ctx context.Context, _ string) {
	s.base.StopAllMetrics(ctx)
	if !s.Connected() {
		return
	}
	if err := s.send(context.WithoutCancel(ctx), wsMessage{Type: auraMsgClear}); err != nil {
		slog.Error("deepgram websocket tts: sending the Clear message failed", "service", s.base.Name(), "err", err)
	}
}

// RunTTS sends the text on the connection and returns: the audio arrives on
// the receive loop. A connection that has closed is opened again first. No
// Flush is sent here; that waits until the turn's text has all been sent (see
// FlushAudio). A failure is yielded as an error in the context's stream.
func (s *wsSynth) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	if err := s.send(ctx, wsMessage{Type: auraMsgSpeak, Text: text}); err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
		ef.Err = err
		return yield(ef)
	}
	return nil
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*wsSynth)(nil)
	_ tts.AudioFlusher            = (*wsSynth)(nil)
	_ tts.AudioContextInterrupter = (*wsSynth)(nil)
	_ tts.Starter                 = (*wsSynth)(nil)
	_ tts.Describer               = (*wsSynth)(nil)
	_ tts.TrailingSpaceRequirer   = (*wsSynth)(nil)
	_ tts.PronunciationFormatter  = (*wsSynth)(nil)
	_ tts.SettingsUpdater         = (*wsSynth)(nil)
	_ tts.WebsocketService        = (*wsSynth)(nil)
	_ settings.Updater            = (*TTSSettings)(nil)
)
