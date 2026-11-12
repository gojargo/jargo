package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
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
)

// errTTSNotConnected is returned when there is no connection to use.
//
//nolint:gochecknoglobals // sentinel error
var errTTSNotConnected = errors.New("sarvam tts: websocket not connected")

const (
	defaultTTSURL   = "wss://api.sarvam.ai/text-to-speech/ws"
	defaultTTSModel = "bulbul:v3"
	// deprecatedTTSModel is the previous model, which Sarvam's API now rejects.
	deprecatedTTSModel = "bulbul:v2"
	// defaultTTSLanguage is the Sarvam code used when no language is configured.
	defaultTTSLanguage = "en-IN"
	// Audio format defaults. linear16 yields raw 16-bit mono PCM.
	defaultTTSCodec   = "linear16"
	defaultTTSBitrate = "128k"
	// Buffering defaults for the WebSocket streaming protocol.
	defaultMinBufferSize  = 50
	defaultMaxChunkLength = 150
	// defaultPace is the neutral speaking rate.
	defaultPace = 1.0
	// readLimitTTS bounds a single inbound WebSocket message; audio arrives base64.
	readLimitTTS = 1 << 20
	// ttsKeepaliveInterval is how often an idle connection is pinged, so the
	// server keeps it open through silences.
	ttsKeepaliveInterval = 20 * time.Second
)

// Neutral values of the model-specific controls: a model that does not support
// a control is not warned about one left at its neutral value.
const (
	neutralPitch       = 0.0
	neutralLoudness    = 1.0
	neutralTemperature = 0.6
)

// ttsModelConfig captures the per-model capabilities and defaults that shape the
// synthesis request.
type ttsModelConfig struct {
	supportsPitch              bool
	supportsLoudness           bool
	supportsTemperature        bool
	defaultSampleRate          int
	defaultSpeaker             string
	paceMin, paceMax           float64
	preprocessingAlwaysEnabled bool
}

// ttsModelConfigs describes each supported bulbul model. v2 and v4-flash
// support pitch and loudness; the v3 variants trade those for temperature. All
// but v2 always preprocess.
//
//nolint:gochecknoglobals // static model capability table
var ttsModelConfigs = map[string]ttsModelConfig{
	"bulbul:v2": {
		supportsPitch:              true,
		supportsLoudness:           true,
		supportsTemperature:        false,
		defaultSampleRate:          22050,
		defaultSpeaker:             "anushka",
		paceMin:                    0.3,
		paceMax:                    3.0,
		preprocessingAlwaysEnabled: false,
	},
	"bulbul:v3-beta": {
		supportsPitch:              false,
		supportsLoudness:           false,
		supportsTemperature:        true,
		defaultSampleRate:          24000,
		defaultSpeaker:             "shubh",
		paceMin:                    0.5,
		paceMax:                    2.0,
		preprocessingAlwaysEnabled: true,
	},
	"bulbul:v3": {
		supportsPitch:              false,
		supportsLoudness:           false,
		supportsTemperature:        true,
		defaultSampleRate:          24000,
		defaultSpeaker:             "shubh",
		paceMin:                    0.5,
		paceMax:                    2.0,
		preprocessingAlwaysEnabled: true,
	},
	"bulbul:v4-flash": {
		supportsPitch:              true,
		supportsLoudness:           true,
		supportsTemperature:        false,
		defaultSampleRate:          24000,
		defaultSpeaker:             "shubh_en_narration_gentle",
		paceMin:                    0.5,
		paceMax:                    2.0,
		preprocessingAlwaysEnabled: true,
	},
}

// TTSConfig configures the Sarvam WebSocket TTS service. Optional fields modeled
// as pointers are omitted from the request when unset, and model-specific fields
// are dropped for models that do not support them.
type TTSConfig struct {
	// APIKey is the Sarvam API subscription key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the TTS model; empty uses "bulbul:v3". "bulbul:v4-flash" is also
	// available. "bulbul:v2" is deprecated: Sarvam's API rejects it.
	Model string `validate:"omitempty,oneof=bulbul:v2 bulbul:v3-beta bulbul:v3 bulbul:v4-flash"`
	// Voice is the speaker id; empty uses the model's default speaker.
	Voice string
	// Language for synthesis; the zero value uses English (India). Mapped to
	// Sarvam's regional code.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses the
	// model default (24000 Hz, 22050 Hz for bulbul:v2).
	SampleRate int
	// EnablePreprocessing toggles text preprocessing; nil defaults to off. Always
	// forced on for every model but bulbul:v2.
	EnablePreprocessing *bool
	// Pace multiplies the speaking rate; nil uses 1.0. Clamped to the model's
	// supported range (0.5 to 2.0, 0.3 to 3.0 for bulbul:v2).
	Pace *float64
	// Pitch adjusts voice pitch (-0.75 to 0.75); nil omits it. Only bulbul:v4-flash
	// and bulbul:v2.
	Pitch *float64
	// Loudness multiplies volume (0.3 to 3.0); nil omits it. Only
	// bulbul:v4-flash and bulbul:v2.
	Loudness *float64
	// Temperature controls output randomness (0.01 to 1.0); nil omits it. Only
	// the v3 variants.
	Temperature *float64
	// MinBufferSize is the minimum characters buffered before audio generation;
	// nil uses 50.
	MinBufferSize *int
	// MaxChunkLength is the maximum characters processed per chunk; nil uses 150.
	MaxChunkLength *int
	// OutputAudioCodec is the audio codec; empty uses "linear16" (raw PCM).
	OutputAudioCodec string
	// OutputAudioBitrate is the audio bitrate; empty uses "128k".
	OutputAudioBitrate string
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Sarvam WebSocket TTS service.
//
// It holds one connection open for the call, opened when the pipeline starts.
// When it drops it is restored with backoff, and when that keeps failing the
// service reports a permanent error and becomes unusable. Sarvam's audio names
// no request, so an interruption while the bot speaks resets the connection,
// and the next request's frames are held until this one's audio has played.
// See tts.NewInterruptibleWebsocket.
func NewTTS(cfg TTSConfig) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns its synthesizer alongside it.
func newTTS(cfg TTSConfig) (*tts.Base, *synthesizer) {
	s := newSynthesizer(cfg)
	b := tts.New("SarvamTTS", s)
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	s.svc = b
	s.ws = tts.NewInterruptibleWebsocket(b, s, cfg.Websocket)
	return b, s
}

// newSynthesizer builds the synthesizer NewTTS drives, applying the defaults and
// settling the model-specific controls against the configured model.
func newSynthesizer(cfg TTSConfig) *synthesizer {
	if cfg.URL == "" {
		cfg.URL = defaultTTSURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultTTSModel
	}
	if cfg.Model == deprecatedTTSModel {
		slog.Warn("sarvam tts: bulbul:v2 is deprecated, use bulbul:v3 instead")
	}
	mc, ok := ttsModelConfigs[cfg.Model]
	if !ok {
		mc = ttsModelConfigs[defaultTTSModel]
	}
	s := &synthesizer{
		apiKey:     cfg.APIKey,
		url:        cfg.URL,
		sampleRate: firstNonZero(cfg.SampleRate, mc.defaultSampleRate),
		codec:      firstNonEmpty(cfg.OutputAudioCodec, defaultTTSCodec),
		bitrate:    firstNonEmpty(cfg.OutputAudioBitrate, defaultTTSBitrate),
		mc:         mc,
		live:       newTTSSettings(cfg, mc),
	}
	s.refreshConfig()
	return s
}

// resolveTTSLanguage maps a Language to Sarvam's regional code, defaulting to
// English (India) when unset or unsupported.
func resolveTTSLanguage(l language.Language) string {
	switch l.BaseCode() {
	case "bn":
		return "bn-IN"
	case "en":
		return "en-IN"
	case "gu":
		return "gu-IN"
	case "hi":
		return "hi-IN"
	case "kn":
		return "kn-IN"
	case "ml":
		return "ml-IN"
	case "mr":
		return "mr-IN"
	case "or":
		return "od-IN"
	case "pa":
		return "pa-IN"
	case "ta":
		return "ta-IN"
	case "te":
		return "te-IN"
	default:
		return defaultTTSLanguage
	}
}

// clampPace resolves the pace, defaulting to 1.0 and clamping to the model
// range with a warning.
func clampPace(pace *float64, mc ttsModelConfig) float64 {
	v := defaultPace
	if pace != nil {
		v = *pace
	}
	if v < mc.paceMin || v > mc.paceMax {
		slog.Warn("sarvam tts: pace is outside the model's range, clamping",
			"pace", v, "min", mc.paceMin, "max", mc.paceMax)
	}
	return min(max(v, mc.paceMin), mc.paceMax)
}

func firstNonZero(v, def int) int {
	if v != 0 {
		return v
	}
	return def
}

func firstNonEmpty(v, def string) string {
	if v != "" {
		return v
	}
	return def
}

func intOr(v *int, def int) int {
	if v != nil {
		return *v
	}
	return def
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
	apiKey     string
	url        string
	sampleRate int
	codec      string
	bitrate    string

	host tts.AudioContextHost
	svc  ttsService
	ws   *tts.Websocket

	// live is the settings store and mc the capabilities of its model, both
	// updated on the frame goroutine and read only there. The URL and the
	// config message built from them are copied under mu, since a redial uses
	// them from the receive loop.
	live *TTSSettings
	mc   ttsModelConfig

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex

	mu       sync.Mutex
	conn     *ttsConn
	endpoint string
	config   []byte
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

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.sampleRate }

// Metadata reports the Sarvam model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	model, _ := s.live.Model.Value()
	voice, _ := s.live.Voice.Value()
	return tts.Metadata{Model: model, VoiceID: voice}
}

// refreshConfig builds the URL a connection is dialed on, which pins the model
// and asks for the event that marks a synthesis complete, and the config
// message that opens a session. It runs on the frame goroutine, which is where
// the settings store is written.
func (s *synthesizer) refreshConfig() {
	model := s.live.Model.Or("")
	endpoint := fmt.Sprintf("%s?model=%s&send_completion_event=true", s.url, model)
	config := s.configMessage()
	s.mu.Lock()
	s.endpoint, s.config = endpoint, config
	s.mu.Unlock()
}

// configMessage builds the config message carrying the voice and the format.
// A control the model does not support is left off rather than sent and
// ignored.
func (s *synthesizer) configMessage() []byte {
	data := map[string]any{
		"target_language_code": s.live.Language.AsAny(),
		"speaker":              s.live.Voice.AsAny(),
		"speech_sample_rate":   strconv.Itoa(s.sampleRate),
		"enable_preprocessing": s.live.EnablePreprocessing.AsAny(),
		"min_buffer_size":      s.live.MinBufferSize.AsAny(),
		"max_chunk_length":     s.live.MaxChunkLength.AsAny(),
		"output_audio_codec":   s.codec,
		"output_audio_bitrate": s.bitrate,
		"pace":                 s.live.Pace.AsAny(),
		"model":                s.live.Model.AsAny(),
	}
	if v, ok := s.live.Pitch.Value(); ok && s.mc.supportsPitch {
		data["pitch"] = v
	}
	if v, ok := s.live.Loudness.Value(); ok && s.mc.supportsLoudness {
		data["loudness"] = v
	}
	if v, ok := s.live.Temperature.Value(); ok && s.mc.supportsTemperature {
		data["temperature"] = v
	}
	b, _ := json.Marshal(map[string]any{msgType: "config", msgData: data}) //nolint:errchkjson // serializable map
	return b
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
		s.ws.CreateKeepaliveTask(ctx, ttsKeepaliveInterval, s)
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
// tts.WebsocketHandler: it dials, then sends the config message. A failure is
// reported here and leaves no connection behind, which a redial reads as a
// failed attempt.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	s.mu.Lock()
	endpoint, config := s.endpoint, s.config
	s.mu.Unlock()
	header := http.Header{}
	header.Set("api-subscription-key", s.apiKey)
	conn, err := s.ws.Dial(ctx, endpoint, header, readLimitTTS)
	if err == nil {
		slog.Debug("connected to sarvam tts websocket")
		if err = s.writeConfig(ctx, conn, config); err != nil {
			_ = conn.CloseNow()
		}
	}
	if err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Error connecting to Sarvam TTS Websocket: %v", err))
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

// writeConfig sends the config message on conn. A failure is reported before it
// is returned.
func (s *synthesizer) writeConfig(ctx context.Context, conn *wsutil.Conn, config []byte) error {
	if err := conn.Write(ctx, websocket.MessageText, config); err != nil {
		if s.host != nil {
			s.host.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
		}
		return err
	}
	slog.Debug("sarvam tts: configuration sent")
	return nil
}

// sendConfig sends the config message on the open connection, for a change of
// settings the live session takes.
func (s *synthesizer) sendConfig(ctx context.Context) error {
	s.mu.Lock()
	c, config := s.conn, s.config
	s.mu.Unlock()
	if c == nil {
		return errTTSNotConnected
	}
	return s.writeConfig(ctx, c.conn, config)
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
	slog.Debug("disconnecting from sarvam tts")
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Error closing websocket: %v", err), err, false)
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

// ttsMessage is the subset of a Sarvam TTS WebSocket message read.
type ttsMessage struct {
	Type string `json:"type"`
	Data struct {
		Audio     string `json:"audio"`
		RequestID string `json:"request_id"`
		EventType string `json:"event_type"`
		Message   string `json:"message"`
	} `json:"data"`
}

// handleMessage handles one server message. Sarvam does not say which request
// a message belongs to, so it is taken to be the context playing.
//
// The final event says the context's synthesis is complete, which ends it at
// once rather than after the stop-frame timeout. An error is reported, and
// handed to the context playing.
func (s *synthesizer) handleMessage(ctx context.Context, data []byte) error {
	var m ttsMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("sarvam tts: decode message: %w", err)
	}
	if s.host == nil {
		return nil
	}
	contextID := s.svc.ActiveAudioContextID()
	switch {
	case m.Type == "audio":
		slog.Debug("sarvam tts: audio", "request", m.Data.RequestID, "context", contextID)
		pcm, err := base64.StdEncoding.DecodeString(m.Data.Audio)
		if err != nil {
			return fmt.Errorf("sarvam tts: decode audio: %w", err)
		}
		f := frames.NewTTSAudioRawFrame(pcm, s.sampleRate, 1)
		f.ContextID = contextID
		s.host.AppendToAudioContext(contextID, f)
	case m.Type == "event" && m.Data.EventType == "final":
		slog.Debug("sarvam tts: final event", "context", contextID)
		if contextID != "" && s.host.AudioContextAvailable(contextID) {
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = contextID
			s.host.AppendToAudioContext(contextID, stopped)
			s.host.RemoveAudioContext(contextID)
		}
	case m.Type == "error":
		msg := "TTS Error: " + m.Data.Message
		s.host.PushError(ctx, msg, nil, false)
		if lower := strings.ToLower(m.Data.Message); strings.Contains(lower, "too long") ||
			strings.Contains(lower, "timeout") {
			slog.Warn("sarvam tts: connection timeout detected, the service may need a restart")
		}
		s.host.AppendToAudioContext(contextID, frames.NewErrorFrame(msg))
	}
	return nil
}

// KeepaliveReady reports whether the connection is open, implementing
// tts.KeepaliveHandler.
func (s *synthesizer) KeepaliveReady() bool { return s.Connected() }

// SendKeepalive pings the server, which keeps the connection open through
// silences, implementing tts.KeepaliveHandler.
func (s *synthesizer) SendKeepalive(ctx context.Context) error {
	return s.send(ctx, map[string]any{msgType: "ping"})
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
		return fmt.Errorf("sarvam tts: %w", err)
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// sendText sends a sentence for synthesis. With no open connection it is not
// sent, and a warning says so.
func (s *synthesizer) sendText(ctx context.Context, text string) error {
	if !s.Connected() {
		slog.Warn("sarvam tts: websocket not ready, cannot send text")
		return nil
	}
	return s.send(ctx, map[string]any{msgType: "text", msgData: map[string]any{"text": text}})
}

// FlushAudio has Sarvam generate the text it is still buffering, implementing
// tts.AudioFlusher.
func (s *synthesizer) FlushAudio(ctx context.Context, _ string) {
	s.mu.Lock()
	connected := s.conn != nil
	s.mu.Unlock()
	if !connected {
		return
	}
	if err := s.send(ctx, map[string]any{msgType: "flush"}); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Error sending flush to Sarvam: %v", err), err, false)
	}
}

// send writes one JSON message on the open connection.
func (s *synthesizer) send(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errTTSNotConnected
	}
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.conn.Write(ctx, websocket.MessageText, b)
}

var (
	_ tts.Synthesizer        = (*synthesizer)(nil)
	_ tts.ContextSynthesizer = (*synthesizer)(nil)
	_ tts.Describer          = (*synthesizer)(nil)
	_ tts.WebsocketService   = (*synthesizer)(nil)
	_ tts.KeepaliveHandler   = (*synthesizer)(nil)
	_ tts.Starter            = (*synthesizer)(nil)
	_ tts.AudioFlusher       = (*synthesizer)(nil)
)
