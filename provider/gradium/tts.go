package gradium

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

// errTTSNotConnected is returned by a send that found no connection to send
// on, the dial having failed and been reported.
//
//nolint:gochecknoglobals // sentinel error
var errTTSNotConnected = errors.New("gradium tts: websocket not connected")

const (
	defaultTTSURL = "wss://api.gradium.ai/api/speech/tts"
	// defaultVoiceID is Gradium's default voice.
	defaultVoiceID = "_6Aslh2DxfmnRLmP"
	// defaultTTSModel is the model the synthesis is reported against. Gradium
	// is not told which model to use: the voice decides.
	defaultTTSModel = "default"
	// ttsSampleRate is the fixed PCM rate of Gradium's TTS output.
	ttsSampleRate = 48000
	// ttsAPISource names the integration to Gradium.
	ttsAPISource = "jargo"
)

// The Gradium TTS message types this service sends and reads, beyond the ones
// it shares with the STT service.
const (
	msgSetup = "setup"
	msgReady = "ready"
)

// TTSConfig configures the Gradium TTS service.
type TTSConfig struct {
	// APIKey is the Gradium API key, sent as the x-api-key header. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// VoiceID is the voice identifier; empty uses a default voice.
	VoiceID string
	// JSONConfig is an optional JSON configuration string forwarded to the model;
	// empty omits it.
	JSONConfig string
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// TTSSettings is the part of the synthesis configuration that can change while
// the pipeline runs. A new voice reconnects, so every context opened after it
// is set up with it. The model and the language are not sent to Gradium, so a
// change to either is stored and warned about.
type TTSSettings struct {
	settings.TTS
}

// newTTSSettings is the starting state, taken from what the service was built
// with.
func newTTSSettings(cfg TTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(defaultTTSModel)
	s.Voice = settings.Set(cfg.VoiceID)
	return s
}

// NewTTS builds a Gradium TTS service. It holds one connection open for the
// call, restoring it with backoff if it fails, and speaks each turn on a
// request of its own on that connection.
func NewTTS(cfg TTSConfig) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and the synthesizer behind it.
func newTTS(cfg TTSConfig) (*tts.Base, *ttsSynthesizer) {
	if cfg.URL == "" {
		cfg.URL = defaultTTSURL
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoiceID
	}
	s := &ttsSynthesizer{cfg: cfg, live: newTTSSettings(cfg), setupContexts: map[string]bool{}}
	b := tts.New("GradiumTTS", s)
	s.base = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	// A reply waits until the one before it has all its audio.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s
}

// ttsSynthesizer speaks over one Gradium WebSocket held open for the call.
//
// Each turn is a request of its own on the connection, named by its context
// id: the request is set up with the voice before its first text, and every
// message Gradium sends back carries the id, so audio still coming for a turn
// the user cut off is dropped once its context is gone.
type ttsSynthesizer struct {
	cfg TTSConfig
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is set up with.
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

	// mu guards the connection and the requests set up on it. It is never held
	// across a call into the host: appending blocks while the audio plays out.
	mu   sync.Mutex
	conn *wsutil.Conn
	// closed records that the read on conn ended: the connection is gone.
	closed bool
	// setupContexts are the contexts a request has been set up for on this
	// connection.
	setupContexts map[string]bool
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *ttsSynthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports Gradium's fixed 48 kHz PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return ttsSampleRate }

// Metadata reports the model and voice synthesis is billed against.
func (s *ttsSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(defaultTTSModel), VoiceID: s.live.Voice.Or(s.cfg.VoiceID)}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *ttsSynthesizer) Settings() any { return s.live }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. A new
// voice reconnects; any other change is stored and warned about, since
// Gradium is never sent it.
func (s *ttsSynthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if changed.Has("voice") {
		s.Disconnect(ctx)
		s.Connect(ctx)
		return nil
	}
	s.base.WarnUnhandledUpdatedSettings(ctx, changed.Names())
	return nil
}

// Start connects when the pipeline starts, so the handshake is paid while the
// transport is still negotiating rather than in front of the first thing the
// bot says. It implements tts.Starter.
func (s *ttsSynthesizer) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect readies a fresh connection, opens it, and starts the receive loop on
// it, implementing tts.WebsocketService. A receive loop left over from a
// connection the server closed is stopped first, so a new one starts on the
// new connection.
func (s *ttsSynthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	slog.Debug("connecting to gradium tts")
	if !s.Connected() {
		s.ws.CancelReceiveTask()
	}
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect ends the audio contexts of the connection, stops the receive loop
// and closes the connection, implementing tts.WebsocketService.
func (s *ttsSynthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	slog.Debug("disconnecting from gradium tts")
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens the connection, implementing tts.WebsocketHandler. A
// failure is reported here and leaves no connection behind, which the receive
// loop reads as a failed attempt.
func (s *ttsSynthesizer) ConnectWebsocket(ctx context.Context) error {
	if s.Connected() {
		return nil
	}
	header := http.Header{}
	header.Set("x-api-key", s.cfg.APIKey)
	header.Set("x-api-source", ttsAPISource)
	conn, err := s.ws.Dial(ctx, s.cfg.URL, header, wsutil.DefaultReadLimit)
	if err != nil {
		s.mu.Lock()
		s.conn, s.closed = nil, false
		s.mu.Unlock()
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
		ef.Err = err
		s.ws.ReportError(ctx, ef, false)
		return nil
	}
	s.mu.Lock()
	s.conn, s.closed = conn, false
	s.mu.Unlock()
	return nil
}

// DisconnectWebsocket closes the connection and forgets the requests set up on
// it, implementing tts.WebsocketHandler. What the service was timing is
// reported as far as it got first: the audio it was waiting for will not come
// on this connection.
func (s *ttsSynthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.base.StopAllMetrics(ctx)
	s.mu.Lock()
	conn, open := s.conn, s.conn != nil && !s.closed
	s.conn, s.closed = nil, false
	clear(s.setupContexts)
	s.mu.Unlock()
	if !open {
		return nil
	}
	if err := conn.Close(websocket.StatusNormalClosure, ""); err != nil {
		s.host.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
	}
	return nil
}

// Connected reports whether there is a live connection, implementing
// tts.WebsocketHandler.
func (s *ttsSynthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.closed
}

// send sends one JSON message on the connection, serialized against the other
// writers, failing with errTTSNotConnected when there is none.
func (s *ttsSynthesizer) send(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return errTTSNotConnected
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(ctx, websocket.MessageText, payload)
}

// setupMessage sets up the request of one context: the voice, the output
// format, and the connection kept open when the request ends.
func (s *ttsSynthesizer) setupMessage(contextID string) map[string]any {
	msg := map[string]any{
		msgType:           msgSetup,
		"output_format":   encPCM,
		"voice_id":        s.live.Voice.Or(s.cfg.VoiceID),
		"close_ws_on_eos": false,
		keyClientReqID:    contextID,
	}
	if s.cfg.JSONConfig != "" {
		msg["json_config"] = s.cfg.JSONConfig
	}
	return msg
}

// textMessage adds text to the request of one context.
func textMessage(text, contextID string) map[string]any {
	return map[string]any{msgText: text, msgType: msgText, keyClientReqID: contextID}
}

// ttsMessage is the subset of a Gradium TTS WebSocket message we read.
type ttsMessage struct {
	Type        string  `json:"type"`
	ClientReqID string  `json:"client_req_id"`
	Audio       string  `json:"audio"`
	Text        string  `json:"text"`
	StartS      float64 `json:"start_s"`
	Message     string  `json:"message"`
}

// ReceiveMessages reads and handles messages until the connection ends,
// implementing tts.WebsocketHandler. Each message is delivered to the context
// its request id names. A message that cannot be read ends the connection,
// which the receive loop restores.
func (s *ttsSynthesizer) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	conn, closed := s.conn, s.closed
	s.mu.Unlock()
	if conn == nil || closed {
		return errTTSNotConnected
	}
	for {
		_, data, err := conn.Read(ctx)
		if err == nil {
			err = s.handleMessage(ctx, data)
		}
		if err != nil {
			s.mu.Lock()
			if s.conn == conn {
				s.closed = true
			}
			s.mu.Unlock()
			return err
		}
	}
}

// handleMessage acts on one message from Gradium.
func (s *ttsSynthesizer) handleMessage(ctx context.Context, data []byte) error {
	var m ttsMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("gradium tts: reading a message: %w", err)
	}
	contextID := m.ClientReqID
	available := contextID != "" && s.host.AudioContextAvailable(contextID)

	switch m.Type {
	case msgAudio:
		if !available {
			return nil
		}
		pcm, err := base64.StdEncoding.DecodeString(m.Audio)
		if err != nil {
			return fmt.Errorf("gradium tts: decoding audio: %w", err)
		}
		f := frames.NewTTSAudioRawFrame(pcm, s.SampleRate(), 1)
		f.ContextID = contextID
		s.host.AppendToAudioContext(contextID, f)
	case msgText:
		if available {
			s.host.AddWordTimestamps(contextID,
				[]uctx.WordTiming{{Word: m.Text, Offset: m.StartS}}, tts.WordTimingOptions{})
		}
	case msgReady:
	case msgEndStream:
		if available {
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = contextID
			s.host.AppendToAudioContext(contextID, stopped)
			s.host.RemoveAudioContext(contextID)
		}
		if contextID != "" {
			s.mu.Lock()
			delete(s.setupContexts, contextID)
			s.mu.Unlock()
		}
		s.base.StopAllMetrics(ctx)
	case msgError:
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		_ = s.host.PushFrame(ctx, stopped, processor.Downstream)
		s.base.StopAllMetrics(ctx)
		reason := m.Message
		if reason == "" {
			reason = string(data)
		}
		s.host.PushError(ctx, "Error: "+reason, nil, false)
	}
	return nil
}

// RunTTS sends the text on the request of its context and returns: the audio
// arrives on the receive loop, which appends it to the context it names. The
// request is set up with the voice before its first text.
//
// A sentence that finds the connection closed opens a new one. A send that
// fails stops the sentence's speech and has the connection dropped and dialed
// again.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	err := s.sendText(ctx, text, contextID)
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
	return fmt.Errorf("gradium tts: unknown error occurred: %w", err)
}

// sendText sets the context's request up if it is new on this connection, and
// sends the text on it.
func (s *ttsSynthesizer) sendText(ctx context.Context, text, contextID string) error {
	if !s.Connected() {
		return errTTSNotConnected
	}
	s.mu.Lock()
	isSetUp := s.setupContexts[contextID]
	s.mu.Unlock()
	if !isSetUp {
		if err := s.send(ctx, s.setupMessage(contextID)); err != nil {
			return err
		}
		s.mu.Lock()
		s.setupContexts[contextID] = true
		s.mu.Unlock()
	}
	return s.send(ctx, textMessage(text, contextID))
}

// RunTTSTimed sends the text like RunTTS, implementing tts.WordTimestamps.
// Gradium reports when each word is spoken, on the receive loop with the
// audio, so the word callback is not used here; implementing the interface is
// what puts the base on the word-aligned path.
func (s *ttsSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	yield func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, yield)
}

// FlushAudio ends the request of the context, implementing tts.AudioFlusher:
// Gradium generates what it is still holding and closes the request with an
// end_of_stream. With no context named it ends the one playing, or the
// turn's.
func (s *ttsSynthesizer) FlushAudio(ctx context.Context, contextID string) {
	flushID := contextID
	if flushID == "" {
		flushID = s.base.ActiveAudioContextID()
	}
	s.mu.Lock()
	connected := s.conn != nil
	s.mu.Unlock()
	if flushID == "" || !connected {
		return
	}
	err := s.send(ctx, map[string]any{msgType: msgEndStream, keyClientReqID: flushID})
	switch {
	case err == nil:
	case websocket.CloseStatus(err) == websocket.StatusNormalClosure:
		slog.Debug("gradium tts connection closed normally during flush")
	default:
		slog.Error("gradium tts: ending the request failed", "context", flushID, "err", err)
	}
}

// OnAudioContextInterrupted reports what the service was timing as far as it
// got, implementing tts.AudioContextInterrupter. Gradium is told nothing: the
// audio still coming for the interrupted request is dropped once its context
// is gone.
func (s *ttsSynthesizer) OnAudioContextInterrupted(ctx context.Context, _ string) {
	s.base.StopAllMetrics(ctx)
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*ttsSynthesizer)(nil)
	_ tts.WordTimestamps          = (*ttsSynthesizer)(nil)
	_ tts.AudioFlusher            = (*ttsSynthesizer)(nil)
	_ tts.AudioContextInterrupter = (*ttsSynthesizer)(nil)
	_ tts.Starter                 = (*ttsSynthesizer)(nil)
	_ tts.Describer               = (*ttsSynthesizer)(nil)
	_ tts.SettingsUpdater         = (*ttsSynthesizer)(nil)
	_ tts.WebsocketService        = (*ttsSynthesizer)(nil)
)
