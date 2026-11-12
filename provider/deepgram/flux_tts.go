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
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	ttstext "github.com/gojargo/jargo/utils/text"
)

const (
	// fluxSpeakURL is the streaming synthesis WebSocket.
	fluxSpeakURL = "wss://api.deepgram.com/v2/speak"
	// defaultFluxVoice is a current English Flux voice. Deepgram passes the
	// voice identifier as its "model" query parameter.
	defaultFluxVoice = "flux-heather-en"
	// fluxTTSReadLimit bounds a single incoming message. Audio frames can be
	// larger than the library's default, so raise the ceiling.
	fluxTTSReadLimit = 1 << 24
	// fluxDefaultSpeed is the rate Flux speaks at when no speed is set. A
	// Configure cannot ask Flux to forget a speed, so clearing the setting
	// sends this rate explicitly.
	fluxDefaultSpeed = 1.0
)

// The Flux synthesis messages the service sends and reads.
const (
	fluxTTSMsgSpeak             = "Speak"
	fluxTTSMsgFlush             = "Flush"
	fluxTTSMsgInterrupt         = "Interrupt"
	fluxTTSMsgConfigure         = "Configure"
	fluxTTSMsgConnected         = "Connected"
	fluxTTSMsgSpeechStarted     = "SpeechStarted"
	fluxTTSMsgFlushed           = "Flushed"
	fluxTTSMsgSpeechMetadata    = "SpeechMetadata"
	fluxTTSMsgSpeechInterrupted = "SpeechInterrupted"
	fluxTTSMsgSessionMetadata   = "SessionMetadata"
	fluxTTSMsgConfigureSuccess  = "ConfigureSuccess"
	fluxTTSMsgConfigureFailure  = "ConfigureFailure"
	fluxTTSMsgWarning           = "Warning"
	// fluxTTSWarningNoActiveSpeech is the warning code for an Interrupt the
	// server had no turn left to cancel.
	fluxTTSWarningNoActiveSpeech = "NO_ACTIVE_SPEECH"
)

// fluxTTSConfigureFields are the settings Flux can change on an open
// connection, with a Configure message. Every other setting is a query
// parameter, fixed for the life of the connection.
//
//nolint:gochecknoglobals // lookup table, read-only
var fluxTTSConfigureFields = []string{"speed"}

// errFluxUnsupportedRate is returned when a configured sample rate is not one
// Flux accepts.
//
//nolint:gochecknoglobals // sentinel error
var errFluxUnsupportedRate = errors.New("deepgram flux tts: unsupported sample rate")

// errFluxTTSNotConnected is returned by a send that found no connection to
// send on.
//
//nolint:gochecknoglobals // sentinel error
var errFluxTTSNotConnected = errors.New("deepgram flux tts: websocket not connected")

// fluxTTSSampleRates lists the sample rates Flux synthesis accepts.
func fluxTTSSampleRates() []int { return []int{8000, 16000, 24000, 32000, 44100, 48000} }

// FluxTTSConfig configures the Flux streaming TTS service. The audio is always
// linear16 (raw 16-bit little-endian mono PCM), the format the pipeline uses.
type FluxTTSConfig struct {
	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// SpeakURL overrides the synthesis WebSocket endpoint; empty uses the hosted
	// endpoint.
	SpeakURL string
	// Voice is the Flux voice identifier (e.g. "flux-heather-en"); empty uses
	// "flux-heather-en". It is sent as Deepgram's "model" query parameter, and
	// is the model the synthesis is reported against.
	Voice string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	// A non-zero rate must be one of 8000, 16000, 24000, 32000, 44100, 48000.
	SampleRate int
	// Speed multiplies the speaking rate, from 0.5 to 1.5 in steps of 0.05. Nil
	// leaves Flux at its default rate. A change while the pipeline runs is
	// applied to the open connection, so it keeps the acoustic state Flux holds
	// across turns.
	Speed *float64 `validate:"omitempty,gte=0.5,lte=1.5"`
	// Expressivity is the expressive range, on a calm (-2) to animated (2)
	// axis. Nil leaves Flux at its default range. Flux fixes it when the
	// connection opens, so a change while the pipeline runs reconnects.
	Expressivity *int `validate:"omitempty,gte=-2,lte=2"`
	// NormalizeVolume says whether Flux levels the loudness of the speech
	// across voices and turns. Nil leaves Flux at its default, which levels it.
	// Fixed when the connection opens, so a change while the pipeline runs
	// reconnects.
	NormalizeVolume *bool
	// MipOptOut opts out of Deepgram's model-improvement program.
	MipOptOut *bool
	// Tag attaches billing tags to the request.
	Tag []string
	// ExtraQuery sets arbitrary additional query parameters; values override any
	// param of the same name set from other fields.
	ExtraQuery map[string]string
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c FluxTTSConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.SampleRate == 0 || slices.Contains(fluxTTSSampleRates(), c.SampleRate) {
		return nil
	}
	return fmt.Errorf("%w: %d", errFluxUnsupportedRate, c.SampleRate)
}

// withDefaults returns a copy of the config with empty fields filled in.
func (c FluxTTSConfig) withDefaults() FluxTTSConfig {
	if c.Voice == "" {
		c.Voice = defaultFluxVoice
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSSampleRate
	}
	if c.SpeakURL == "" {
		c.SpeakURL = fluxSpeakURL
	}
	return c
}

// FluxTTSSettings is the part of the Flux synthesis configuration that can
// change while the pipeline runs.
//
// The Flux voice is a single identifier (e.g. "flux-heather-en") carried by
// Voice. Deepgram takes it as its "model" query parameter, so Model follows
// Voice: it is what the synthesis is reported and priced against.
//
// Speed is applied to the open connection with a Configure message. Every
// other setting is a query parameter fixed for the life of the connection, so
// a change to one reconnects.
type FluxTTSSettings struct {
	settings.TTS

	// Speed multiplies the speaking rate, from 0.5 to 1.5. No value leaves
	// Flux at its default rate.
	Speed settings.Opt[float64] `settings:"speed"`
	// Expressivity is the expressive range, from -2 (calm) to 2 (animated). No
	// value leaves Flux at its default range.
	Expressivity settings.Opt[int] `settings:"expressivity"`
	// NormalizeVolume says whether Flux levels the loudness of the speech. No
	// value leaves Flux at its default, which levels it.
	NormalizeVolume settings.Opt[bool] `settings:"normalize_volume"`
}

// newFluxTTSSettings is the starting state, taken from what the service was
// built with.
func newFluxTTSSettings(cfg FluxTTSConfig) *FluxTTSSettings {
	s := &FluxTTSSettings{}
	s.Voice = settings.Set(cfg.Voice)
	s.Model = settings.Set(cfg.Voice)
	if cfg.Speed != nil {
		s.Speed = settings.Set(*cfg.Speed)
	}
	if cfg.Expressivity != nil {
		s.Expressivity = settings.Set(*cfg.Expressivity)
	}
	if cfg.NormalizeVolume != nil {
		s.NormalizeVolume = settings.Set(*cfg.NormalizeVolume)
	}
	return s
}

// ApplyUpdate merges an update and keeps the model in step with the voice,
// implementing settings.Updater. Deepgram takes the voice as the model, so a
// voice change is also reported as a model change, which relabels the
// synthesis the service reports from then on.
func (s *FluxTTSSettings) ApplyUpdate(delta any) (settings.Changed, error) {
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

// NewFluxTTS builds a Deepgram Flux streaming WebSocket TTS service.
//
// Flux keeps acoustic state across turns on a single connection, so prosody
// and pacing stay consistent throughout a conversation: the service opens one
// connection when the pipeline starts and speaks every turn on it, restoring it
// with backoff if it fails.
//
// The model's tokens are streamed straight through rather than grouped into
// sentences first, which is what Flux is for: synthesis starts on the first
// token instead of waiting for the sentence to finish. Call SetTextAggregator on
// the returned Base to group them into sentences instead, which reads more
// naturally at the cost of that wait.
//
// Flux's inline pronunciation markup is early access and is refused with a
// fatal error on any connection whose speed is not 1.0, so the service formats
// no pronunciations: words given a pronunciation are spoken as written.
func NewFluxTTS(cfg FluxTTSConfig) *tts.Base {
	b, _ := newFluxTTS(cfg)
	return b
}

// newFluxTTS builds the service and the synthesizer behind it.
func newFluxTTS(cfg FluxTTSConfig) (*tts.Base, *fluxSynth) {
	cfg = cfg.withDefaults()
	s := &fluxSynth{cfg: cfg, live: newFluxTTSSettings(cfg)}
	b := tts.New("DeepgramFluxTTS", s)
	s.base = b
	// The connection carries one Flux session, whose audio names no turn: it
	// is labeled with the context playing, so replacing the connection leaves
	// the open contexts to end as they would on their own.
	s.ws = tts.NewWebsocketConnection(b, s, cfg.Websocket)
	b.SetTextAggregator(ttstext.NewTokenAggregator())
	// The audio names no turn, so turns are taken one at a time: a reply waits
	// until the one before it has all its audio.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s
}

// fluxTTSQuery builds the synthesis query string, from the settings as they
// stand rather than as the service was built: a connection opened after a
// change to a connection-only setting is where that change takes effect.
func fluxTTSQuery(cfg FluxTTSConfig, live *FluxTTSSettings) url.Values {
	q := url.Values{}
	q.Set("model", live.Voice.Or(cfg.Voice))
	q.Set("encoding", fluxEncoding)
	q.Set("sample_rate", strconv.Itoa(cfg.SampleRate))
	if v, ok := live.Speed.Value(); ok {
		q.Set("speed", strconv.FormatFloat(v, 'f', -1, 64))
	}
	if v, ok := live.Expressivity.Value(); ok {
		q.Set("expressivity", strconv.Itoa(v))
	}
	if v, ok := live.NormalizeVolume.Value(); ok {
		q.Set("normalize_volume", strconv.FormatBool(v))
	}
	query.SetBoolOpt(q, "mip_opt_out", cfg.MipOptOut)
	query.AddAll(q, "tag", cfg.Tag)
	for k, v := range cfg.ExtraQuery {
		q.Set(k, v)
	}
	return q
}

// fluxSpeak is the message that sends text to synthesize.
type fluxSpeak struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// fluxTTSControl is a control message: a Flush, an Interrupt, or a Configure
// carrying the speed.
type fluxTTSControl struct {
	Type  string   `json:"type"`
	Speed *float64 `json:"speed,omitempty"`
}

// fluxTTSMessage is the subset of a Flux synthesis control message we read.
type fluxTTSMessage struct {
	Type                   string          `json:"type"`
	RequestID              string          `json:"request_id"`
	ModelName              string          `json:"model_name"`
	SpeechID               string          `json:"speech_id"`
	AudioDurationMs        any             `json:"audio_duration_ms"`
	AudioPlayedMs          any             `json:"audio_played_ms"`
	BillableCharacterCount any             `json:"billable_character_count"`
	Applied                json.RawMessage `json:"applied"`
	Code                   any             `json:"code"`
	Field                  any             `json:"field"`
	Value                  any             `json:"value"`
	Description            string          `json:"description"`
}

// describe is the message's description, or fallback when it has none.
func (m fluxTTSMessage) describe(fallback string) string {
	if m.Description != "" {
		return m.Description
	}
	return fallback
}

// fluxSynth speaks over one Flux WebSocket held open for the call.
//
// Flux keeps one turn in flight per connection and labels nothing it sends
// with the turn it belongs to: audio arrives as bare binary frames, and the
// turn ends with a SpeechMetadata message. Both are attributed to the context
// playing, which is safe because frame handling pauses while a turn's audio is
// generated, so turns never overlap, and an interruption leaves no context
// playing, so audio the server had already generated for an abandoned turn is
// dropped.
type fluxSynth struct {
	cfg FluxTTSConfig
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every connection is opened with.
	live *FluxTTSSettings

	// base is the service this synthesizes for, and host the audio contexts
	// its audio is appended to.
	base *tts.Base
	host tts.AudioContextHost
	// ws is the connection lifecycle: it runs the receive loop, restores a lost
	// connection with backoff, and reports what fails.
	ws *tts.Websocket

	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	// mu guards the connection. It is never held across a call into the host:
	// appending blocks while the audio plays out.
	mu   sync.Mutex
	conn *wsutil.Conn
	// closed records that the read on conn ended: the connection is gone.
	closed bool
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *fluxSynth) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *fluxSynth) SampleRate() int { return s.cfg.SampleRate }

// RequiresTrailingSpace reports that Flux neither inserts nor strips whitespace
// between the units it is sent, so consecutive sentences would otherwise run
// together into one word. When tokens are streamed their own whitespace is
// used as it is.
func (s *fluxSynth) RequiresTrailingSpace() bool { return true }

// Metadata reports the voice synthesis is billed against, which Deepgram takes
// as the model.
func (s *fluxSynth) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(s.cfg.Voice), VoiceID: s.live.Voice.Or(s.cfg.Voice)}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *fluxSynth) Settings() any { return s.live }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. A speed
// change is sent to Flux with a Configure message. Every other setting is a
// query parameter, so a change to one reconnects.
func (s *fluxSynth) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed.Except(fluxTTSConfigureFields...)) > 0 {
		s.Disconnect(ctx)
		s.Connect(ctx)
	} else if len(changed) > 0 {
		s.sendConfigure(ctx)
	}
	return nil
}

// sendConfigure applies the current speed to the open connection.
func (s *fluxSynth) sendConfigure(ctx context.Context) {
	if !s.Connected() {
		return
	}
	speed := s.live.Speed.Or(fluxDefaultSpeed)
	if err := s.send(ctx, fluxTTSControl{Type: fluxTTSMsgConfigure, Speed: &speed}); err != nil {
		slog.Error("deepgram flux tts: sending the Configure message failed", "service", s.base.Name(), "err", err)
	}
}

// Start connects when the pipeline starts, so the handshake is paid while the
// transport is still negotiating rather than in front of the first thing the
// bot says. It implements tts.Starter.
func (s *fluxSynth) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect readies a fresh connection, opens it, and starts the receive loop on
// it, implementing tts.WebsocketService.
func (s *fluxSynth) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect stops the receive loop and closes the connection, implementing
// tts.WebsocketService.
func (s *fluxSynth) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens the connection with the settings as they stand,
// implementing tts.WebsocketHandler. A failure is reported here, with
// Deepgram's reason when it refused the handshake, and leaves no connection
// behind, which the receive loop reads as a failed attempt.
func (s *fluxSynth) ConnectWebsocket(ctx context.Context) error {
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to deepgram flux tts")
	s.validateSampleRate()

	header := http.Header{}
	header.Set("Authorization", authToken(s.cfg.APIKey))
	endpoint := s.cfg.SpeakURL + "?" + fluxTTSQuery(s.cfg, s.live).Encode()
	conn, err := s.ws.Dial(ctx, endpoint, header, fluxTTSReadLimit)
	if err != nil {
		s.mu.Lock()
		s.conn, s.closed = nil, false
		s.mu.Unlock()
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
	s.conn, s.closed = conn, false
	s.mu.Unlock()
	return nil
}

// validateSampleRate warns when the rate is one Flux does not synthesize. The
// connection is still attempted, so Deepgram's own answer is what decides.
func (s *fluxSynth) validateSampleRate() {
	if !slices.Contains(fluxTTSSampleRates(), s.cfg.SampleRate) {
		slog.Warn("deepgram flux tts: sample rate is not supported",
			"service", s.base.Name(), "sample_rate", s.cfg.SampleRate, "supported", fluxTTSSampleRates())
	}
}

// DisconnectWebsocket closes the connection, implementing
// tts.WebsocketHandler. What the service was timing is reported as far as it
// got first: the audio it was waiting for will not come on this connection.
//
// No Close message is sent: Close asks Flux to generate the rest of the
// active turn, which a teardown has no use for. Closing the socket ends the
// session outright.
func (s *fluxSynth) DisconnectWebsocket(ctx context.Context) error {
	s.base.StopAllMetrics(ctx)
	s.mu.Lock()
	conn, open := s.conn, s.conn != nil && !s.closed
	s.conn, s.closed = nil, false
	s.mu.Unlock()
	if !open {
		return nil
	}
	slog.Debug("disconnecting from deepgram flux tts")
	if err := conn.Close(websocket.StatusNormalClosure, ""); err != nil {
		s.host.PushError(ctx, fmt.Sprintf("%s error: %v", s.base.Name(), err), err, false)
	}
	return nil
}

// Connected reports whether there is a live connection, implementing
// tts.WebsocketHandler.
func (s *fluxSynth) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.closed
}

// send sends one JSON message on the connection, serialized against the other
// writers, failing with errFluxTTSNotConnected when there is none.
func (s *fluxSynth) send(ctx context.Context, msg any) error {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return errFluxTTSNotConnected
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(ctx, websocket.MessageText, payload)
}

// ReceiveMessages reads and handles messages until the connection ends,
// implementing tts.WebsocketHandler: binary frames are audio, text frames
// control messages.
func (s *fluxSynth) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return errFluxTTSNotConnected
	}
	for {
		typ, data, err := conn.Read(ctx)
		if err != nil {
			s.mu.Lock()
			if s.conn == conn {
				s.closed = true
			}
			s.mu.Unlock()
			return err
		}
		if typ == websocket.MessageBinary {
			s.handleAudio(data)
			continue
		}
		var m fluxTTSMessage
		if err := json.Unmarshal(data, &m); err != nil {
			slog.Error("deepgram flux tts sent an invalid JSON message", "message", string(data))
			continue
		}
		s.handleMessage(ctx, &m, data)
	}
}

// handleAudio appends a chunk of audio to the context playing.
//
// Audio carries no speech id, so it is attributed to the context playing. Turns
// are taken one at a time, and an interruption leaves no context playing, so
// audio the server had already generated for an abandoned turn is dropped.
func (s *fluxSynth) handleAudio(pcm []byte) {
	contextID := s.base.ActiveAudioContextID()
	f := frames.NewTTSAudioRawFrame(pcm, s.SampleRate(), 1)
	f.ContextID = contextID
	s.host.AppendToAudioContext(contextID, f)
}

// handleMessage acts on one control message from Flux.
func (s *fluxSynth) handleMessage(ctx context.Context, m *fluxTTSMessage, raw []byte) {
	name := s.base.Name()
	switch m.Type {
	case fluxTTSMsgConnected:
		slog.Debug("deepgram flux tts connected", "service", name,
			"request_id", m.RequestID, "model", m.ModelName)
	case fluxTTSMsgSpeechStarted:
		slog.Debug("deepgram flux tts speech started", "service", name, "speech_id", m.SpeechID)
	case fluxTTSMsgFlushed:
		// Not the end of the turn: Flux acknowledges the flush and may still
		// send audio afterwards. SpeechMetadata is what ends the turn.
		slog.Debug("deepgram flux tts flushed", "service", name, "speech_id", m.SpeechID)
	case fluxTTSMsgSpeechMetadata:
		// Sent once per turn, after all of its audio.
		slog.Debug("deepgram flux tts speech complete", "service", name, "speech_id", m.SpeechID,
			"duration_ms", m.AudioDurationMs, "billable_characters", m.BillableCharacterCount)
		contextID := s.base.ActiveAudioContextID()
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		s.host.AppendToAudioContext(contextID, stopped)
		s.host.RemoveAudioContext(contextID)
	case fluxTTSMsgSpeechInterrupted:
		// Acknowledges an Interrupt. The total it reports is the audio the
		// server generated, not the audio the user heard: the request carries
		// no playback offset.
		slog.Debug("deepgram flux tts speech interrupted", "service", name,
			"speech_id", m.SpeechID, "audio_played_ms", m.AudioPlayedMs)
	case fluxTTSMsgSessionMetadata:
		slog.Debug("deepgram flux tts session totals", "service", name, "message", string(raw))
	case fluxTTSMsgConfigureSuccess:
		slog.Debug("deepgram flux tts configuration applied", "service", name, "applied", string(m.Applied))
	case fluxTTSMsgConfigureFailure:
		// Not fatal: synthesis carries on with the previous configuration, so
		// the application decides what a rejected settings update means for
		// the conversation.
		s.host.PushError(ctx, fmt.Sprintf("%s configuration rejected %v (%v=%v): %s",
			name, m.Code, m.Field, m.Value, m.describe("Unknown failure")), nil, false)
	case fluxTTSMsgWarning:
		if fmt.Sprint(m.Code) == fluxTTSWarningNoActiveSpeech {
			// Routine: a context outlives the turn that filled it, so an
			// interruption late in playback sends an Interrupt the server has
			// nothing left to cancel.
			slog.Debug("deepgram flux tts had no active turn to interrupt", "service", name)
			return
		}
		slog.Warn("deepgram flux tts warning", "service", name,
			"code", m.Code, "description", m.describe("Unknown warning"))
	case fluxMsgError:
		msg := fmt.Sprintf("%s error %v: %s", name, m.Code, m.describe("Unknown error"))
		slog.Error(msg)
		s.host.PushError(ctx, msg, nil, false)
	default:
		slog.Debug("deepgram flux tts sent an unknown message type", "message", string(raw))
	}
}

// RunTTS sends the text on the connection and returns: the audio arrives on
// the receive loop, which appends it to the context playing.
//
// A connection the server has closed (Flux caps a session at an hour) is
// dropped and dialed again in full, which restarts the receive loop.
func (s *fluxSynth) RunTTS(ctx context.Context, text, _ string, _ func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Disconnect(ctx)
		s.Connect(ctx)
		if !s.Connected() {
			return fmt.Errorf("%s: %w", s.base.Name(), errFluxTTSNotConnected)
		}
	}
	if err := s.send(ctx, fluxSpeak{Type: fluxTTSMsgSpeak, Text: text}); err != nil {
		return fmt.Errorf("deepgram flux tts: unknown error occurred: %w", err)
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// FlushAudio sends a Flush, implementing tts.AudioFlusher. It ends the turn:
// Flux generates the rest of its audio and then reports the turn's
// SpeechMetadata. A connection carries one turn at a time, so no context is
// named.
func (s *fluxSynth) FlushAudio(ctx context.Context, _ string) {
	if !s.Connected() {
		return
	}
	if err := s.send(ctx, fluxTTSControl{Type: fluxTTSMsgFlush}); err != nil {
		slog.Error("deepgram flux tts: sending the Flush message failed", "service", s.base.Name(), "err", err)
	}
}

// OnAudioContextInterrupted ends the turn on the connection when the user cuts
// in, implementing tts.AudioContextInterrupter. An Interrupt ends the active
// turn without closing the connection, so the acoustic state Flux keeps across
// turns survives the interruption.
func (s *fluxSynth) OnAudioContextInterrupted(ctx context.Context, _ string) {
	if !s.Connected() {
		return
	}
	// Interrupt takes no other field: Flux rejects unknown ones and closes the
	// connection.
	if err := s.send(context.WithoutCancel(ctx), fluxTTSControl{Type: fluxTTSMsgInterrupt}); err != nil {
		slog.Error("deepgram flux tts: sending the Interrupt message failed", "service", s.base.Name(), "err", err)
	}
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*fluxSynth)(nil)
	_ tts.AudioFlusher            = (*fluxSynth)(nil)
	_ tts.AudioContextInterrupter = (*fluxSynth)(nil)
	_ tts.Starter                 = (*fluxSynth)(nil)
	_ tts.Describer               = (*fluxSynth)(nil)
	_ tts.TrailingSpaceRequirer   = (*fluxSynth)(nil)
	_ tts.SettingsUpdater         = (*fluxSynth)(nil)
	_ tts.WebsocketService        = (*fluxSynth)(nil)
	_ settings.Updater            = (*FluxTTSSettings)(nil)
)
