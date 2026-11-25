package rime

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/rime/internal/rimepb"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	ttstext "github.com/gojargo/jargo/text"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// The WebSocket protocols the V1 service speaks: the envelope encoding, which
// the connection selects with its subprotocol.
const (
	// V1ProtocolBinary sends the envelopes in protobuf binary encoding, in
	// binary frames.
	V1ProtocolBinary = "binary"
	// V1ProtocolJSON sends the envelopes in proto3 JSON encoding, in text
	// frames.
	V1ProtocolJSON = "json"
)

const (
	// modelMistV2 is the older mist model, which takes no time scale factor.
	modelMistV2 = "mistv2"
	// v1ProtocolVersion is the WebSocket v1 protocol version the service speaks.
	v1ProtocolVersion = 1
	// v1ReadyTimeout bounds the wait for Rime to say it can accept contexts.
	v1ReadyTimeout = 10 * time.Second
	// v1AudioFormat is the audio format requested: raw 16-bit PCM.
	v1AudioFormat = "audio/pcm"
)

// v1ModelURLs are the public endpoints of the models that have one.
//
//nolint:gochecknoglobals // read-only lookup table
var v1ModelURLs = map[string]string{
	modelCoda: "wss://coda.api.rime.ai/ws",
	"mistv3":  "wss://mist.api.rime.ai/ws",
}

// v1Subprotocols names the subprotocol that selects each envelope encoding.
//
//nolint:gochecknoglobals // read-only lookup table
var v1Subprotocols = map[string]string{
	V1ProtocolBinary: "rime.v1.binary",
	V1ProtocolJSON:   "rime.v1.json",
}

// v1ErrorCategories classifies the error kinds Rime reports.
//
//nolint:gochecknoglobals // read-only lookup table
var v1ErrorCategories = map[string]errclass.Category{
	"invalid_input":      errclass.InvalidRequest,
	"unauthenticated":    errclass.Authentication,
	"permission_denied":  errclass.Authorization,
	"not_found":          errclass.InvalidRequest,
	"resource_exhausted": errclass.RateLimit,
	"timeout":            errclass.Connectivity,
	"unavailable":        errclass.Connectivity,
	"unimplemented":      errclass.Server,
	"internal":           errclass.Server,
}

// Configuration errors the V1 service is built with.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errV1Protocol      = errors.New(`rime v1 tts: the protocol must be "binary" or "json"`)
	errV1ModelRequired = errors.New("rime v1 tts: Rime WebSocket v1 requires a model")
	errV1URLRequired   = errors.New("rime v1 tts: a URL is required for models other than coda and mistv3")
	errV1URLInvalid    = errors.New("rime v1 tts: the URL must be an absolute ws or wss URL")
)

// Connection errors.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errV1NotConnected     = errors.New("rime v1 tts: websocket not connected")
	errV1Subprotocol      = errors.New("rime v1 tts: Rime did not accept the subprotocol")
	errV1Ready            = errors.New("rime v1 tts: Rime did not send a supported ready event")
	errV1TextOnBinaryWire = errors.New("rime v1 tts: Rime sent a text frame on the binary protocol")
)

// v1ErrorMessage renders an error Rime reported.
func v1ErrorMessage(e *rimepb.WebSocketError) string {
	msg := fmt.Sprintf("Rime error (%s): %s", e.GetKind(), e.GetMessage())
	if e.RequestId != nil {
		msg += fmt.Sprintf(" (request ID: %s)", e.GetRequestId())
	}
	return msg
}

// v1Error is a connection-level error Rime reported.
type v1Error struct {
	kind string
	msg  string
}

func (e *v1Error) Error() string { return e.msg }

// V1TTSConfig configures the Rime TTS service over the WebSocket v1 API.
type V1TTSConfig struct {
	// APIKey is the Rime API key. Required.
	APIKey string `validate:"required"`
	// Model is the Rime model id. Required: "coda" and "mistv3" select their
	// public endpoints unless URL is set, and any other model needs URL. The
	// model is not inferred from the URL.
	Model string
	// URL overrides the WebSocket endpoint, for a custom deployment; any path
	// is accepted. Empty uses the public endpoint of the model.
	URL string
	// Protocol is the envelope encoding: V1ProtocolBinary (the default when
	// empty) or V1ProtocolJSON.
	Protocol string
	// Speaker is the voice id; empty leaves it to Rime.
	Speaker string
	// Language for synthesis; the zero value leaves it unset. Mapped to the
	// base BCP-47 tag Rime takes ("en" for English).
	Language language.Language
	// SampleRate is the PCM rate requested from Rime and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// TimeScaleFactor scales playback speed. Values above 1.0 slow the audio
	// down; values below 1.0 speed it up. mistv2 does not take it.
	TimeScaleFactor *float64
	// TextLookaheadTokens is how many input tokens are collected before audio
	// starts (coda only).
	TextLookaheadTokens *int
	// PauseBetweenBrackets adds pauses between bracketed content (mist models
	// only).
	PauseBetweenBrackets *bool
	// PhonemizeBetweenBrackets phonemizes bracketed content (mist models only).
	PhonemizeBetweenBrackets *bool
	// SaveOovs saves out-of-vocabulary words (mist models only).
	SaveOovs *bool
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.TextTypeSentence (the zero value) sends whole sentences,
	// frames.TextTypeToken sends the tokens as they arrive and has Rime
	// accumulate them into sentences.
	TextAggregation frames.TextType
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable: a known protocol, a
// model, and an endpoint that is an absolute ws or wss URL.
func (c V1TTSConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.Protocol != "" && v1Subprotocols[c.Protocol] == "" {
		return errV1Protocol
	}
	_, err := c.endpoint()
	return err
}

// endpoint is the WebSocket URL the service connects to: the one configured, or
// the public endpoint of the model.
func (c V1TTSConfig) endpoint() (string, error) {
	if c.Model == "" {
		return "", errV1ModelRequired
	}
	endpoint := c.URL
	if endpoint == "" {
		endpoint = v1ModelURLs[c.Model]
		if endpoint == "" {
			return "", errV1URLRequired
		}
	}
	u, err := url.Parse(endpoint)
	if err != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Hostname() == "" {
		return "", errV1URLInvalid
	}
	return endpoint, nil
}

// withDefaults fills in what the caller left unset.
func (c V1TTSConfig) withDefaults() V1TTSConfig {
	c.URL, _ = c.endpoint()
	if c.Protocol == "" {
		c.Protocol = V1ProtocolBinary
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultSampleRate
	}
	if c.TextAggregation == "" {
		c.TextAggregation = frames.TextTypeSentence
	}
	return c
}

// NewV1TTS builds a Rime TTS service over the WebSocket v1 API, which carries
// protobuf envelopes in binary or proto3 JSON encoding. It holds one connection
// open for the call and gives each turn a synthesis context on it; an
// interruption cancels the turn's context without reconnecting. Text is sent as
// whole sentences by default; with frames.TextTypeToken aggregation the tokens
// are sent as they arrive and Rime accumulates them into sentences. The
// WebSocket v1 API reports no word timestamps.
//
// The configuration is validated first, and an error is returned for one that
// is not usable.
func NewV1TTS(cfg V1TTSConfig) (*tts.Base, error) {
	b, _, err := newV1TTS(cfg)
	return b, err
}

// newV1TTS builds the service and returns its synthesizer alongside it.
func newV1TTS(cfg V1TTSConfig) (*tts.Base, *v1Synthesizer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	cfg = cfg.withDefaults()
	s := &v1Synthesizer{
		cfg:            cfg,
		live:           newV1TTSSettings(cfg),
		endpointModel:  cfg.Model,
		contexts:       map[string]bool{},
		failedContexts: map[string]bool{},
	}
	b := tts.New("RimeV1TTS", s)
	s.base = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	// Spelled-out text is kept in one aggregate, so a split never breaks it.
	b.SetTextAggregator(ttstext.NewSkipTagsAggregator(cfg.TextAggregation, b.TextAggregationLanguage(), spellTags()))
	// Frames are held behind a reply until its audio has all played.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s, nil
}

// v1Synthesizer speaks over one WebSocket v1 connection held open for the call,
// with a synthesis context per turn.
type v1Synthesizer struct {
	cfg  V1TTSConfig
	live *V1TTSSettings
	host tts.AudioContextHost
	base *tts.Base
	ws   *tts.Websocket
	// endpointModel is the model the endpoint serves, fixed when the service
	// is built.
	endpointModel string

	// dialMu serializes connecting, which waits for Rime to be ready.
	dialMu sync.Mutex
	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	// mu guards everything below. It is never held across a call into the host
	// or a write.
	mu   sync.Mutex
	conn *readerConn
	// contexts are the contexts open on the current connection, from their
	// start until done, error or cancel. Audio for any other context is stale
	// and dropped.
	contexts map[string]bool
	// failedContexts are the contexts Rime closed with an error; the rest of
	// their turn is skipped.
	failedContexts map[string]bool
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *v1Synthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *v1Synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Rime model and voice synthesis is billed against.
func (s *v1Synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(s.cfg.Model), VoiceID: s.live.Voice.Or(s.cfg.Speaker)}
}

// RequiresTrailingSpace reports that each sentence is sent ending in a space,
// implementing tts.TrailingSpaceRequirer.
func (s *v1Synthesizer) RequiresTrailingSpace() bool { return true }

// warnIgnored reports a setting the model does not accept.
func warnIgnored(name, model string) {
	slog.Warn(fmt.Sprintf("rime v1 tts: %s is not applied, %s does not accept it", name, model))
}

// buildStartRequest builds the start envelope that fixes a context's settings.
// Settings the endpoint's model does not accept are left out with a warning.
func (s *v1Synthesizer) buildStartRequest(contextID string) *rimepb.WebSocketRequest {
	live := s.live
	model := s.endpointModel
	if m := live.Model.Or(""); m != model {
		slog.Warn(fmt.Sprintf("rime v1 tts: model %s is not applied, the endpoint serves %s", m, model))
	}

	audio := &rimepb.AudioParameters{
		AudioFormat:  new(v1AudioFormat),
		SamplingRate: new(int32(s.cfg.SampleRate)), //nolint:gosec // a sample rate fits
	}
	if v, ok := live.TimeScaleFactor.Value(); ok {
		if model == modelMistV2 {
			warnIgnored("time_scale_factor", model)
		} else {
			audio.TimeScaleFactor = new(float32(v))
		}
	}

	start := &rimepb.SynthesisRequest{Text: "", AudioParameters: audio}
	if s.base.IsStreamingTokens() {
		start.SplitStrategy = rimepb.SplitStrategy_SPLIT_STRATEGY_ACCUMULATE_SENTENCE.Enum()
	}
	if voice, ok := live.Voice.Value(); ok {
		start.Speaker = new(voice)
	}
	if lang, ok := live.Language.Value(); ok {
		start.Language = new(lang)
	}

	if v, ok := live.TextLookaheadTokens.Value(); ok {
		if model == modelCoda {
			tokens := int32(v) //nolint:gosec // a token count fits
			start.CodaParameters = &rimepb.CodaParameters{TextLookaheadTokens: &tokens}
		} else {
			warnIgnored("text_lookahead_tokens", model)
		}
	}

	mist := []struct {
		name  string
		value func() (bool, bool)
		set   func(p *rimepb.MistParameters, v bool)
	}{
		{
			"pause_between_brackets", live.PauseBetweenBrackets.Value,
			func(p *rimepb.MistParameters, v bool) { p.PauseBetweenBrackets = new(v) },
		},
		{
			"phonemize_between_brackets", live.PhonemizeBetweenBrackets.Value,
			func(p *rimepb.MistParameters, v bool) { p.PhonemizeBetweenBrackets = new(v) },
		},
		{"save_oovs", live.SaveOovs.Value, func(p *rimepb.MistParameters, v bool) { p.SaveOovs = new(v) }},
	}
	for _, setting := range mist {
		v, ok := setting.value()
		if !ok {
			continue
		}
		if !strings.HasPrefix(model, "mist") {
			warnIgnored(setting.name, model)
			continue
		}
		if start.MistParameters == nil {
			start.MistParameters = &rimepb.MistParameters{}
		}
		setting.set(start.MistParameters, v)
	}
	return &rimepb.WebSocketRequest{
		ContextId: contextID,
		Payload:   &rimepb.WebSocketRequest_Start{Start: start},
	}
}

// encode encodes a request in the connection's envelope encoding.
func (s *v1Synthesizer) encode(req *rimepb.WebSocketRequest) (websocket.MessageType, []byte, error) {
	if s.cfg.Protocol == V1ProtocolBinary {
		data, err := proto.Marshal(req)
		return websocket.MessageBinary, data, err
	}
	data, err := protojson.Marshal(req)
	return websocket.MessageText, data, err
}

// decode decodes a response in the connection's envelope encoding. A field the
// service does not know is skipped.
func (s *v1Synthesizer) decode(typ websocket.MessageType, data []byte) (*rimepb.WebSocketResponse, error) {
	resp := &rimepb.WebSocketResponse{}
	if s.cfg.Protocol == V1ProtocolBinary {
		if typ != websocket.MessageBinary {
			return nil, errV1TextOnBinaryWire
		}
		if err := proto.Unmarshal(data, resp); err != nil {
			return nil, err
		}
		return resp, nil
	}
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(data, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// send writes one envelope on the connection, serialized against the other
// writers, failing with errV1NotConnected when there is none.
func (s *v1Synthesizer) send(ctx context.Context, req *rimepb.WebSocketRequest) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errV1NotConnected
	}
	typ, data, err := s.encode(req)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return c.conn.Write(ctx, typ, data)
}

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *v1Synthesizer) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect opens the connection and starts the receive loop on it,
// implementing tts.WebsocketService.
func (s *v1Synthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect stops the receive loop and closes the connection, implementing
// tts.WebsocketService.
func (s *v1Synthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens the connection unless one is open, and waits for Rime
// to say it is ready to accept contexts, implementing tts.WebsocketHandler. A
// failure is reported here, with its category when Rime gave one, and leaves no
// connection behind, which a redial reads as a failed attempt.
func (s *v1Synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to rime websocket v1")
	conn, err := s.dial(ctx)
	s.mu.Lock()
	old := s.conn
	s.conn = nil
	if err == nil {
		s.conn = newReaderConn(conn)
		clear(s.contexts)
		clear(s.failedContexts)
	}
	s.mu.Unlock()
	if old != nil {
		// A connection whose reading had already ended, replaced rather than
		// closed.
		old.drop()
	}
	if err != nil {
		if conn != nil {
			_ = conn.Close(websocket.StatusNormalClosure, "")
		}
		ef := frames.NewErrorFrame(fmt.Sprintf("Error connecting: %v", err))
		ef.Err = err
		if rerr, ok := errors.AsType[*v1Error](err); ok {
			ef.Category = v1ErrorCategories[rerr.kind]
		}
		s.ws.ReportError(ctx, ef, false)
	}
	return nil
}

// dial opens the connection with the subprotocol of the configured encoding and
// reads Rime's ready event. The connection is returned alongside an error once
// it is open, for the caller to close.
func (s *v1Synthesizer) dial(ctx context.Context) (*wsutil.Conn, error) {
	subprotocol := v1Subprotocols[s.cfg.Protocol]
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	conn, err := s.ws.DialSubprotocols(ctx, s.cfg.URL, header, wsutil.DefaultReadLimit, []string{subprotocol})
	if err != nil {
		return nil, err
	}
	if conn.Subprotocol() != subprotocol {
		return conn, fmt.Errorf("%w: %s", errV1Subprotocol, subprotocol)
	}
	readyCtx, cancel := context.WithTimeout(ctx, v1ReadyTimeout)
	defer cancel()
	typ, data, err := conn.Read(readyCtx)
	if err != nil {
		return conn, err
	}
	ready, err := s.decode(typ, data)
	if err != nil {
		return conn, err
	}
	if e := ready.GetError(); e != nil {
		return conn, &v1Error{kind: e.GetKind(), msg: v1ErrorMessage(e)}
	}
	if r := ready.GetReady(); r == nil || r.GetProtocol() != v1ProtocolVersion {
		return conn, errV1Ready
	}
	return conn, nil
}

// DisconnectWebsocket closes the connection, implementing
// tts.WebsocketHandler. What the service was timing is reported as far as it
// got first: the audio it was waiting for will not come on this connection.
func (s *v1Synthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	s.base.StopAllMetrics(ctx)
	s.mu.Lock()
	c := s.conn
	s.conn = nil
	clear(s.contexts)
	clear(s.failedContexts)
	s.mu.Unlock()
	if c == nil {
		return nil
	}
	if !c.open() {
		c.drop()
		return nil
	}
	slog.Debug("disconnecting from rime websocket v1")
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil {
		s.host.PushError(ctx, fmt.Sprintf("Error disconnecting: %v", err), err, false)
	}
	return nil
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *v1Synthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && s.conn.open()
}

// OnAudioContextInterrupted cancels the Rime context when the bot is
// interrupted, implementing tts.AudioContextInterrupter.
func (s *v1Synthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	ctx = context.WithoutCancel(ctx)
	s.base.StopAllMetrics(ctx)
	s.mu.Lock()
	delete(s.failedContexts, contextID)
	open := s.contexts[contextID]
	delete(s.contexts, contextID)
	s.mu.Unlock()
	if !open {
		return
	}
	cancel := &rimepb.WebSocketRequest{
		ContextId: contextID,
		Payload:   &rimepb.WebSocketRequest_Cancel{Cancel: &rimepb.WebSocketCancel{}},
	}
	if err := s.send(ctx, cancel); err != nil {
		slog.Warn("rime v1 tts: canceling the context failed", "context", contextID, "err", err)
	}
}

// OnTurnContextCompleted closes a failed context's audio once its turn has no
// more text, implementing tts.TurnContextCompleter.
func (s *v1Synthesizer) OnTurnContextCompleted(_ context.Context, contextID string) {
	s.mu.Lock()
	failed := s.failedContexts[contextID]
	s.mu.Unlock()
	if failed {
		s.closeFailedContext(contextID)
	}
}

// closeFailedContext ends the audio of a context Rime closed with an error.
func (s *v1Synthesizer) closeFailedContext(contextID string) {
	s.mu.Lock()
	delete(s.failedContexts, contextID)
	s.mu.Unlock()
	if s.host.AudioContextAvailable(contextID) {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		s.host.AppendToAudioContext(contextID, stopped)
		s.host.RemoveAudioContext(contextID)
	}
}

// FlushAudio ends the context's input so Rime synthesizes the rest of its text,
// implementing tts.AudioFlusher. With no context named it ends the active one.
func (s *v1Synthesizer) FlushAudio(ctx context.Context, contextID string) {
	flushID := contextID
	if flushID == "" {
		flushID = s.base.ActiveAudioContextID()
	}
	s.mu.Lock()
	ok := flushID != "" && s.conn != nil && s.contexts[flushID]
	s.mu.Unlock()
	if !ok {
		return
	}
	slog.Debug("rime v1 tts: flushing audio", "context", flushID)
	end := &rimepb.WebSocketRequest{
		ContextId: flushID,
		Payload:   &rimepb.WebSocketRequest_End{End: &rimepb.WebSocketEnd{}},
	}
	if err := s.send(ctx, end); err != nil {
		slog.Warn("rime v1 tts: ending the context failed", "context", flushID, "err", err)
	}
}

// ReceiveMessages handles the connection's messages until it ends,
// implementing tts.WebsocketHandler. A message that cannot be decoded ends the
// read, as a connection that has failed.
func (s *v1Synthesizer) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errV1NotConnected
	}
	for {
		typ, data, err := c.next(ctx)
		if err != nil {
			return err
		}
		resp, err := s.decode(typ, data)
		if err != nil {
			// The connection cannot be read any further; it is dropped and
			// the receive loop reconnects.
			c.drop()
			return err
		}
		s.handleResponse(ctx, resp)
	}
}

// handleResponse handles one response. A connection-level error is reported;
// anything for a context that is no longer open is dropped, which is what keeps
// the audio of a turn the user cut off out of the next one.
func (s *v1Synthesizer) handleResponse(ctx context.Context, resp *rimepb.WebSocketResponse) {
	ctxID := resp.GetContextId()
	if e := resp.GetError(); e != nil && ctxID == "" {
		var opts []processor.ErrorOption
		if category, ok := v1ErrorCategories[e.GetKind()]; ok {
			opts = append(opts, processor.WithErrorCategory(category))
		}
		s.host.PushError(ctx, v1ErrorMessage(e), nil, false, opts...)
		return
	}
	s.mu.Lock()
	open := s.contexts[ctxID]
	s.mu.Unlock()
	if !open || !s.host.AudioContextAvailable(ctxID) {
		return
	}

	switch payload := resp.GetPayload().(type) {
	case *rimepb.WebSocketResponse_Audio:
		if len(payload.Audio) > 0 {
			f := frames.NewTTSAudioRawFrame(payload.Audio, s.SampleRate(), 1)
			f.ContextID = ctxID
			s.host.AppendToAudioContext(ctxID, f)
		}
	case *rimepb.WebSocketResponse_Done:
		s.mu.Lock()
		delete(s.contexts, ctxID)
		s.mu.Unlock()
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = ctxID
		s.host.AppendToAudioContext(ctxID, stopped)
		s.host.RemoveAudioContext(ctxID)
	case *rimepb.WebSocketResponse_Error:
		s.mu.Lock()
		delete(s.contexts, ctxID)
		s.failedContexts[ctxID] = true
		s.mu.Unlock()
		s.base.StopAllMetrics(ctx)
		// One failed context leaves the service usable for the next turn.
		category, ok := v1ErrorCategories[payload.Error.GetKind()]
		if !ok || category.IsPermanent() {
			category = errclass.Unknown
		}
		s.host.PushError(ctx, v1ErrorMessage(payload.Error), nil, false, processor.WithErrorCategory(category))
		// While the turn is still sending text, its later sentences reuse this
		// audio context; closing it now would have the base reopen an empty one
		// for them. OnTurnContextCompleted closes it instead.
		if ctxID != s.base.TurnContextID() {
			s.closeFailedContext(ctxID)
		}
	}
}

// textMessages splits prepared text into the text messages Rime v1 expects.
// Rime accumulates streamed tokens itself, so each token is one message;
// otherwise each message is one sentence, ending in a space.
func (s *v1Synthesizer) textMessages(text string) []string {
	if s.base.IsStreamingTokens() {
		return []string{text}
	}
	aggregator := ttstext.NewSkipTagsAggregator(frames.TextTypeSentence, "", spellTags())
	var sentences []string
	for _, a := range aggregator.Aggregate(text) {
		sentences = append(sentences, withTrailingSpace(a.Text))
	}
	if rest, ok := aggregator.Flush(); ok && rest.Text != "" {
		sentences = append(sentences, withTrailingSpace(rest.Text))
	}
	return sentences
}

// withTrailingSpace ends text in a space.
func withTrailingSpace(text string) string {
	if strings.HasSuffix(text, " ") {
		return text
	}
	return text + " "
}

// RunTTS sends text on the turn's context, starting the context with its first
// text. It yields nothing: the call returns once the text is sent, and the
// audio arrives on the receive loop. A context Rime failed has the rest of its
// turn skipped.
func (s *v1Synthesizer) RunTTS(
	ctx context.Context, text, contextID string, yield func(f frames.Frame) error,
) error {
	s.mu.Lock()
	failed := s.failedContexts[contextID]
	s.mu.Unlock()
	if failed {
		return nil
	}
	if !s.Connected() {
		s.Connect(ctx)
	}
	err := s.sendTurnText(ctx, text, contextID)
	if err == nil {
		return nil
	}
	// The connection's state is unknown from here, so the sentence's speech is
	// stopped and the connection is dropped and dialed again.
	if yield != nil {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		_ = yield(stopped)
	}
	s.Disconnect(ctx)
	s.Connect(ctx)
	return fmt.Errorf("rime v1 tts: %w", err)
}

// sendTurnText starts the context if this is its first text, then sends the
// text on it.
func (s *v1Synthesizer) sendTurnText(ctx context.Context, text, contextID string) error {
	s.mu.Lock()
	open := s.contexts[contextID]
	s.mu.Unlock()
	if !open {
		if err := s.send(ctx, s.buildStartRequest(contextID)); err != nil {
			return err
		}
		s.mu.Lock()
		s.contexts[contextID] = true
		s.mu.Unlock()
	}
	for _, sentence := range s.textMessages(text) {
		req := &rimepb.WebSocketRequest{
			ContextId: contextID,
			Payload:   &rimepb.WebSocketRequest_Text{Text: sentence},
		}
		if err := s.send(ctx, req); err != nil {
			return err
		}
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*v1Synthesizer)(nil)
	_ tts.WebsocketService        = (*v1Synthesizer)(nil)
	_ tts.Starter                 = (*v1Synthesizer)(nil)
	_ tts.AudioFlusher            = (*v1Synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*v1Synthesizer)(nil)
	_ tts.TurnContextCompleter    = (*v1Synthesizer)(nil)
	_ tts.Describer               = (*v1Synthesizer)(nil)
	_ tts.TrailingSpaceRequirer   = (*v1Synthesizer)(nil)
)
