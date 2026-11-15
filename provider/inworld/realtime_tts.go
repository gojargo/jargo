package inworld

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
	ttstext "github.com/gojargo/jargo/utils/text"
	"github.com/google/uuid"
)

const (
	// defaultRealtimeURL is Inworld's bidirectional streaming endpoint.
	defaultRealtimeURL = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
	// defaultTimestampTransportStrategy is how word timestamps travel relative
	// to the audio they time when none is configured.
	defaultTimestampTransportStrategy = "ASYNC"
	// defaultMaxBufferDelayMs is how long the server buffers text before
	// synthesizing it when auto mode is off, unless configured otherwise.
	defaultMaxBufferDelayMs = 3000
	// defaultBufferCharThreshold is how many buffered characters trigger
	// synthesis, unless configured otherwise.
	defaultBufferCharThreshold = 250
	// realtimeKeepaliveInterval is how often an idle connection is sent empty
	// text, so the server does not close it.
	realtimeKeepaliveInterval = 60 * time.Second
	// timestampTypeWord asks for one timestamp per word.
	timestampTypeWord = "WORD"
	// userAgent identifies the client to Inworld.
	userAgent = "jargo"
	// fieldContextID names the context a message belongs to.
	fieldContextID = "contextId"
	// statusNotFound is the status code of a message naming a context the
	// server no longer has.
	statusNotFound = 5
)

// breakTag matches a pause tag, which Inworld reports as a word of its own.
//
//nolint:gochecknoglobals // compiled once, read-only
var breakTag = regexp.MustCompile(`(?i)<break\b[^>]*>`)

// errWebsocketUnavailable is returned for a sentence when there is still no
// connection after connecting again.
//
//nolint:gochecknoglobals // sentinel error
var errWebsocketUnavailable = errors.New("inworld tts: websocket unavailable")

// errNotConnected is returned by a send that found no connection to send on.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("inworld tts: websocket not connected")

// RealtimeTTSConfig configures the Inworld AI WebSocket TTS service.
type RealtimeTTSConfig struct {
	// APIKey is the Inworld API key (a base64 Basic credential). Required.
	APIKey string `validate:"required"`
	// URL overrides the WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// VoiceID is the voice id; empty uses "Ashley".
	VoiceID string
	// Model is the model id; empty uses "inworld-tts-2".
	Model string
	// Language for synthesis; the zero value leaves it unset. Mapped to Inworld's
	// BCP-47 language tag.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 48 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "PCM".
	Encoding string
	// SpeakingRate multiplies the speaking rate; nil omits it.
	SpeakingRate *float64
	// Temperature guides synthesis variability; nil omits it.
	Temperature *float64
	// DeliveryMode trades stability against expressiveness ("STABLE", "BALANCED",
	// "CREATIVE"; inworld-tts-2 only); empty omits it.
	DeliveryMode string
	// AutoMode lets the server decide when to flush the text it buffers, for the
	// lowest latency at full quality. nil leaves it on. With
	// frames.TextTypeToken aggregation and an inworld-tts-2 model, it also has the
	// server find the sentence boundaries in the raw token stream.
	AutoMode *bool
	// ApplyTextNormalization controls spoken-form normalization ("ON", "OFF");
	// empty omits it.
	ApplyTextNormalization string
	// TimestampTransportStrategy is how word timestamps travel relative to the
	// audio they time ("ASYNC" or "SYNC"); empty uses "ASYNC".
	TimestampTransportStrategy string `validate:"omitempty,oneof=ASYNC SYNC"`
	// MaxBufferDelayMs is the longest the server buffers text before
	// synthesizing it when auto mode is off; 0 uses 3000.
	MaxBufferDelayMs int
	// BufferCharThreshold is how many buffered characters trigger synthesis.
	// When the server finds the sentence boundaries it also cuts sentences longer
	// than it. 0 uses 250, or Inworld's own default when the server finds the
	// sentence boundaries.
	BufferCharThreshold int
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.TextTypeSentence (the zero value) sends a sentence at a time,
	// frames.TextTypeToken streams the tokens as they arrive.
	TextAggregation frames.TextType
	// AppendTrailingSpace appends a space to each sentence sent; nil leaves it
	// on.
	AppendTrailingSpace *bool
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c RealtimeTTSConfig) Validate() error { return validate.Struct(c) }

// withDefaults fills in what the caller left unset.
func (c RealtimeTTSConfig) withDefaults() RealtimeTTSConfig {
	if c.URL == "" {
		c.URL = defaultRealtimeURL
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
	if c.Encoding == "" {
		c.Encoding = defaultEncoding
	}
	if c.AutoMode == nil {
		c.AutoMode = new(true)
	}
	if c.TimestampTransportStrategy == "" {
		c.TimestampTransportStrategy = defaultTimestampTransportStrategy
	}
	if c.TextAggregation == "" {
		c.TextAggregation = frames.TextTypeSentence
	}
	if c.AppendTrailingSpace == nil {
		c.AppendTrailingSpace = new(true)
	}
	return c
}

// NewRealtimeTTS builds an Inworld AI TTS service over the bidirectional
// WebSocket API. Unlike NewTTS, which issues one HTTP request per sentence, it
// holds one connection open for the call and one synthesis context open for
// each turn: the context is created with the turn's first text, flushed and
// closed once the turn's text has all been sent, and canceled when the user
// interrupts. Inworld reports when each word is spoken, so the conversation
// keeps only what was heard before an interruption.
//
// The connection is opened when the pipeline starts and kept alive while idle.
// When it drops it is restored with backoff, and when that keeps failing the
// service reports a permanent error and becomes unusable. See tts.Websocket.
func NewRealtimeTTS(cfg RealtimeTTSConfig) *tts.Base {
	b, _ := newRealtimeTTS(cfg)
	return b
}

// newRealtimeTTS builds the service and returns its synthesizer alongside it.
func newRealtimeTTS(cfg RealtimeTTSConfig) (*tts.Base, *realtimeSynthesizer) {
	cfg = cfg.withDefaults()
	s := &realtimeSynthesizer{
		cfg:                    cfg,
		live:                   newTTSSettings(cfg),
		contextTexts:           map[string]string{},
		contextsWithTimestamps: map[string]bool{},
		contextsCreated:        map[string]bool{},
	}
	s.write = func(ctx context.Context, conn *wsutil.Conn, payload []byte) error {
		return conn.Write(ctx, websocket.MessageText, payload)
	}
	b := tts.New("InworldRealtimeTTS", s)
	s.svc = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	if cfg.TextAggregation == frames.TextTypeToken {
		b.SetTextAggregator(ttstext.NewTokenAggregator())
	}
	// Frames are held behind a reply until its audio has all played.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s
}

// realtimeService is the part of the TTS service the synthesizer asks about the
// audio contexts and the text aggregation. tts.Base implements it.
type realtimeService interface {
	Name() string
	ActiveAudioContextID() string
	AudioContextIDs() []string
	AppendsToContext(contextID string) bool
	IsStreamingTokens() bool
	StopAllMetrics(ctx context.Context)
}

// realtimeSynthesizer speaks over one WebSocket held open for the call, with a
// server-side context per turn. Every message names its context, so audio the
// server had already generated for a turn the user cut off is told apart from
// the next turn's and dropped.
type realtimeSynthesizer struct {
	cfg  RealtimeTTSConfig
	live *TTSSettings
	host tts.AudioContextHost
	svc  realtimeService
	ws   *tts.Websocket

	// write sends one message on a connection.
	write func(ctx context.Context, conn *wsutil.Conn, payload []byte) error

	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	// mu guards everything below. It is never held across a call into the host
	// or a write.
	mu   sync.Mutex
	conn *wsutil.Conn
	// closed records that the read on conn ended: the connection is gone.
	closed bool
	// cumulativeTime keeps the timestamps of a turn increasing across the
	// generations the server splits it into. In auto mode the server decides
	// when a generation ends and starts the next one's timestamps from zero, so
	// each generation's are offset by where the previous ones ended, as a
	// flushCompleted message reports.
	cumulativeTime float64
	// generationEndTime is where the last word of the current generation ends.
	generationEndTime float64
	// contextTexts is the text sent on each context, and contextsWithTimestamps
	// the contexts the server has timed words for. A context that was never
	// timed has its whole text reported as spoken when it ends: without timings
	// there is no telling which part was heard.
	contextTexts           map[string]string
	contextsWithTimestamps map[string]bool
	// contextsCreated holds the contexts created on the server over this
	// connection. The first text a context is given creates it.
	contextsCreated map[string]bool
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *realtimeSynthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *realtimeSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Inworld model and voice synthesis is billed against.
func (s *realtimeSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(s.cfg.Model), VoiceID: s.live.Voice.Or(s.cfg.VoiceID)}
}

// ExpressionAdapter implements tts.ExpressionAdapterProvider: audio markup on
// TTS-2 models, and none on the older ones.
func (s *realtimeSynthesizer) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	return tts2ExpressionAdapter(s.live.Model.Or(""))
}

// FormatPronunciation implements tts.PronunciationFormatter.
func (s *realtimeSynthesizer) FormatPronunciation(word, ipa string) (string, bool) {
	return FormatPronunciation(word, ipa)
}

// RequiresTrailingSpace reports whether each sentence is sent ending in a space,
// implementing tts.TrailingSpaceRequirer.
func (s *realtimeSynthesizer) RequiresTrailingSpace() bool { return *s.cfg.AppendTrailingSpace }

// supportsSentenceBoundary reports whether the model serves the auto-mode
// strategy that finds sentence boundaries in a raw token stream: only the
// inworld-tts-2 models do.
func supportsSentenceBoundary(model string) bool {
	return strings.HasPrefix(model, "inworld-tts-2")
}

// Start connects when the pipeline starts, implementing tts.Starter.
func (s *realtimeSynthesizer) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect opens the connection, then starts the receive loop and the
// keepalive, implementing tts.WebsocketService.
func (s *realtimeSynthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
		s.ws.CreateKeepaliveTask(ctx, realtimeKeepaliveInterval, s)
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
	if s.Connected() {
		return nil
	}
	requestID := uuid.NewString()
	slog.Debug("connecting to inworld websocket tts", "request_id", requestID)
	header := http.Header{}
	header.Set("Authorization", "Basic "+s.cfg.APIKey)
	header.Set("X-User-Agent", userAgent)
	header.Set("X-Request-Id", requestID)
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

// DisconnectWebsocket cancels the contexts still open on the connection and
// closes it, implementing tts.WebsocketHandler. What the service was timing is
// reported as far as it got first: the audio it was waiting for will not come
// on this connection.
func (s *realtimeSynthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.svc.StopAllMetrics(ctx)
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn != nil {
		slog.Debug("disconnecting from inworld websocket tts")
		for _, id := range s.svc.AudioContextIDs() {
			if err := s.sendCancelContext(ctx, id); err != nil {
				// The connection is already gone; it still needs closing.
				slog.Debug("inworld tts: unable to close contexts before disconnecting", "err", err)
				break
			}
		}
		if err := conn.Close(websocket.StatusNormalClosure, ""); err != nil && !closedConnection(err) {
			s.host.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
		}
		slog.Debug("disconnected from inworld websocket tts")
	}
	s.mu.Lock()
	s.conn, s.closed = nil, false
	s.cumulativeTime, s.generationEndTime = 0, 0
	clear(s.contextTexts)
	clear(s.contextsWithTimestamps)
	clear(s.contextsCreated)
	s.mu.Unlock()
	return nil
}

// closedConnection reports whether err is a close failing because the
// connection was already closed.
func closedConnection(err error) bool {
	return websocket.CloseStatus(err) != -1 || errors.Is(err, net.ErrClosed)
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *realtimeSynthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.closed
}

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. Every
// setting is sent when a context is created, so the service reconnects and the
// next context is created with the new values.
func (s *realtimeSynthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.Disconnect(ctx)
	s.Connect(ctx)
	return nil
}

// FlushAudio has the server synthesize the text it is still buffering for the
// context, or for the active one when none is named, implementing
// tts.AudioFlusher. The context stays open for more text: it is closed only at
// the end of the turn, on an interruption, or on a disconnect.
func (s *realtimeSynthesizer) FlushAudio(ctx context.Context, contextID string) {
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
	slog.Debug("inworld tts: flushing audio", "context", flushID)
	if err := s.send(ctx, map[string]any{"flush_context": map[string]any{}, fieldContextID: flushID}); err != nil {
		s.host.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
	}
}

// resetGenerationTiming starts the timestamps of a new context from zero.
func (s *realtimeSynthesizer) resetGenerationTiming() {
	s.mu.Lock()
	s.cumulativeTime, s.generationEndTime = 0, 0
	s.mu.Unlock()
}

// timestampInfo carries the word timings of an audio chunk.
type timestampInfo struct {
	WordAlignment struct {
		Words                []string  `json:"words"`
		WordStartTimeSeconds []float64 `json:"wordStartTimeSeconds"` //nolint:tagliatelle // vendor API field
		WordEndTimeSeconds   []float64 `json:"wordEndTimeSeconds"`   //nolint:tagliatelle // vendor API field
	} `json:"wordAlignment"` //nolint:tagliatelle // vendor API field
}

// calculateWordTimes turns one message's word alignment into the words to
// report, each offset by the generations before it so a turn's timestamps keep
// increasing, and records where the generation's last word ends. A pause tag is
// taken out of the word it is reported as.
func (s *realtimeSynthesizer) calculateWordTimes(info *timestampInfo) []uctx.WordTiming {
	alignment := info.WordAlignment
	words, starts, ends := alignment.Words, alignment.WordStartTimeSeconds, alignment.WordEndTimeSeconds
	if len(words) == 0 || len(starts) == 0 || len(words) != len(starts) {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]uctx.WordTiming, 0, len(words))
	for i, word := range words {
		out = append(out, uctx.WordTiming{Word: breakTag.ReplaceAllString(word, ""), Offset: s.cumulativeTime + starts[i]})
	}
	if len(ends) > 0 {
		s.generationEndTime = s.cumulativeTime + ends[len(ends)-1]
	}
	return out
}

// closeContext ends a server-side context: canceling it discards the text the
// server has not synthesized yet, closing it lets that text be spoken first.
func (s *realtimeSynthesizer) closeContext(ctx context.Context, contextID string, cancel bool) {
	if contextID == "" {
		return
	}
	s.mu.Lock()
	connected := s.conn != nil
	s.mu.Unlock()
	if !connected {
		return
	}
	action := "Closing"
	if cancel {
		action = "Canceling"
	}
	slog.Info("inworld tts: "+strings.ToLower(action)+" context", "context", contextID)
	var err error
	if cancel {
		err = s.sendCancelContext(ctx, contextID)
	} else {
		err = s.send(ctx, map[string]any{"close_context": map[string]any{}, fieldContextID: contextID})
	}
	if err != nil {
		s.host.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
	}
}

// OnTurnContextCompleted closes the server-side context once the turn's text
// has all been sent, implementing tts.TurnContextCompleter, so the server marks
// it closed right after its last audio. A turn whose text never reached the
// service opened no context, and has nothing to close.
func (s *realtimeSynthesizer) OnTurnContextCompleted(ctx context.Context, contextID string) {
	if contextID == "" || !s.host.AudioContextAvailable(contextID) {
		return
	}
	s.closeContext(ctx, contextID, false)
}

// OnAudioContextInterrupted cancels a context nobody is listening to any more,
// implementing tts.AudioContextInterrupter. A context that was never timed has
// its text reported first.
func (s *realtimeSynthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	ctx = context.WithoutCancel(ctx)
	s.maybePushFallbackText(ctx, contextID)
	s.closeContext(ctx, contextID, true)
	s.mu.Lock()
	delete(s.contextsCreated, contextID)
	s.mu.Unlock()
}

// maybePushFallbackText reports a context's whole text as spoken when the
// server timed none of its words, so the conversation still records what the
// bot said. Without timings the whole text is committed even on an
// interruption, since there is no telling which part was heard.
func (s *realtimeSynthesizer) maybePushFallbackText(ctx context.Context, contextID string) {
	if contextID == "" {
		return
	}
	s.mu.Lock()
	hadTimestamps := s.contextsWithTimestamps[contextID]
	text := strings.TrimSpace(s.contextTexts[contextID])
	delete(s.contextTexts, contextID)
	delete(s.contextsWithTimestamps, contextID)
	s.mu.Unlock()
	if hadTimestamps || text == "" {
		return
	}
	slog.Debug("inworld tts: no timestamps for context, pushing fallback", "context", contextID, "text", text)
	fallback := frames.NewTTSTextFrame(text, frames.TextTypeSentence)
	fallback.ContextID = contextID
	fallback.AppendToContext = s.svc.AppendsToContext(contextID)
	_ = s.host.PushFrame(ctx, fallback, processor.Downstream)
}

// ReceiveMessages handles the connection's messages until it ends,
// implementing tts.WebsocketHandler.
func (s *realtimeSynthesizer) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return errNotConnected
	}
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			s.mu.Lock()
			if s.conn == conn {
				s.closed = true
			}
			s.mu.Unlock()
			return err
		}
		s.handleMessage(ctx, data)
	}
}

// realtimeMessage is the subset of a server message the service reads.
type realtimeMessage struct {
	Result struct {
		ContextID      string `json:"contextId"`  //nolint:tagliatelle // vendor API field
		ContextIDSnake string `json:"context_id"` //nolint:tagliatelle // vendor API field
		Status         struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"status"`
		ContextCreated json.RawMessage `json:"contextCreated"` //nolint:tagliatelle // vendor API field
		AudioChunk     struct {
			AudioContent  string         `json:"audioContent"`  //nolint:tagliatelle // vendor API field
			TimestampInfo *timestampInfo `json:"timestampInfo"` //nolint:tagliatelle // vendor API field
		} `json:"audioChunk"` //nolint:tagliatelle // vendor API field
		FlushCompleted json.RawMessage `json:"flushCompleted"` //nolint:tagliatelle // vendor API field
		ContextClosed  json.RawMessage `json:"contextClosed"`  //nolint:tagliatelle // vendor API field
	} `json:"result"`
	Error json.RawMessage `json:"error"`
}

// handleMessage handles one server message: audio and word timings for the
// context it names, the end of a generation, or the end of a context.
//
// Audio for a context that is no longer open is dropped, which is what keeps
// the audio of a turn the user cut off out of the next one.
func (s *realtimeSynthesizer) handleMessage(ctx context.Context, data []byte) {
	var msg realtimeMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		slog.Warn("inworld tts received non-JSON message")
		return
	}
	result := &msg.Result
	ctxID := result.ContextID
	if ctxID == "" {
		ctxID = result.ContextIDSnake
	}

	if result.Status.Code != 0 {
		s.handleStatus(ctx, ctxID, result.Status.Code, result.Status.Message)
		return
	}

	if len(msg.Error) > 0 && string(msg.Error) != "null" {
		s.host.PushError(ctx, errorText(msg.Error), nil, false)
		return
	}

	if len(result.ContextCreated) > 0 {
		slog.Debug("inworld tts: context created on server", "context", ctxID)
	}

	if audioB64 := result.AudioChunk.AudioContent; audioB64 != "" {
		s.handleAudio(ctxID, audioB64)
	}

	// The timestamps travel inside the audio chunk.
	if info := result.AudioChunk.TimestampInfo; info != nil {
		s.handleTimestamps(ctxID, info)
	}

	if len(result.FlushCompleted) > 0 {
		// The end of a generation: the next one's timestamps start from zero.
		s.mu.Lock()
		s.cumulativeTime = s.generationEndTime
		s.mu.Unlock()
	}

	if len(result.ContextClosed) > 0 {
		s.handleContextClosed(ctx, ctxID)
	}
}

// handleStatus handles a message reporting an error. A message about a context
// that has expired, or one sent with no context open, is a keepalive that went
// out at the wrong moment and is benign; anything else is reported.
func (s *realtimeSynthesizer) handleStatus(ctx context.Context, ctxID string, code int, message string) {
	if message == "" {
		message = "Unknown error"
	}
	lower := strings.ToLower(message)
	switch {
	case code == statusNotFound && strings.Contains(lower, "not found"):
		// A keepalive sent after its context expired.
		slog.Debug("inworld tts: context not found", "context", ctxID)
	case strings.Contains(lower, "context_id is required") || strings.Contains(lower, "no open context"):
		// A keepalive sent without an open context.
		slog.Debug("inworld tts: contextless message rejected (benign)", "error", message)
	default:
		s.host.PushError(ctx, "Inworld API error: "+message, nil, false)
	}
}

// handleAudio appends a chunk of audio to the context it belongs to, stripping
// any WAV header.
func (s *realtimeSynthesizer) handleAudio(ctxID, audioB64 string) {
	audio, err := base64.StdEncoding.DecodeString(audioB64)
	if err != nil {
		slog.Warn("inworld tts sent audio that could not be decoded", "err", err)
		return
	}
	if len(audio) > wavHeaderSize && strings.HasPrefix(string(audio[:4]), "RIFF") {
		audio = audio[wavHeaderSize:]
	}
	if ctxID == "" {
		return
	}
	f := frames.NewTTSAudioRawFrame(audio, s.SampleRate(), 1)
	f.ContextID = ctxID
	s.host.AppendToAudioContext(ctxID, f)
}

// handleTimestamps reports the words a chunk times, and records that its
// context was timed.
func (s *realtimeSynthesizer) handleTimestamps(ctxID string, info *timestampInfo) {
	words := s.calculateWordTimes(info)
	if len(words) == 0 {
		return
	}
	if ctxID != "" {
		s.mu.Lock()
		s.contextsWithTimestamps[ctxID] = true
		s.mu.Unlock()
	}
	s.host.AddWordTimestamps(ctxID, words, tts.WordTimingOptions{PreMergeTokens: true})
}

// handleContextClosed ends a context the server has closed: its stop frame
// follows the audio already queued for it.
func (s *realtimeSynthesizer) handleContextClosed(ctx context.Context, ctxID string) {
	slog.Debug("inworld tts: context closed on server", "context", ctxID)
	s.maybePushFallbackText(ctx, ctxID)
	s.mu.Lock()
	delete(s.contextsCreated, ctxID)
	s.mu.Unlock()
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = ctxID
	s.host.AppendToAudioContext(ctxID, stopped)
	s.host.RemoveAudioContext(ctxID)
}

// errorText renders a message's error the way it is reported: a string as it
// is, anything else as the JSON it arrived as.
func errorText(raw json.RawMessage) string {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	return string(raw)
}

// KeepaliveReady reports whether the connection is open, implementing
// tts.KeepaliveHandler.
func (s *realtimeSynthesizer) KeepaliveReady() bool { return s.Connected() }

// SendKeepalive sends empty text, on the active context if there is one,
// implementing tts.KeepaliveHandler.
func (s *realtimeSynthesizer) SendKeepalive(ctx context.Context) error {
	msg := map[string]any{"send_text": map[string]any{fieldText: ""}}
	if id := s.svc.ActiveAudioContextID(); id != "" {
		msg[fieldContextID] = id
	}
	return s.send(ctx, msg)
}

// sendContext creates a context on the server, with the settings every request
// on it is synthesized with.
func (s *realtimeSynthesizer) sendContext(ctx context.Context, contextID string) error {
	audioConfig := map[string]any{
		"audioEncoding":   s.cfg.Encoding,
		"sampleRateHertz": s.cfg.SampleRate,
	}
	if rate, ok := s.live.SpeakingRate.Value(); ok {
		audioConfig["speakingRate"] = rate
	}
	model := s.live.Model.Or("")
	createConfig := map[string]any{
		fieldVoiceID:     s.live.Voice.Or(""),
		fieldModelID:     model,
		fieldAudioConfig: audioConfig,
	}
	if temperature, ok := s.live.Temperature.Value(); ok {
		createConfig["temperature"] = temperature
	}
	if mode, ok := s.live.DeliveryMode.Value(); ok {
		createConfig["deliveryMode"] = mode
	}
	if lang, ok := s.live.Language.Value(); ok {
		createConfig["language"] = lang
	}
	if s.cfg.ApplyTextNormalization != "" {
		createConfig["applyTextNormalization"] = s.cfg.ApplyTextNormalization
	}
	autoMode := *s.cfg.AutoMode
	createConfig["autoMode"] = autoMode
	sentenceBoundary := autoMode && s.svc.IsStreamingTokens() && supportsSentenceBoundary(model)
	if sentenceBoundary {
		// Let the server find sentence boundaries in the raw token stream.
		createConfig["autoModeStrategy"] = "SENTENCE_BOUNDARY"
	}
	createConfig["timestampTransportStrategy"] = s.cfg.TimestampTransportStrategy

	// Buffer settings for timely audio generation: what was configured, or
	// defaults that work well for streaming LLM output.
	createConfig["maxBufferDelayMs"] = cmp.Or(s.cfg.MaxBufferDelayMs, defaultMaxBufferDelayMs)
	// With the server finding sentence boundaries the threshold also cuts long
	// sentences, so only a configured value is sent and the server default
	// applies otherwise.
	if s.cfg.BufferCharThreshold != 0 || !sentenceBoundary {
		createConfig["bufferCharThreshold"] = cmp.Or(s.cfg.BufferCharThreshold, defaultBufferCharThreshold)
	}
	createConfig["timestampType"] = timestampTypeWord

	return s.send(ctx, map[string]any{"create": createConfig, fieldContextID: contextID})
}

// sendText sends text on a context.
func (s *realtimeSynthesizer) sendText(ctx context.Context, contextID, text string) error {
	return s.send(ctx, map[string]any{"send_text": map[string]any{fieldText: text}, fieldContextID: contextID})
}

// sendCancelContext stops a context, discarding text the server has not
// synthesized yet.
func (s *realtimeSynthesizer) sendCancelContext(ctx context.Context, contextID string) error {
	return s.send(ctx, map[string]any{"cancel_context": map[string]any{}, fieldContextID: contextID})
}

// send writes one JSON message on the connection, serialized against the other
// writers, failing with errNotConnected when there is none.
func (s *realtimeSynthesizer) send(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return errNotConnected
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.write(ctx, conn, payload)
}

// RunTTS sends one sentence on the turn's context, creating the context on the
// server with its first text. It yields nothing: the call returns once the text
// is sent, and the audio arrives on the receive loop, which appends it to the
// context it belongs to.
func (s *realtimeSynthesizer) RunTTS(
	ctx context.Context, text, contextID string, yield func(f frames.Frame) error,
) error {
	slog.Debug("inworld tts: generating websocket TTS", "text", text, "context", contextID)
	if !s.Connected() {
		if s.ws.ReceiveTaskRunning() {
			// The receive loop reconnects. A connection opened here would race
			// it, leaving this turn's audio on a connection nothing reads.
			s.host.PushError(ctx, s.svc.Name()+": websocket reconnecting, skipping TTS", nil, false)
			return nil
		}
		s.Connect(ctx)
	}
	if !s.Connected() {
		slog.Warn("inworld tts: websocket unavailable after connect attempt, skipping TTS")
		return errWebsocketUnavailable
	}

	if err := s.sendTurnText(ctx, text, contextID); err != nil {
		if yield != nil {
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = contextID
			_ = yield(stopped)
		}
		return fmt.Errorf("inworld tts: %w", err)
	}
	return nil
}

// sendTurnText creates the context on the server if this is its first text,
// then sends the text on it.
func (s *realtimeSynthesizer) sendTurnText(ctx context.Context, text, contextID string) error {
	s.mu.Lock()
	created := s.contextsCreated[contextID]
	s.mu.Unlock()
	if !created {
		s.resetGenerationTiming()
		s.mu.Lock()
		s.contextsCreated[contextID] = true
		s.mu.Unlock()
		if err := s.sendContext(ctx, contextID); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.contextTexts[contextID] += text
	s.mu.Unlock()
	if err := s.sendText(ctx, contextID, text); err != nil {
		return err
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// RunTTSTimed sends the sentence like RunTTS, implementing tts.WordTimestamps.
// The timings arrive on the receive loop with the audio, so the word callback
// is not used here; implementing the interface is what puts the base on the
// word-aligned path.
func (s *realtimeSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	yield func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, yield)
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*realtimeSynthesizer)(nil)
	_ tts.WebsocketService        = (*realtimeSynthesizer)(nil)
	_ tts.KeepaliveHandler        = (*realtimeSynthesizer)(nil)
	_ tts.Starter                 = (*realtimeSynthesizer)(nil)
	_ tts.AudioFlusher            = (*realtimeSynthesizer)(nil)
	_ tts.TurnContextCompleter    = (*realtimeSynthesizer)(nil)
	_ tts.AudioContextInterrupter = (*realtimeSynthesizer)(nil)
	_ tts.WordTimestamps          = (*realtimeSynthesizer)(nil)
	_ tts.Describer               = (*realtimeSynthesizer)(nil)
	_ tts.TrailingSpaceRequirer   = (*realtimeSynthesizer)(nil)
	_ tts.PronunciationFormatter  = (*realtimeSynthesizer)(nil)
)
