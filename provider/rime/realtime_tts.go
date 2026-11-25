package rime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	ttstext "github.com/gojargo/jargo/text"
	"github.com/gojargo/jargo/text/align"
)

const (
	// defaultRealtimeURL is Rime's WebSocket JSON endpoint.
	defaultRealtimeURL = "wss://users-ws.rime.ai/ws3"
	// realtimeAudioFormat is the audio format requested on the connection: raw
	// PCM, so the audio needs no decoding.
	realtimeAudioFormat = "pcm"
)

// The Rime WebSocket messages the service reads.
const (
	realtimeMsgChunk      = "chunk"
	realtimeMsgTimestamps = "timestamps"
	realtimeMsgDone       = "done"
	realtimeMsgError      = "error"
)

// The operations the service sends besides text, under fieldOperation.
const (
	fieldOperation  = "operation"
	realtimeOpClear = "clear"
	realtimeOpEOS   = "eos"
	realtimeOpFlush = "flush"
)

// errRealtimeNotConnected is returned by a send that found no connection to
// send on.
//
//nolint:gochecknoglobals // sentinel error
var errRealtimeNotConnected = errors.New("rime tts: websocket not connected")

// spellTags are the delimiters of Rime's spell function, whose content the
// services never split into sentences.
func spellTags() []ttstext.StartEndTags { return []ttstext.StartEndTags{{Start: "spell(", End: ")"}} }

// RealtimeTTSConfig configures the Rime TTS service over the WebSocket JSON
// API. The model controls are pointers (or empty strings) and are left out of
// the connection when unset, and only the ones the chosen model accepts are
// sent at all.
type RealtimeTTSConfig struct {
	// APIKey is the Rime API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the WebSocket endpoint; empty uses
	// "wss://users-ws.rime.ai/ws3".
	URL string
	// Model is the Rime model id; empty uses "coda". It also selects which of
	// the controls below are sent: the sampling controls and the time scale
	// factor go to coda, the latency, bracket and vocabulary controls to the
	// mist models.
	Model string
	// Speaker is the voice id; empty leaves it to Rime.
	Speaker string
	// Language for synthesis; the zero value leaves it unset. Mapped to Rime's
	// language code ("eng" for English).
	Language language.Language
	// SampleRate is the PCM rate requested from Rime and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Segment is the text segmentation mode: "immediate", "bySentence" or
	// "never". Empty leaves it to Rime.
	Segment string
	// SpeedAlpha is the speech speed multiplier.
	SpeedAlpha *float64
	// RepetitionPenalty penalizes token repetition (coda only, 1.0 to 2.0).
	RepetitionPenalty *float64
	// Temperature is the sampling temperature (coda only, 0.0 to 1.0).
	Temperature *float64
	// TopP is the cumulative probability threshold (coda only, 0.0 to 1.0).
	TopP *float64
	// TimeScaleFactor scales playback speed (coda only). Values above 1.0 slow
	// the audio down; values below 1.0 speed it up.
	TimeScaleFactor *float64
	// ReduceLatency trades quality for latency (mist models only).
	ReduceLatency *bool
	// PauseBetweenBrackets adds pauses between bracketed content (mist models
	// only).
	PauseBetweenBrackets *bool
	// PhonemizeBetweenBrackets phonemizes bracketed content (mist models only).
	PhonemizeBetweenBrackets *bool
	// NoTextNormalization leaves numbers, dates and abbreviations as written
	// (mist models only).
	NoTextNormalization *bool
	// SaveOovs saves out-of-vocabulary words (mist models only).
	SaveOovs *bool
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.TextTypeSentence (the zero value) sends a sentence at a time,
	// frames.TextTypeToken streams the tokens as they arrive.
	TextAggregation frames.TextType
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
	if c.Model == "" {
		c.Model = defaultModel
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultSampleRate
	}
	if c.TextAggregation == "" {
		c.TextAggregation = frames.TextTypeSentence
	}
	return c
}

// NewRealtimeTTS builds a Rime TTS service over the WebSocket JSON API. Unlike
// NewTTS, which issues one HTTP request per sentence, it holds one connection
// open for the call and gives each turn a context on it: the sentences of a
// turn are sent on the turn's context, which is flushed once its text has all
// been sent and cleared when the user interrupts. Rime reports when each word
// is spoken, so the conversation keeps only what was heard before an
// interruption.
//
// Every setting is a query parameter of the connection, so a settings change
// reconnects. The connection is opened when the pipeline starts. When it drops
// it is restored with backoff, and when that keeps failing the service reports
// a permanent error and becomes unusable. See tts.Websocket.
func NewRealtimeTTS(cfg RealtimeTTSConfig) *tts.Base {
	b, _ := newRealtimeTTS(cfg)
	return b
}

// newRealtimeTTS builds the service and returns its synthesizer alongside it.
func newRealtimeTTS(cfg RealtimeTTSConfig) (*tts.Base, *realtimeSynthesizer) {
	cfg = cfg.withDefaults()
	s := &realtimeSynthesizer{cfg: cfg, live: newRealtimeTTSSettings(cfg)}
	b := tts.New("RimeRealtimeTTS", s)
	s.base = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	// Spelled-out text is always kept in one piece. The preferred way to use
	// Rime's spelling is to have a text transform insert the spell function for
	// the synthesizer alone; this keeps one that reaches the service by any
	// other route intact.
	b.SetTextAggregator(ttstext.NewSkipTagsAggregator(cfg.TextAggregation, b.TextAggregationLanguage(), spellTags()))
	// Frames are held behind a reply until its audio has all played.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s
}

// Pronounce has the Rime WebSocket service svc say word in text as phoneme,
// using Rime's custom pronunciation: every occurrence of word is replaced by
// the phoneme in braces, and the next message the service sends asks Rime to
// phonemize what is between brackets. See
// https://docs.rime.ai/api-reference/custom-pronunciation.
func Pronounce(svc *tts.Base, text, word, phoneme string) string {
	if s, ok := realtimeSynthesizerOf(svc); ok {
		s.setExtraMsgField("phonemizeBetweenBrackets", true)
	}
	return strings.ReplaceAll(text, word, "{"+phoneme+"}")
}

// InlineSpeed has the Rime WebSocket service svc say text at speed: the text is
// wrapped in brackets, and the next message the service sends carries the
// speed, one per bracketed span in the order they were made.
func InlineSpeed(svc *tts.Base, text string, speed float64) string {
	if s, ok := realtimeSynthesizerOf(svc); ok {
		s.mu.Lock()
		if s.extraMsgFields == nil {
			s.extraMsgFields = map[string]any{}
		}
		value := formatFloat(speed)
		if existing, _ := s.extraMsgFields["inlineSpeedAlpha"].(string); existing != "" {
			value = existing + "," + value
		}
		s.extraMsgFields["inlineSpeedAlpha"] = value
		s.mu.Unlock()
	}
	return "[" + text + "]"
}

// realtimeSynthesizerOf finds the WebSocket synthesizer behind svc.
func realtimeSynthesizerOf(svc *tts.Base) (*realtimeSynthesizer, bool) {
	if svc == nil {
		return nil, false
	}
	s, ok := svc.Synthesizer().(*realtimeSynthesizer)
	return s, ok
}

// realtimeSynthesizer speaks over one WebSocket held open for the call, with
// each turn given a context of its own on it. Every message names its context,
// so audio Rime had already generated for a turn the user cut off is told apart
// from the next turn's and dropped.
type realtimeSynthesizer struct {
	cfg  RealtimeTTSConfig
	live *TTSSettings
	host tts.AudioContextHost
	base *tts.Base
	ws   *tts.Websocket

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex
	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	// mu guards everything below. It is never held across a call into the host
	// or a write.
	mu   sync.Mutex
	conn *readerConn
	// timedContext is the context the word timings are being accumulated for,
	// and cumulativeTime where its audio so far ends. Rime times each message's
	// words from zero, so each batch is offset by the ones before it.
	timedContext   string
	cumulativeTime float64
	// extraMsgFields are added to the next text message, then cleared.
	extraMsgFields map[string]any
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *realtimeSynthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *realtimeSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Rime model and voice synthesis is billed against.
func (s *realtimeSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(s.cfg.Model), VoiceID: s.live.Voice.Or(s.cfg.Speaker)}
}

// RequiresTrailingSpace reports that each sentence is sent ending in a space,
// implementing tts.TrailingSpaceRequirer.
func (s *realtimeSynthesizer) RequiresTrailingSpace() bool { return true }

// setExtraMsgField adds a field to the next text message.
func (s *realtimeSynthesizer) setExtraMsgField(key string, value any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.extraMsgFields == nil {
		s.extraMsgFields = map[string]any{}
	}
	s.extraMsgFields[key] = value
}

// wsParams builds the query parameters of the connection from the settings as
// they stand. Only the values set are included, and of the model controls only
// the ones the model accepts. reduceLatency is sent as a capitalized boolean,
// the other mist flags as JSON booleans.
func (s *realtimeSynthesizer) wsParams() url.Values {
	q := url.Values{}
	if voice, ok := s.live.Voice.Value(); ok {
		q.Set(fieldSpeaker, voice)
	}
	model := s.live.Model.Or(s.cfg.Model)
	q.Set("modelId", model)
	q.Set("audioFormat", realtimeAudioFormat)
	q.Set("samplingRate", strconv.Itoa(s.cfg.SampleRate))
	if lang, ok := s.live.Language.Value(); ok {
		q.Set("lang", lang)
	}
	if segment, ok := s.live.Segment.Value(); ok {
		q.Set("segment", segment)
	}
	setFloatParam(q, "speedAlpha", s.live.SpeedAlpha)

	if model == modelCoda {
		setFloatParam(q, "repetition_penalty", s.live.RepetitionPenalty)
		setFloatParam(q, "temperature", s.live.Temperature)
		setFloatParam(q, "top_p", s.live.TopP)
		setFloatParam(q, "timeScaleFactor", s.live.TimeScaleFactor)
	} else {
		if v, ok := s.live.ReduceLatency.Value(); ok {
			q.Set("reduceLatency", capitalizedBool(v))
		}
		setBoolParam(q, "pauseBetweenBrackets", s.live.PauseBetweenBrackets)
		setBoolParam(q, "phonemizeBetweenBrackets", s.live.PhonemizeBetweenBrackets)
		setBoolParam(q, "noTextNormalization", s.live.NoTextNormalization)
		setBoolParam(q, "saveOovs", s.live.SaveOovs)
	}
	return q
}

// setFloatParam sets key when the setting has a value.
func setFloatParam(q url.Values, key string, o settings.Opt[float64]) {
	if v, ok := o.Value(); ok {
		q.Set(key, formatFloat(v))
	}
}

// setBoolParam sets key, as a JSON boolean, when the setting has a value.
func setBoolParam(q url.Values, key string, o settings.Opt[bool]) {
	if v, ok := o.Value(); ok {
		q.Set(key, strconv.FormatBool(v))
	}
}

// capitalizedBool writes a boolean as "True" or "False".
func capitalizedBool(v bool) string {
	if v {
		return "True"
	}
	return "False"
}

// buildMsg is the message sending text on a context, with the fields the
// markup helpers asked for since the last one.
func (s *realtimeSynthesizer) buildMsg(text, contextID string) map[string]any {
	msg := map[string]any{"text": text, "contextId": contextID}
	s.mu.Lock()
	maps.Copy(msg, s.extraMsgFields)
	s.extraMsgFields = nil
	s.mu.Unlock()
	return msg
}

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *realtimeSynthesizer) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect opens the connection and starts the receive loop on it,
// implementing tts.WebsocketService.
func (s *realtimeSynthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect stops the receive loop and closes the connection, implementing
// tts.WebsocketService.
func (s *realtimeSynthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens the connection with the settings as they stand,
// unless one is open, implementing tts.WebsocketHandler. A failure is reported
// here and leaves no connection behind, which a redial reads as a failed
// attempt.
func (s *realtimeSynthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	conn, err := s.ws.Dial(ctx, s.cfg.URL+"?"+s.wsParams().Encode(), header, wsutil.DefaultReadLimit)
	s.mu.Lock()
	old := s.conn
	s.conn = nil
	if err == nil {
		s.conn = newReaderConn(conn)
	}
	s.mu.Unlock()
	if old != nil {
		// A connection whose reading had already ended, replaced rather than
		// closed: nothing is left to say to Rime on it.
		old.drop()
	}
	if err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Error connecting: %v", err))
		ef.Err = err
		s.ws.ReportError(ctx, ef, false)
	}
	return nil
}

// DisconnectWebsocket tells Rime the stream has ended and closes the
// connection, implementing tts.WebsocketHandler. What the service was timing
// is reported as far as it got first: the audio it was waiting for will not
// come on this connection.
func (s *realtimeSynthesizer) DisconnectWebsocket(ctx context.Context) error {
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
	if !c.open() {
		// The connection is already gone: there is no one left to tell.
		c.drop()
		return nil
	}
	c.halt()
	err := s.write(ctx, c, map[string]any{fieldOperation: realtimeOpEOS})
	if err == nil {
		err = c.conn.Close(websocket.StatusNormalClosure, "")
	} else {
		_ = c.conn.CloseNow()
	}
	if err != nil {
		s.host.PushError(ctx, fmt.Sprintf("Error disconnecting: %v", err), err, false)
	}
	return nil
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *realtimeSynthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && s.conn.open()
}

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. Every
// setting is a query parameter of the connection, so any change reconnects.
func (s *realtimeSynthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 || !s.Connected() {
		return nil
	}
	s.Disconnect(ctx)
	s.Connect(ctx)
	return nil
}

// send writes one JSON message on the connection, serialized against the other
// writers, failing with errRealtimeNotConnected when there is none.
func (s *realtimeSynthesizer) send(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errRealtimeNotConnected
	}
	return s.write(ctx, c, msg)
}

// write writes one JSON message on c, serialized against the other writers.
func (s *realtimeSynthesizer) write(ctx context.Context, c *readerConn, msg map[string]any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return c.conn.Write(ctx, websocket.MessageText, payload)
}

// closeContext clears Rime's speech queue and reports what the service was
// timing as far as it got.
func (s *realtimeSynthesizer) closeContext(ctx context.Context, contextID string) {
	s.base.StopAllMetrics(ctx)
	if contextID == "" {
		return
	}
	if err := s.send(ctx, map[string]any{fieldOperation: realtimeOpClear}); err != nil {
		slog.Warn("rime tts: clearing the context failed", "context", contextID, "err", err)
	}
}

// OnAudioContextInterrupted clears Rime's speech queue when the bot is
// interrupted, implementing tts.AudioContextInterrupter.
func (s *realtimeSynthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	s.closeContext(context.WithoutCancel(ctx), contextID)
}

// OnAudioContextCompleted clears what Rime still holds for a context once its
// audio has all played, implementing tts.AudioContextCompleter.
func (s *realtimeSynthesizer) OnAudioContextCompleted(ctx context.Context, contextID string) {
	s.closeContext(ctx, contextID)
}

// FlushAudio has Rime synthesize the text it is still holding, implementing
// tts.AudioFlusher. Nothing is sent when there is no context or no connection.
func (s *realtimeSynthesizer) FlushAudio(ctx context.Context, contextID string) {
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
	slog.Debug("rime tts: flushing audio", "context", flushID)
	if err := s.send(ctx, map[string]any{fieldOperation: realtimeOpFlush}); err != nil {
		slog.Warn("rime tts: flushing audio failed", "context", flushID, "err", err)
	}
}

// calculateWordTimes turns one message's word timings into the words to
// report, each offset by where the context's audio so far ends. Blank words are
// skipped, and punctuation Rime reports as a word of its own is attached to the
// word before it.
func (s *realtimeSynthesizer) calculateWordTimes(words []string, starts, ends []float64) []align.WordTiming {
	s.mu.Lock()
	cumulative := s.cumulativeTime
	s.mu.Unlock()
	var pairs []align.WordTiming
	for i, word := range words {
		if i >= len(starts) || i >= len(ends) {
			break
		}
		if strings.TrimSpace(word) == "" {
			continue
		}
		if strings.Trim(word, ",.!?") == "" && len(pairs) > 0 {
			pairs[len(pairs)-1].Word += word
			continue
		}
		pairs = append(pairs, align.WordTiming{Word: word, Offset: starts[i] + cumulative})
	}
	return pairs
}

// realtimeMessage is the subset of a Rime message the service reads.
type realtimeMessage struct {
	Type           string `json:"type"`
	ContextID      string `json:"contextId"` //nolint:tagliatelle // vendor API field
	Data           string `json:"data"`
	Message        string `json:"message"`
	WordTimestamps struct {
		Words []string  `json:"words"`
		Start []float64 `json:"start"`
		End   []float64 `json:"end"`
	} `json:"word_timestamps"` //nolint:tagliatelle // vendor API field
}

// ReceiveMessages handles the connection's messages until it ends,
// implementing tts.WebsocketHandler.
func (s *realtimeSynthesizer) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errRealtimeNotConnected
	}
	for {
		_, data, err := c.next(ctx)
		if err != nil {
			return err
		}
		var msg realtimeMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			slog.Warn("rime tts received non-JSON message")
			continue
		}
		s.handleMessage(ctx, &msg)
	}
}

// handleMessage handles one message for the context it names. A message for a
// context that is no longer open is dropped, which is what keeps the audio of
// a turn the user cut off out of the next one.
func (s *realtimeSynthesizer) handleMessage(ctx context.Context, msg *realtimeMessage) {
	contextID := msg.ContextID
	if !s.host.AudioContextAvailable(contextID) {
		return
	}
	switch msg.Type {
	case realtimeMsgChunk:
		audio, err := base64.StdEncoding.DecodeString(msg.Data)
		if err != nil {
			slog.Warn("rime tts sent audio that could not be decoded", "err", err)
			return
		}
		f := frames.NewTTSAudioRawFrame(audio, s.SampleRate(), 1)
		f.ContextID = contextID
		s.host.AppendToAudioContext(contextID, f)
	case realtimeMsgTimestamps:
		ts := msg.WordTimestamps
		if len(ts.Words) == 0 || len(ts.Start) == 0 {
			return
		}
		pairs := s.calculateWordTimes(ts.Words, ts.Start, ts.End)
		if len(pairs) == 0 {
			return
		}
		s.host.AddWordTimestamps(contextID, pairs, tts.WordTimingOptions{})
		s.mu.Lock()
		s.cumulativeTime += ts.End[len(ts.End)-1]
		slog.Debug("rime tts: updated cumulative time", "cumulative_time", s.cumulativeTime)
		s.mu.Unlock()
	case realtimeMsgDone:
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		s.host.AppendToAudioContext(contextID, stopped)
		s.host.RemoveAudioContext(contextID)
	case realtimeMsgError:
		_ = s.host.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
		s.base.StopAllMetrics(ctx)
		s.host.PushError(ctx, "Error: "+msg.Message, nil, false)
		s.host.ResetActiveAudioContext()
	}
}

// startTimedContext starts the word timings of a context from zero the first
// time it is given text.
func (s *realtimeSynthesizer) startTimedContext(contextID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.timedContext != contextID {
		s.timedContext = contextID
		s.cumulativeTime = 0
	}
}

// RunTTS sends one sentence on the turn's context. It yields nothing: the call
// returns once the text is sent, and the audio arrives on the receive loop,
// which appends it to the context it belongs to.
func (s *realtimeSynthesizer) RunTTS(
	ctx context.Context, text, contextID string, yield func(f frames.Frame) error,
) error {
	if !s.Connected() {
		s.Connect(ctx)
	}
	s.startTimedContext(contextID)
	err := s.send(ctx, s.buildMsg(text, contextID))
	if err == nil {
		tts.StartUsageMetrics(ctx, text)
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
	return fmt.Errorf("rime tts: %w", err)
}

// RunTTSTimed sends the sentence like RunTTS, implementing tts.WordTimestamps.
// The timings arrive on the receive loop with the audio, so the word callback
// is not used here; implementing the interface is what puts the base on the
// word-aligned path.
func (s *realtimeSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	yield func(f frames.Frame) error,
	_ func(words []align.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, yield)
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*realtimeSynthesizer)(nil)
	_ tts.WebsocketService        = (*realtimeSynthesizer)(nil)
	_ tts.Starter                 = (*realtimeSynthesizer)(nil)
	_ tts.AudioFlusher            = (*realtimeSynthesizer)(nil)
	_ tts.AudioContextInterrupter = (*realtimeSynthesizer)(nil)
	_ tts.AudioContextCompleter   = (*realtimeSynthesizer)(nil)
	_ tts.WordTimestamps          = (*realtimeSynthesizer)(nil)
	_ tts.Describer               = (*realtimeSynthesizer)(nil)
	_ tts.TrailingSpaceRequirer   = (*realtimeSynthesizer)(nil)
)
