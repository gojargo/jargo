package heygen

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/errclass"
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
	// defaultURL is the HeyGen WebSocket base.
	defaultURL = "wss://api.heygen.com"
	// ttsPath is the multi-stream-input endpoint, which carries a context per
	// turn on one connection.
	ttsPath = "/v3/models/audio/tts/multi-stream-input"
	// keepaliveInterval is how often an idle connection is kept open.
	keepaliveInterval = 10 * time.Second
	// keepaliveContextID names a context the service never opens. A keepalive
	// sent while no context is open closes it, which resets HeyGen's inactivity
	// timer and changes nothing else.
	keepaliveContextID = "jargo-keepalive"
)

// Message keys on the multi-stream-input protocol.
const (
	keyText         = "text"
	keyContextID    = "context_id"
	keyCloseContext = "close_context"
	keyFlush        = "flush"
)

// errorCategories are the categories of the errors HeyGen sends by code. An
// error of any other code is left for the base to categorize.
//
//nolint:gochecknoglobals // read-only lookup table
var errorCategories = map[string]errclass.Category{
	"rate_limit_exceeded":   errclass.RateLimit,
	"usage_limit_reached":   errclass.Quota,
	"insufficient_credit":   errclass.Quota,
	"plan_upgrade_required": errclass.Quota,
}

// errNotConnected is the failure of text sent with no connection to send it on.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("heygen: websocket not connected")

// TTSConfig configures the HeyGen Voice WebSocket TTS service.
type TTSConfig struct {
	// APIKey is the HeyGen API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the WebSocket base; empty uses "wss://api.heygen.com".
	URL string
	// SampleRate is the PCM rate emitted downstream; 0 uses 24 kHz. A rate
	// HeyGen does not synthesize (it synthesizes 8000, 16000, 22050, 24000,
	// 32000, 44100 and 48000) is synthesized at 24 kHz and resampled to this
	// one.
	SampleRate int
	// Model is the HeyGen Voice model: "heygen-voice-1", "heygen-voice-1-turbo"
	// or "heygen-voice-1-turbo-free". Empty uses "heygen-voice-1-turbo".
	Model string
	// VoiceID is the HeyGen voice: a public voice or an instant clone in your
	// workspace. Professional voice clones are served only by NewHTTPTTS. Empty
	// uses the public voice Nadine.
	VoiceID string
	// Language is the language of the text; the zero value lets HeyGen detect
	// it.
	Language language.Language
	// ApplyTextNormalization is the text normalization mode: "auto", "on" or
	// "off". Empty leaves it to HeyGen, which uses "auto".
	ApplyTextNormalization string
	// ExpressivenessBoost is the voice's expressiveness, from 0.0 to 1.0. Nil
	// leaves it to HeyGen.
	ExpressivenessBoost *float64
	// EnableSSMLParsing says whether <break time="1s"/> tags become pauses.
	// Without it, HeyGen closes the WebSocket on a break tag. Nil leaves it
	// unset.
	EnableSSMLParsing *bool
	// EnableLogging says whether HeyGen stores a copy of the audio and text.
	// Nil leaves it to HeyGen, which stores them.
	EnableLogging *bool
	// EndUserID is your identifier for the end user, recorded with HeyGen's
	// usage for attribution. Empty leaves it out.
	EndUserID string
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.TextTypeSentence (the zero value) sends a sentence at a time, and
	// frames.TextTypeToken streams the LLM's tokens as they arrive, and HeyGen
	// generates at each sentence end.
	TextAggregation frames.TextType
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// withDefaults fills in what the caller left unset.
func (c TTSConfig) withDefaults() TTSConfig {
	c.URL = strings.TrimRight(cmp.Or(c.URL, defaultURL), "/")
	if c.SampleRate == 0 {
		c.SampleRate = defaultSampleRate
	}
	if c.TextAggregation == "" {
		c.TextAggregation = frames.TextTypeSentence
	}
	return c
}

// TTSSettings is the HeyGen Voice WebSocket configuration that can change while
// the pipeline runs. Every setting is a query parameter of the connection, so a
// change reconnects.
type TTSSettings struct {
	settings.TTS

	// ApplyTextNormalization is the text normalization mode: "auto", "on" or
	// "off". None leaves it to HeyGen, which uses "auto".
	ApplyTextNormalization settings.Opt[string] `settings:"apply_text_normalization"`
	// ExpressivenessBoost is the voice's expressiveness, from 0.0 to 1.0. None
	// leaves it to HeyGen.
	ExpressivenessBoost settings.Opt[float64] `settings:"expressiveness_boost"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with.
func newTTSSettings(cfg TTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cmp.Or(cfg.Model, defaultModel))
	s.Voice = settings.Set(cmp.Or(cfg.VoiceID, defaultVoiceID))
	s.Language = languageSetting(cfg.Language)
	s.ApplyTextNormalization = settings.Cleared[string]()
	if cfg.ApplyTextNormalization != "" {
		s.ApplyTextNormalization = settings.Set(cfg.ApplyTextNormalization)
	}
	s.ExpressivenessBoost = settings.Cleared[float64]()
	if cfg.ExpressivenessBoost != nil {
		s.ExpressivenessBoost = settings.Set(*cfg.ExpressivenessBoost)
	}
	return s
}

// NewTTS builds a HeyGen Voice TTS service over the multi-stream-input
// WebSocket.
//
// It streams text to HeyGen and plays the audio as it arrives. One connection
// carries a context per turn, and an interruption closes the turn's context
// without reconnecting. HeyGen's per-character alignment drives word-level
// TTSTextFrames, so an interruption commits only the words actually spoken.
//
// It defaults to "heygen-voice-1-turbo" and the public voice Nadine.
// Professional voice clones are served only by NewHTTPTTS.
func NewTTS(cfg TTSConfig) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns the synthesizer driving it.
func newTTS(cfg TTSConfig) (*tts.Base, *synthesizer) {
	cfg = cfg.withDefaults()
	s := &synthesizer{
		cfg:               cfg,
		live:              newTTSSettings(cfg),
		synthesisRate:     synthesisRate(cfg.SampleRate, websocketSampleRates),
		keepaliveInterval: keepaliveInterval,
		openContexts:      map[string]bool{},
		wordClocks:        map[string]*wordClock{},
	}
	b := tts.New("HeyGenTTS", s)
	s.svc = b
	s.ws = tts.NewWebsocket(b, s, cfg.Websocket)
	if cfg.TextAggregation == frames.TextTypeToken {
		b.SetTextAggregator(ttstext.NewTokenAggregator())
	}
	// HeyGen closes the socket at more than 5 running contexts. So a turn plays
	// its audio before the next turn opens a context.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s
}

// wordClock is the word-timing state of one context.
type wordClock struct {
	// offset is the seconds of the context's audio before the current message.
	// Each alignment's times start at zero, and some audio has no alignment.
	offset float64
	// partialWord is a word whose end has not arrived yet, and partialStart the
	// time it starts at.
	partialWord  string
	partialStart float64
}

// add takes one audio message's character alignment and returns the words it
// completes, each with its start in seconds. perCharacter says each character
// is a word, as in Chinese and Japanese, which have no spaces between words.
func (c *wordClock) add(a *alignment, perCharacter bool) []align.WordTiming {
	var words []align.WordTiming
	for i := range min(len(a.Chars), len(a.CharStartTimesMs)) {
		char := a.Chars[i]
		if char != " " {
			if c.partialWord == "" {
				c.partialStart = c.offset + a.CharStartTimesMs[i]/1000
			}
			c.partialWord += char
		}
		if c.partialWord != "" && (char == " " || perCharacter) {
			words = append(words, align.WordTiming{Word: c.partialWord, Offset: c.partialStart})
			c.partialWord = ""
		}
	}
	return words
}

// synthesizer streams over one connection held open for the call, with one
// HeyGen context per turn. The receive loop is run by the service's websocket
// base, which reconnects when the connection drops.
type synthesizer struct {
	cfg  TTSConfig
	live *TTSSettings
	// svc is the service this synthesizer drives. It is also the host audio is
	// appended to.
	svc  *tts.Base
	host tts.AudioContextHost
	ws   *tts.Websocket
	// synthesisRate is the rate HeyGen synthesizes at, resolved from the
	// service's.
	synthesisRate int
	// keepaliveInterval is held here so a test can shorten it.
	keepaliveInterval time.Duration

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be connecting on the frame goroutine.
	dialMu sync.Mutex

	// mu guards everything below. It is never held across a call into the host
	// or a write.
	mu   sync.Mutex
	conn *wsutil.Conn
	// closed records that reading from conn failed, so the connection is gone
	// although it has not been disconnected yet.
	closed bool
	// connectNotBefore is the time before which a refused handshake asked for no
	// other.
	connectNotBefore time.Time
	// openContexts are the contexts open at HeyGen. A message that names any
	// other context opens it there.
	openContexts map[string]bool
	// wordClocks holds each context's word timing state.
	wordClocks map[string]*wordClock
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *synthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the HeyGen model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(""), VoiceID: s.live.Voice.Or("")}
}

// RequiresTrailingSpace reports that text sent ends in a space, implementing
// tts.TrailingSpaceRequirer: HeyGen joins text messages without a separator.
func (s *synthesizer) RequiresTrailingSpace() bool { return true }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way HeyGen Voice does, implementing
// tts.LanguageNamer.
func (s *synthesizer) ServiceLanguage(l language.Language) string { return heygenLanguage(l) }

// UpdateSettings applies a change, implementing tts.SettingsUpdater. Every
// setting is in the URL, so a change reconnects.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	slog.Debug("heygen tts: settings changed, reconnecting", "fields", changed.String())
	s.Disconnect(ctx)
	s.Connect(ctx)
	return nil
}

// endpoint builds the multi-stream-input URL, which carries the whole
// configuration of the connection.
func (s *synthesizer) endpoint() string {
	q := url.Values{}
	q.Set("voice_id", s.live.Voice.Or(""))
	q.Set("model", s.live.Model.Or(""))
	q.Set("output_format", "pcm_"+strconv.Itoa(s.synthesisRate))
	// Without auto mode, HeyGen waits for 120 characters of text.
	q.Set("auto_mode", "true")
	if lang := s.live.Language.Or(""); supportedLanguage(lang) {
		q.Set("language", lang)
	}
	if normalization, ok := s.live.ApplyTextNormalization.Value(); ok {
		q.Set("apply_text_normalization", normalization)
	}
	if boost, ok := s.live.ExpressivenessBoost.Value(); ok {
		q.Set("expressiveness_boost", strconv.FormatFloat(boost, 'f', -1, 64))
	}
	if s.cfg.EnableSSMLParsing != nil {
		q.Set("enable_ssml_parsing", strconv.FormatBool(*s.cfg.EnableSSMLParsing))
	}
	if s.cfg.EnableLogging != nil {
		q.Set("enable_logging", strconv.FormatBool(*s.cfg.EnableLogging))
	}
	if s.cfg.EndUserID != "" {
		q.Set("end_user_id", s.cfg.EndUserID)
	}
	return s.cfg.URL + ttsPath + "?" + q.Encode()
}

// Start connects when the pipeline starts, implementing tts.Starter.
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
		s.ws.CreateKeepaliveTask(ctx, s.keepaliveInterval, s)
	}
}

// Disconnect stops the receive loop and the keepalive and closes the
// connection, implementing tts.WebsocketService. The contexts of the connection
// are ended first (their bookkeeping too, see CloseConnectionContexts).
func (s *synthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	s.ws.CancelKeepaliveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens the connection unless one is open, implementing
// tts.WebsocketHandler. A failure is reported here and leaves no connection
// behind, which a redial reads as a failed attempt. A refused handshake reports
// HeyGen's reason, and the wait it asks for in Retry-After holds off the next
// handshake.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to heygen tts")
	header := http.Header{}
	header.Set(headerAPIKey, s.cfg.APIKey)
	conn, err := s.ws.Dial(ctx, s.endpoint(), header, wsutil.DefaultReadLimit)
	if err != nil {
		s.mu.Lock()
		s.conn, s.closed = nil, false
		s.mu.Unlock()
		ef := frames.NewErrorFrame("Error connecting to HeyGen: " + s.refusal(err))
		ef.Err = err
		s.ws.ReportError(ctx, ef, false)
		return nil
	}
	s.mu.Lock()
	s.conn, s.closed = conn, false
	s.mu.Unlock()
	return nil
}

// refusal is why a dial failed. A refused handshake's response holds the
// reason for the refusal, and the wait it asks for in Retry-After is recorded,
// so no handshake is made before that time.
func (s *synthesizer) refusal(err error) string {
	reason := err.Error()
	refused, ok := errors.AsType[*wsutil.HandshakeError](err)
	if !ok {
		return reason
	}
	if len(refused.Body) > 0 {
		reason += ": " + errorDetail(refused.Body)
	}
	if retryAfter := refused.Header.Get("Retry-After"); isDigits(retryAfter) {
		if seconds, convErr := strconv.Atoi(retryAfter); convErr == nil {
			s.mu.Lock()
			s.connectNotBefore = time.Now().Add(time.Duration(seconds) * time.Second)
			s.mu.Unlock()
		}
	}
	return reason
}

// isDigits reports whether v is a whole number of seconds: digits and nothing
// else.
func isDigits(v string) bool {
	return v != "" && strings.Trim(v, "0123456789") == ""
}

// DisconnectWebsocket closes the connection, implementing
// tts.WebsocketHandler. What the service was timing is reported as far as it
// got, and the contexts HeyGen held open on the connection are forgotten.
func (s *synthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	s.svc.StopAllMetrics(ctx)
	s.mu.Lock()
	conn := s.conn
	s.conn, s.closed = nil, false
	clear(s.openContexts)
	s.mu.Unlock()
	if conn == nil {
		return nil
	}
	slog.Debug("disconnecting from heygen tts")
	if err := conn.Close(websocket.StatusNormalClosure, ""); err != nil && !closedConnection(err) {
		s.svc.PushError(ctx, fmt.Sprintf("Error disconnecting from HeyGen: %v", err), err, false)
	}
	return nil
}

// closedConnection reports whether err is the connection having been closed
// already.
func closedConnection(err error) bool {
	return websocket.CloseStatus(err) != -1 || errors.Is(err, net.ErrClosed)
}

// PrepareReconnectWebsocket waits as long as the last refused handshake asked
// before the connection is dialed again, implementing
// tts.WebsocketReconnectPreparer. Every handshake spends HeyGen's rate limit,
// so one made before that time is refused and only delays recovery. The
// contexts of the connection are ended before the wait, so a turn in progress
// does not hang on audio that will not come.
func (s *synthesizer) PrepareReconnectWebsocket(ctx context.Context) {
	s.mu.Lock()
	delay := time.Until(s.connectNotBefore)
	s.mu.Unlock()
	if delay <= 0 {
		return
	}
	s.CloseConnectionContexts()
	s.svc.ReplaceConnection()
	slog.Debug("heygen tts: waiting before reconnecting, as HeyGen asked", "delay", delay)
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}

// CloseConnectionContexts forgets the contexts a replaced or closed connection
// held open at HeyGen, implementing tts.ConnectionContextsCloser. The base ends
// the audio contexts themselves, right after.
func (s *synthesizer) CloseConnectionContexts() {
	s.mu.Lock()
	defer s.mu.Unlock()
	clear(s.openContexts)
	clear(s.wordClocks)
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *synthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.closed
}

// send writes one message naming a context. A connection that has closed is
// left to the receive loop, which handles it and its contexts.
func (s *synthesizer) send(ctx context.Context, contextID string, fields map[string]any) {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return
	}
	if err := s.write(ctx, conn, contextID, fields); err != nil {
		slog.Debug("heygen tts: message not sent", "context", contextID, "err", err)
	}
}

// write writes one message naming a context on conn.
func (s *synthesizer) write(ctx context.Context, conn *wsutil.Conn, contextID string, fields map[string]any) error {
	msg := map[string]any{keyContextID: contextID}
	maps.Copy(msg, fields)
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, payload)
}

// isOpen reports whether the context is open at HeyGen.
func (s *synthesizer) isOpen(contextID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return contextID != "" && s.openContexts[contextID]
}

// closeContext closes a context open at HeyGen, so its final message arrives.
func (s *synthesizer) closeContext(ctx context.Context, contextID string) {
	s.mu.Lock()
	open := contextID != "" && s.openContexts[contextID]
	delete(s.openContexts, contextID)
	s.mu.Unlock()
	if open {
		s.send(ctx, contextID, map[string]any{keyCloseContext: true})
	}
}

// forgetContext drops what the service holds for a context.
func (s *synthesizer) forgetContext(contextID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.openContexts, contextID)
	delete(s.wordClocks, contextID)
}

// FlushAudio asks HeyGen to generate any text still buffered for a context, or
// for the active one when none is named, implementing tts.AudioFlusher. A
// flush naming a context the service never opened would open it at HeyGen, so
// it names only an open one.
func (s *synthesizer) FlushAudio(ctx context.Context, contextID string) {
	if contextID == "" {
		contextID = s.svc.ActiveAudioContextID()
	}
	if s.isOpen(contextID) {
		s.send(ctx, contextID, map[string]any{keyFlush: true})
	}
}

// OnTurnContextCompleted closes the turn's context so its final message
// arrives, implementing tts.TurnContextCompleter. The base flushes it first: a
// close without a flush discards unspoken text.
func (s *synthesizer) OnTurnContextCompleted(ctx context.Context, contextID string) {
	s.closeContext(ctx, contextID)
}

// OnAudioContextInterrupted closes the context when the bot is interrupted,
// keeping the connection, implementing tts.AudioContextInterrupter.
func (s *synthesizer) OnAudioContextInterrupted(ctx context.Context, contextID string) {
	ctx = context.WithoutCancel(ctx)
	s.svc.StopAllMetrics(ctx)
	s.closeContext(ctx, contextID)
	s.forgetContext(contextID)
}

// OnAudioContextCompleted forgets the context once its audio has played,
// implementing tts.AudioContextCompleter.
func (s *synthesizer) OnAudioContextCompleted(_ context.Context, contextID string) {
	s.forgetContext(contextID)
}

// KeepaliveReady reports whether the connection is open, implementing
// tts.KeepaliveHandler.
func (s *synthesizer) KeepaliveReady() bool { return s.Connected() }

// SendKeepalive keeps an idle connection open, implementing
// tts.KeepaliveHandler.
//
// A keepalive that names a closed context opens it at HeyGen, and one with no
// context ID opens "default", so a keepalive names only an open context. With
// no open context, the service closes a context it never opens, which resets
// HeyGen's inactivity timer and changes nothing else.
func (s *synthesizer) SendKeepalive(ctx context.Context) error {
	if id := s.svc.ActiveAudioContextID(); s.isOpen(id) {
		s.send(ctx, id, map[string]any{keyText: ""})
		return nil
	}
	s.send(ctx, keepaliveContextID, map[string]any{keyCloseContext: true})
	return nil
}

// ReceiveMessages handles the connection's messages until it ends,
// implementing tts.WebsocketHandler.
func (s *synthesizer) ReceiveMessages(ctx context.Context) error {
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
		if err := s.handleMessage(ctx, data); err != nil {
			return err
		}
	}
}

// serverMessage is one message HeyGen sends.
//
//nolint:tagliatelle // HeyGen's wire keys are camelCase
type serverMessage struct {
	ContextID string          `json:"contextId"`
	Audio     string          `json:"audio"`
	IsFinal   *bool           `json:"isFinal"`
	Alignment *alignment      `json:"alignment"`
	Error     json.RawMessage `json:"error"`
}

// alignment times each character of a message's audio, from the start of that
// audio.
//
//nolint:tagliatelle // HeyGen's wire keys are camelCase
type alignment struct {
	Chars            []string  `json:"chars"`
	CharStartTimesMs []float64 `json:"charStartTimesMs"`
}

// handleMessage handles one message HeyGen sent.
func (s *synthesizer) handleMessage(ctx context.Context, data []byte) error {
	var msg serverMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return fmt.Errorf("heygen tts: decode message: %w", err)
	}

	// HeyGen sends an error message just before it closes the socket.
	if errorGiven(msg.Error) {
		var detail struct {
			Code any `json:"code"`
		}
		_ = json.Unmarshal(msg.Error, &detail)
		code, _ := detail.Code.(string)
		var opts []processor.ErrorOption
		if category, ok := errorCategories[code]; ok {
			opts = append(opts, processor.WithErrorCategory(category))
		}
		s.svc.PushError(ctx, "HeyGen TTS error: "+errorDetail(data), nil, false, opts...)
		return nil
	}

	id := msg.ContextID
	if msg.IsFinal != nil && *msg.IsFinal {
		s.finishContext(id)
		return nil
	}

	// An interrupted context's audio and alignment can still arrive.
	if msg.Audio == "" || s.host == nil || !s.host.AudioContextAvailable(id) {
		return nil
	}
	pcm, err := base64.StdEncoding.DecodeString(msg.Audio)
	if err != nil {
		return fmt.Errorf("heygen tts: decode audio: %w", err)
	}
	seconds := float64(len(pcm)) / 2 / float64(s.synthesisRate)
	f := frames.NewTTSAudioRawFrame(pcm, s.synthesisRate, 1)
	f.ContextID = id
	s.host.AppendToAudioContext(id, f)

	cjk := s.isCJK()
	s.mu.Lock()
	clock := s.wordClocks[id]
	if clock == nil {
		clock = &wordClock{}
		s.wordClocks[id] = clock
	}
	var words []align.WordTiming
	if msg.Alignment != nil {
		words = clock.add(msg.Alignment, cjk)
	}
	clock.offset += seconds
	s.mu.Unlock()
	s.addWords(id, words, cjk)
	return nil
}

// errorGiven reports whether a message's error is set: present, and neither
// null, false, empty nor zero.
func errorGiven(raw json.RawMessage) bool {
	switch strings.TrimSpace(string(raw)) {
	case "", "null", "false", `""`, "{}", "[]", "0":
		return false
	default:
		return true
	}
}

// finishContext ends a context HeyGen reported final, which it can do on its
// own mid-turn.
func (s *synthesizer) finishContext(id string) {
	if s.host != nil && s.host.AudioContextAvailable(id) {
		s.mu.Lock()
		var last []align.WordTiming
		if clock := s.wordClocks[id]; clock != nil && clock.partialWord != "" {
			last = []align.WordTiming{{Word: clock.partialWord, Offset: clock.partialStart}}
		}
		s.mu.Unlock()
		s.addWords(id, last, s.isCJK())
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = id
		s.host.AppendToAudioContext(id, stopped)
		s.host.RemoveAudioContext(id)
	}
	s.forgetContext(id)
	// The turn's next sentence opens a new context.
	s.svc.RenewTurnContext(id)
}

// isCJK reports whether the language is Chinese or Japanese, which have no
// spaces between words.
func (s *synthesizer) isCJK() bool {
	switch s.live.Language.Or("") {
	case "ja", "zh":
		return true
	default:
		return false
	}
}

// addWords adds words to a context's queue. In Chinese and Japanese each
// character is a token of its own, which already reads as continuous text, and
// punctuation joins the character before it.
func (s *synthesizer) addWords(id string, words []align.WordTiming, cjk bool) {
	if len(words) == 0 || s.host == nil {
		return
	}
	s.host.AddWordTimestamps(id, words, tts.WordTimingOptions{IncludesInterFrameSpaces: cjk, PreMergeTokens: cjk})
}

// RunTTS sends text to the context for synthesis. It yields nothing when the
// text is sent: the audio arrives on the receive loop, which appends it to the
// context it belongs to.
//
// A connection that has closed is opened again first, unless the receive loop
// is reconnecting or a refused handshake asked to wait: a handshake then
// spends the rate limit. Text that cannot be sent ends its context with an
// error and a stop frame.
func (s *synthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	s.mu.Lock()
	waiting := time.Now().Before(s.connectNotBefore)
	s.mu.Unlock()
	if !s.Connected() && !s.ws.ReconnectInProgress() && !waiting {
		s.Connect(ctx)
	}
	if err := s.sendText(ctx, text, contextID); err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
		ef.Err = err
		_ = yield(ef)
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		_ = yield(stopped)
		return nil
	}
	tts.StartUsageMetrics(ctx, text)
	return nil
}

// sendText sends text on the context, which opens it at HeyGen.
func (s *synthesizer) sendText(ctx context.Context, text, contextID string) error {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return errNotConnected
	}
	if err := s.write(ctx, conn, contextID, map[string]any{keyText: text}); err != nil {
		return err
	}
	s.mu.Lock()
	s.openContexts[contextID] = true
	s.mu.Unlock()
	return nil
}

// RunTTSTimed sends the text like RunTTS, implementing tts.WordTimestamps. The
// timings arrive on the receive loop with the audio, so neither callback is
// used here; implementing the interface is what puts the base on the
// word-aligned path.
func (s *synthesizer) RunTTSTimed(
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
	_ tts.ContextSynthesizer         = (*synthesizer)(nil)
	_ tts.WordTimestamps             = (*synthesizer)(nil)
	_ tts.AudioFlusher               = (*synthesizer)(nil)
	_ tts.TurnContextCompleter       = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter    = (*synthesizer)(nil)
	_ tts.AudioContextCompleter      = (*synthesizer)(nil)
	_ tts.Starter                    = (*synthesizer)(nil)
	_ tts.Describer                  = (*synthesizer)(nil)
	_ tts.TrailingSpaceRequirer      = (*synthesizer)(nil)
	_ tts.SettingsUpdater            = (*synthesizer)(nil)
	_ tts.LanguageNamer              = (*synthesizer)(nil)
	_ tts.WebsocketService           = (*synthesizer)(nil)
	_ tts.KeepaliveHandler           = (*synthesizer)(nil)
	_ tts.WebsocketReconnectPreparer = (*synthesizer)(nil)
	_ tts.ConnectionContextsCloser   = (*synthesizer)(nil)
)
