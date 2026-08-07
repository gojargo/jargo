package elevenlabs

import (
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
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

// realtimeTTSPath is the multi-stream-input endpoint. Multi-stream is what
// carries a context id on every message, which is how audio is attributed to the
// synthesis that asked for it.
const realtimeTTSPath = "/v1/text-to-speech/%s/multi-stream-input"

// defaultRealtimeTTSHost is the ElevenLabs WebSocket host.
const defaultRealtimeTTSHost = "wss://api.elevenlabs.io"

// keepaliveInterval is how often an idle connection is pinged with empty text.
// The server drops a stream that has gone quiet, and between turns nothing is
// sent for as long as the user talks.
const keepaliveInterval = 10 * time.Second

// abortWriteTimeout bounds the best-effort close sent for an interrupted turn.
const abortWriteTimeout = 2 * time.Second

// Message keys on the multi-stream-input protocol.
const (
	keyText         = "text"
	keyContextID    = "context_id"
	keyCloseContext = "close_context"
	keyFlush        = "flush"
)

// errRealtimeTTS wraps an error the server reported on the stream.
//
//nolint:gochecknoglobals // sentinel error
var errRealtimeTTS = errors.New("elevenlabs realtime tts")

// errStreamClosed releases a turn whose connection failed under it.
//
//nolint:gochecknoglobals // sentinel error
var errStreamClosed = errors.New("elevenlabs realtime tts: stream closed")

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
	// PronunciationDictionaryLocators applies the given dictionaries.
	PronunciationDictionaryLocators []PronunciationDictionaryLocator
	// WordTimestamps reports per-word timing alongside the audio, which lets the
	// assistant context record what was actually spoken before an interruption.
	// ElevenLabs times every character, so the words are assembled from those.
	WordTimestamps bool
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
		// Without auto mode the server holds text until it is explicitly flushed,
		// and closing a context discards whatever was never flushed — the synthesis
		// then ends with a final marker and no audio at all. Sentence-at-a-time
		// synthesis has nothing to gain from that buffering, so it is on unless the
		// caller turns it off.
		c.AutoMode = new(true)
	}
	return c
}

// NewRealtimeTTS builds an ElevenLabs TTS service over the streaming WebSocket
// API. Unlike NewTTS, which issues one HTTP request per sentence, this holds a
// single connection open for the session and one synthesis context open for each
// turn, so neither a sentence nor a turn pays for setup before its audio starts.
func NewRealtimeTTS(cfg RealtimeTTSConfig) *tts.Base {
	s := &realtimeSynthesizer{cfg: cfg.withDefaults()}
	if cfg.WordTimestamps {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the
		// base takes the word-aligned path solely when timings were requested.
		return tts.New("ElevenLabsRealtimeTTS", &timedRealtimeSynthesizer{realtimeSynthesizer: s})
	}
	return tts.New("ElevenLabsRealtimeTTS", s)
}

// realtimeSynthesizer streams over one held-open connection, with one synthesis
// context per turn.
//
// A reader goroutine owns the socket for as long as the connection lasts and
// attributes each message to the context id it carries. Sending a sentence is
// therefore just a write: it does not wait for the audio of the sentence before
// it, so the server generates the next while the current is still playing. The
// audio it reads goes into the base's audio context for that id, which is what
// pushes it downstream in order.
type realtimeSynthesizer struct {
	cfg  RealtimeTTSConfig
	host tts.AudioContextHost

	// writeMu serializes writes; the keepalive and the frame goroutine both
	// write, and the connection permits one writer at a time.
	writeMu sync.Mutex

	// mu guards the connection and the per-context state. It is never held
	// across a call into the host: appending blocks while the audio plays out,
	// and the next sentence has to be able to go out while that happens.
	mu       sync.Mutex
	conn     *wsutil.Conn
	connStop context.CancelFunc
	// active is the context the keepalive may name, set once its opening message
	// has gone out.
	active string
	states map[string]*contextState
}

// timedRealtimeSynthesizer adds word-timestamp streaming on top of
// realtimeSynthesizer. It implements tts.WordTimestamps.
type timedRealtimeSynthesizer struct {
	*realtimeSynthesizer
}

// contextState is what the reader carries across one context's messages.
type contextState struct {
	acc uctx.CharAccumulator
	// cumulative is how much of this context's audio earlier messages accounted
	// for, in seconds. ElevenLabs times each message from the start of the audio
	// it carries, so the offsets have to be rebased onto the context's timeline;
	// without that every message after the first reports its words as starting
	// near zero.
	cumulative float64
	// direct, when set, takes the audio instead of the host. It is the standalone
	// Synthesize path, which has no audio context to append to.
	direct *directSink
}

// directSink receives a standalone synthesis, for a caller holding the
// synthesizer outside a pipeline.
type directSink struct {
	audio func(pcm []byte) error
	word  func(text string, offset float64) error
	done  chan struct{}
	once  sync.Once
	err   error
}

// settle releases the waiter, with err if the synthesis did not finish cleanly.
func (d *directSink) settle(err error) {
	d.once.Do(func() {
		d.err = err
		close(d.done)
	})
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *realtimeSynthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *realtimeSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the ElevenLabs model and voice synthesis is billed against.
func (s *realtimeSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.VoiceID}
}

// spacelessLanguage reports whether the configured language is written without
// spaces between words. Chinese and Japanese are, so the tokens this provider
// times already read as continuous text and a consumer joining them must add no
// spacing of its own.
func (s *realtimeSynthesizer) spacelessLanguage() bool {
	switch s.cfg.Language.BaseCode() {
	case "zh", "ja":
		return true
	default:
		return false
	}
}

// endpoint builds the multi-stream-input URL. The voice, model and output format
// are fixed for the connection; only the text varies per message.
func (s *realtimeSynthesizer) endpoint() string {
	q := url.Values{}
	q.Set("model_id", s.cfg.Model)
	q.Set("output_format", outputFormat(s.cfg.SampleRate))
	if s.cfg.AutoMode != nil {
		q.Set("auto_mode", strconv.FormatBool(*s.cfg.AutoMode))
	}
	if s.cfg.ApplyTextNormalization != "" {
		q.Set("apply_text_normalization", s.cfg.ApplyTextNormalization)
	}
	if s.cfg.EnableLogging != nil {
		q.Set("enable_logging", strconv.FormatBool(*s.cfg.EnableLogging))
	}
	if code := elevenlabsLanguage(s.cfg.Language); code != "" {
		q.Set("language_code", code)
	}
	path := fmt.Sprintf(realtimeTTSPath, s.cfg.VoiceID)
	return s.cfg.Host + path + "?" + q.Encode()
}

// stream returns the shared connection, dialing it on first use and redialing
// when a previous turn left it broken. Dialing also starts the goroutines that
// own the connection for its lifetime. Callers hold mu.
func (s *realtimeSynthesizer) stream(ctx context.Context) (*wsutil.Conn, error) {
	if s.conn != nil {
		return s.conn, nil
	}
	header := http.Header{}
	header.Set("xi-api-key", s.cfg.APIKey)
	conn, err := wsutil.Dial(ctx, s.endpoint(), header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	// The reader outlives the call that dialed: it runs until the connection is
	// dropped, not until this sentence is done.
	runCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	s.conn, s.connStop = conn, stop
	go s.readLoop(runCtx, conn)
	go s.keepalive(runCtx, conn)
	return conn, nil
}

// drop closes the connection so the next sentence dials a fresh one, releasing
// every context still open on it. Callers hold mu.
func (s *realtimeSynthesizer) drop(cause error) {
	if s.connStop != nil {
		s.connStop()
		s.connStop = nil
	}
	if s.conn != nil {
		_ = s.conn.Close(websocket.StatusInternalError, "")
		s.conn = nil
	}
	for id, st := range s.states {
		if st.direct != nil {
			st.direct.settle(cause)
		}
		delete(s.states, id)
	}
	s.active = ""
}

// write sends one JSON message, serialized against the other writers.
func (s *realtimeSynthesizer) write(ctx context.Context, conn *wsutil.Conn, msg map[string]any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(ctx, websocket.MessageText, payload)
}

// readLoop owns the connection's incoming messages for as long as it lasts,
// handing each to the context it belongs to.
func (s *realtimeSynthesizer) readLoop(ctx context.Context, conn *wsutil.Conn) {
	for {
		msg, err := readRealtimeTTSMessage(ctx, conn)
		if err == nil {
			err = s.dispatch(msg)
		}
		if err != nil {
			s.mu.Lock()
			if s.conn == conn {
				s.drop(err)
			}
			s.mu.Unlock()
			return
		}
	}
}

// dispatch delivers one message to the context it names.
//
// Messages for any other context are skipped: an interruption abandons a turn
// mid-flight, and audio the server had already generated for it arrives
// afterwards. Attributing by context id keeps that audio out of the next turn,
// which is what the ids are for.
func (s *realtimeSynthesizer) dispatch(msg *realtimeTTSMessage) error {
	s.mu.Lock()
	st := s.states[msg.ContextID]
	host := s.host
	s.mu.Unlock()
	if st == nil {
		return nil
	}
	if msg.IsFinal {
		return s.finishContext(msg.ContextID, st, host)
	}
	return s.deliver(msg, st, host)
}

// deliver emits a message's audio and reports the words it completes. It runs on
// the reader goroutine with no lock held, so appending is free to block while
// the audio already queued plays out.
func (s *realtimeSynthesizer) deliver(msg *realtimeTTSMessage, st *contextState, host tts.AudioContextHost) error {
	if err := s.deliverAudio(msg, st, host); err != nil {
		return err
	}
	if st.direct != nil && st.direct.word == nil {
		return nil
	}
	if st.direct == nil && !s.cfg.WordTimestamps {
		return nil
	}
	return s.reportWords(msg, st, host)
}

// deliverAudio decodes a message's audio and sends it where the context's output
// goes.
func (s *realtimeSynthesizer) deliverAudio(
	msg *realtimeTTSMessage,
	st *contextState,
	host tts.AudioContextHost,
) error {
	if msg.Audio == "" {
		return nil
	}
	pcm, err := base64.StdEncoding.DecodeString(msg.Audio)
	if err != nil {
		return fmt.Errorf("%w: decode audio: %w", errRealtimeTTS, err)
	}
	if st.direct != nil {
		return st.direct.audio(pcm)
	}
	if host != nil {
		host.AppendToAudioContext(msg.ContextID, frames.NewTTSAudioRawFrame(pcm, s.cfg.SampleRate, 1))
	}
	return nil
}

// reportWords folds a message's character timings into whole words, rebased onto
// the context's timeline, and reports each word that completed.
func (s *realtimeSynthesizer) reportWords(
	msg *realtimeTTSMessage,
	st *contextState,
	host tts.AudioContextHost,
) error {
	alignment := msg.Alignment
	if s.cfg.PronunciationDictionaryLocators != nil && msg.NormalizedAlignment != nil {
		// A pronunciation dictionary rewrites what is spoken, so the normalized
		// form is the one the timings belong to.
		alignment = msg.NormalizedAlignment
	}
	if alignment == nil {
		return nil
	}
	starts := make([]float64, len(alignment.CharStartTimesMs))
	for i, ms := range alignment.CharStartTimesMs {
		starts[i] = st.cumulative + ms/1000
	}
	words, err := st.acc.Add(alignment.Chars, starts)
	if err != nil {
		return err
	}
	st.cumulative = alignment.end(st.cumulative)

	// The words are reported as this provider timed them. Assembling the
	// characters on spaces has already attached each word's punctuation to it,
	// and punctuation the writing separates from its word (as French does before
	// "?") is a token in its own right, which is what the text says it is.
	opts := tts.WordTimingOptions{IncludesInterFrameSpaces: s.spacelessLanguage()}
	if st.direct == nil {
		if host != nil {
			host.AddWordTimestamps(msg.ContextID, words, opts)
		}
		return nil
	}
	for _, wt := range words {
		if err := st.direct.word(wt.Word, wt.Offset); err != nil {
			return err
		}
	}
	return nil
}

// emitWord reports one spoken token to wherever the context's output goes.
func (s *realtimeSynthesizer) emitWord(
	contextID string,
	st *contextState,
	host tts.AudioContextHost,
	word string,
	offset float64,
) error {
	if st.direct != nil {
		return st.direct.word(word, offset)
	}
	if host != nil {
		host.AppendWordToAudioContext(contextID, word, offset)
	}
	return nil
}

// finishContext closes a context out once the server has marked it final.
func (s *realtimeSynthesizer) finishContext(contextID string, st *contextState, host tts.AudioContextHost) error {
	// The closing word ends on the utterance, not on a space.
	var err error
	if wt, ok := st.acc.Flush(); ok {
		err = s.emitWord(contextID, st, host, wt.Word, wt.Offset)
	}
	s.forgetContext(contextID)
	if st.direct != nil {
		st.direct.settle(err)
		return err
	}
	if host != nil {
		// The stop frame rides the queue behind the audio, so it is pushed only
		// once the last of that audio has been.
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		host.AppendToAudioContext(contextID, stopped)
		host.RemoveAudioContext(contextID)
	}
	return err
}

// forgetContext drops a context's reader state.
func (s *realtimeSynthesizer) forgetContext(contextID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.states, contextID)
	if s.active == contextID {
		s.active = ""
	}
}

// keepalive holds an idle connection open. Between turns nothing is written for
// as long as the user is talking, and the server drops a stream that goes quiet.
func (s *realtimeSynthesizer) keepalive(ctx context.Context, conn *wsutil.Conn) {
	tick := time.NewTicker(keepaliveInterval)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		msg := map[string]any{keyText: ""}
		s.mu.Lock()
		active := s.active
		s.mu.Unlock()
		if active != "" {
			// Only a context whose opening message has gone out may be named: a
			// keepalive that opened one would open it without the voice settings
			// the opening message carries, and the server rejects the later one.
			msg[keyContextID] = active
		}
		if err := s.write(ctx, conn, msg); err != nil {
			return
		}
	}
}

// Start dials the shared connection when the pipeline starts, implementing
// tts.Starter. The handshake to the vendor is the slowest part of a session's
// first sentence, so leaving it to be dialed lazily puts it in front of the
// bot's opening words.
func (s *realtimeSynthesizer) Start(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.stream(ctx); err != nil {
		// Best-effort: the first sentence dials again and reports the failure.
		slog.Debug("elevenlabs realtime tts connect on start failed", "error", err)
	}
}

// Close releases the shared connection, implementing tts.Closer.
func (s *realtimeSynthesizer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	conn := s.conn
	if conn == nil {
		return nil
	}
	if s.connStop != nil {
		s.connStop()
		s.connStop = nil
	}
	s.conn = nil
	for id, st := range s.states {
		if st.direct != nil {
			st.direct.settle(errStreamClosed)
		}
		delete(s.states, id)
	}
	s.active = ""
	// Ask the server to close so it does not sit on a half-open stream.
	_ = s.write(context.Background(), conn, map[string]any{"close_socket": true})
	return conn.Close(websocket.StatusNormalClosure, "")
}

// RunTTS sends one sentence on the turn's context, opening that context on the
// wire the first time it is used. It yields nothing: the call returns once the
// text is sent, and the audio arrives on the reader goroutine, which appends it
// to the context it belongs to.
func (s *realtimeSynthesizer) RunTTS(
	ctx context.Context, text, contextID string, _ func(f frames.Frame) error,
) error {
	conn, err := s.openContext(ctx, contextID, nil)
	if err != nil {
		return err
	}
	return s.sendText(ctx, conn, contextID, text)
}

// openContext dials if needed and sends the context's opening message the first
// time the context is used, reporting whether it did.
func (s *realtimeSynthesizer) openContext(
	ctx context.Context,
	contextID string,
	direct *directSink,
) (*wsutil.Conn, error) {
	var (
		conn *wsutil.Conn
		err  error
	)
	s.mu.Lock()
	conn, err = s.stream(ctx)
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	if s.states == nil {
		s.states = map[string]*contextState{}
	}
	if _, ok := s.states[contextID]; ok {
		s.mu.Unlock()
		return conn, nil
	}
	s.states[contextID] = &contextState{direct: direct}
	s.mu.Unlock()

	// The opening message carries the voice parameters for the context. A single
	// space rather than the first sentence: it initializes without committing text.
	open := map[string]any{keyText: " ", keyContextID: contextID}
	if s.cfg.VoiceSettings != nil {
		open["voice_settings"] = s.cfg.VoiceSettings
	}
	if len(s.cfg.PronunciationDictionaryLocators) > 0 {
		open["pronunciation_dictionary_locators"] = s.cfg.PronunciationDictionaryLocators
	}
	if err := s.write(ctx, conn, open); err != nil {
		// A failed write leaves the server's view of the stream unknown.
		s.mu.Lock()
		if s.conn == conn {
			s.drop(err)
		}
		s.mu.Unlock()
		return nil, err
	}
	s.mu.Lock()
	s.active = contextID
	s.mu.Unlock()
	return conn, nil
}

// sendText hands one piece of text to an open context.
func (s *realtimeSynthesizer) sendText(
	ctx context.Context,
	conn *wsutil.Conn,
	contextID, text string,
) error {
	if err := s.write(ctx, conn, map[string]any{keyText: text, keyContextID: contextID}); err != nil {
		s.mu.Lock()
		if s.conn == conn {
			s.drop(err)
		}
		s.mu.Unlock()
		return err
	}
	return nil
}

// FlushAudio tells the server to generate whatever text it is still holding,
// implementing tts.AudioFlusher.
func (s *realtimeSynthesizer) FlushAudio(ctx context.Context, contextID string) {
	s.mu.Lock()
	conn := s.conn
	_, known := s.states[contextID]
	s.mu.Unlock()
	if conn == nil || !known {
		return
	}
	_ = s.write(ctx, conn, map[string]any{keyContextID: contextID, keyFlush: true})
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
func (s *realtimeSynthesizer) OnAudioContextInterrupted(_ context.Context, contextID string) {
	ctx, cancel := context.WithTimeout(context.Background(), abortWriteTimeout)
	defer cancel()
	s.closeContext(ctx, contextID)
	s.forgetContext(contextID)
}

// OnAudioContextCompleted releases the reader state for a context that has
// finished playing, implementing tts.AudioContextCompleter.
func (s *realtimeSynthesizer) OnAudioContextCompleted(_ context.Context, contextID string) {
	s.forgetContext(contextID)
}

// closeContext asks the server to close a context, best effort.
func (s *realtimeSynthesizer) closeContext(ctx context.Context, contextID string) {
	s.mu.Lock()
	conn := s.conn
	_, known := s.states[contextID]
	s.mu.Unlock()
	if conn == nil || !known {
		return
	}
	_ = s.write(ctx, conn, map[string]any{keyContextID: contextID, keyCloseContext: true})
}

// charAlignment times each character of a message's audio, relative to the start
// of that audio rather than of the context.
type charAlignment struct {
	Chars            []string  `json:"chars"`
	CharStartTimesMs []float64 `json:"charStartTimesMs"` //nolint:tagliatelle // ElevenLabs wire keys are camelCase
	CharDurationsMs  []float64 `json:"charDurationsMs"`  //nolint:tagliatelle // ElevenLabs wire keys are camelCase
}

// end reports where the context's timeline stands once this message's audio has
// been accounted for, starting from base. It falls back to the last character's
// start when the server sent no durations.
func (a *charAlignment) end(base float64) float64 {
	if len(a.CharStartTimesMs) == 0 {
		return base
	}
	last := a.CharStartTimesMs[len(a.CharStartTimesMs)-1]
	if len(a.CharDurationsMs) == len(a.CharStartTimesMs) {
		last += a.CharDurationsMs[len(a.CharDurationsMs)-1]
	}
	return base + last/1000
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

// readRealtimeTTSMessage reads and decodes one server message.
func readRealtimeTTSMessage(ctx context.Context, conn *wsutil.Conn) (*realtimeTTSMessage, error) {
	_, data, err := conn.Read(ctx)
	if err != nil {
		return nil, err
	}
	var msg realtimeTTSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, fmt.Errorf("%w: decode message: %w", errRealtimeTTS, err)
	}
	if msg.Error != "" {
		return nil, fmt.Errorf("%w: %s", errRealtimeTTS, msg.Error)
	}
	return &msg, nil
}

var (
	_ tts.Synthesizer             = (*realtimeSynthesizer)(nil)
	_ tts.ContextSynthesizer      = (*realtimeSynthesizer)(nil)
	_ tts.TurnContextCompleter    = (*realtimeSynthesizer)(nil)
	_ tts.AudioContextInterrupter = (*realtimeSynthesizer)(nil)
	_ tts.AudioContextCompleter   = (*realtimeSynthesizer)(nil)
	_ tts.WordTimestamps          = (*timedRealtimeSynthesizer)(nil)
	_ tts.ContextSynthesizer      = (*timedRealtimeSynthesizer)(nil)
)

// RunTTSTimed sends the sentence like RunTTS, implementing tts.WordTimestamps.
// The timings arrive on the reader goroutine with the audio, so neither
// callback is used here.
func (s *timedRealtimeSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	_ func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, nil)
}
