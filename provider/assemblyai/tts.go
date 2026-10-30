package assemblyai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
	errs "github.com/gojargo/jargo/utils/errors"
	ttstext "github.com/gojargo/jargo/utils/text"
)

const (
	// ttsURL is the Streaming TTS WebSocket in the US.
	ttsURL = "wss://tts.us.assemblyai.com/v1/ws"
	// defaultTTSVoiceID is Audrey, a voice from AssemblyAI's public voice
	// library.
	defaultTTSVoiceID = "49c443a4-8a1d-5f7e-b70b-98bdbd9823c3"
	// defaultTTSSampleRate is the PCM rate requested when the caller names none.
	defaultTTSSampleRate = 24000
	// ttsUserAgent identifies the client to AssemblyAI on the handshake.
	ttsUserAgent = "AssemblyAI/1.0 (integration=jargo)"

	// maxGenerateChars is the longest text one Generate message accepts, when
	// Begin doesn't report a limit.
	maxGenerateChars = 2000

	// sessionRenewalMargin is how long before a session's expires_at to replace
	// it between requests, at most half the session's length. The server stops
	// reading text at expires_at, which would cut off a reply that straddles it.
	sessionRenewalMargin = 120 * time.Second

	// keepaliveInterval is how often to send KeepAlive. Every session closes
	// after 120 s without a frame either way, and a bot can be silent that long
	// while the user speaks.
	keepaliveInterval = 30 * time.Second

	// wordBoundariesGrace is how long to wait, after a turn's last FlushDone, for
	// the WordBoundaries frame that trails it. It is best-effort, so the turn
	// closes without it after this.
	wordBoundariesGrace = time.Second

	// receiveStopTimeout bounds the wait for the receive loop to end when a
	// session is closed.
	receiveStopTimeout = time.Second

	// cancelWriteTimeout bounds the write that cancels the synthesis an
	// interruption cut off. It runs off the frame's context, which the
	// interruption may already have canceled.
	cancelWriteTimeout = 2 * time.Second

	// Error codes the service tells apart.
	errorCodeUnauthorized = 1008
	errorCodeRateLimited  = 3009
)

// The message types the Streaming TTS protocol carries, under fieldType.
const (
	fieldType = "type"

	ttsMsgBegin          = "Begin"
	ttsMsgAudio          = "Audio"
	ttsMsgFlushDone      = "FlushDone"
	ttsMsgWordBoundaries = "WordBoundaries"
	ttsMsgError          = "Error"
	ttsMsgGenerate       = "Generate"
	ttsMsgFlush          = "Flush"
	ttsMsgCancel         = "Cancel"
	ttsMsgKeepAlive      = "KeepAlive"
)

// routineErrorTypes are the retryable errors that end a session in the normal
// course of things.
//
//nolint:gochecknoglobals // fixed table
var routineErrorTypes = map[string]bool{
	"inactivity_timeout": true,
	"session_expired":    true,
	"service_restart":    true,
}

// errReconnecting is returned for text sent while the receive loop is
// redialing, and so owns the connection.
//
//nolint:gochecknoglobals // sentinel error
var errReconnecting = errors.New("assemblyai tts: reconnecting; text not spoken")

// TTSConfig configures the AssemblyAI streaming TTS service.
type TTSConfig struct {
	// APIKey is the AssemblyAI API key. Required. A temporary token opens a
	// single session, so it can't carry the service's reconnects and session
	// renewals.
	APIKey string `validate:"required"`
	// URL overrides the Streaming TTS WebSocket; empty uses the US host. Use
	// wss://tts.eu.assemblyai.com/v1/ws to process text and audio in the EU. A
	// cloned voice works only on the host of the zone it was created in.
	URL string
	// VoiceID is the voice: any voice by its ID, an AssemblyAI voice by its
	// library slug, such as "aai-jane", or a cloned voice by the slug the
	// AssemblyAI dashboard shows. Empty uses Audrey.
	VoiceID string
	// Model names the model, such as "assemblyai-tts-1"; empty leaves it to the
	// server, which runs the language's default model.
	Model string
	// Language is the language the session speaks; the zero value leaves it to
	// the server, which speaks the voice's own language.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.TextTypeSentence (the zero value) sends each sentence as its own
	// request, frames.TextTypeToken streams the turn's text as it is generated
	// and flushes it when the turn ends.
	TextAggregation frames.TextType
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// withDefaults fills in what the caller left unset.
func (c TTSConfig) withDefaults() TTSConfig {
	if c.URL == "" {
		c.URL = ttsURL
	}
	if c.VoiceID == "" {
		c.VoiceID = defaultTTSVoiceID
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSSampleRate
	}
	if c.TextAggregation == "" {
		c.TextAggregation = frames.TextTypeSentence
	}
	return c
}

// NewTTS builds an AssemblyAI streaming TTS service.
//
// It streams text to AssemblyAI over a single WebSocket session and plays the
// audio as it arrives. Each sentence is sent as its own request (a Generate
// followed by a Flush). With frames.TextTypeToken the turn's text streams as it
// is generated and is flushed when the turn ends. Interruptions send Cancel,
// which keeps the session open, and per-word timings drive word-level
// TTSTextFrame output.
//
// The session is configured at connect time, so changing the voice, language or
// model reconnects. A session lasts at most an hour; the service replaces it
// between requests before it expires.
func NewTTS(cfg TTSConfig) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns the synthesizer driving it.
func newTTS(cfg TTSConfig) (*tts.Base, *synthesizer) {
	cfg = cfg.withDefaults()
	s := &synthesizer{
		cfg:                  cfg,
		live:                 newTTSSettings(cfg),
		maxGenerateChars:     maxGenerateChars,
		requests:             map[int]string{},
		contexts:             map[string]*ttsContextState{},
		wordContexts:         map[int]string{},
		keepaliveInterval:    keepaliveInterval,
		wordBoundariesGrace:  wordBoundariesGrace,
		sessionRenewalMargin: sessionRenewalMargin,
		now:                  time.Now,
	}
	s.ws = wsservice.New(s, wsservice.Config{})
	b := tts.New("AssemblyAITTS", s)
	s.svc = b
	if cfg.TextAggregation == frames.TextTypeToken {
		b.SetTextAggregator(ttstext.NewTokenAggregator())
	}
	// Frames are held behind a reply until its audio has all arrived.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s
}

// ttsContextState is the bookkeeping for one audio context.
type ttsContextState struct {
	// audioMs is the playback length of the context's finished requests, where
	// the next request's audio starts within the context.
	audioMs float64
	// turnDone reports whether the turn has ended, so no more text will be sent.
	turnDone bool
	// awaitingWords maps the flush IDs whose WordBoundaries frame is still due to
	// where each request's audio starts within the context.
	awaitingWords map[int]float64
	// grace closes the context if WordBoundaries never come.
	grace *graceTimer
}

// graceTimer is the wait for a context's WordBoundaries. It is compared by
// identity, so a timer that fires after being replaced finds it is stale.
type graceTimer struct {
	t *time.Timer
}

// synthesizer speaks over one WebSocket session at a time. The receive loop is
// run by a wsservice.Base, which reconnects when the session closes.
type synthesizer struct {
	cfg  TTSConfig
	live *TTSSettings
	// svc is the service this synthesizer drives, for reporting errors and
	// asking whether it is usable. It is also the host audio is appended to.
	svc  *tts.Base
	host tts.AudioContextHost
	ws   *wsservice.Base

	// The waits, held here so a test can shorten them.
	keepaliveInterval    time.Duration
	wordBoundariesGrace  time.Duration
	sessionRenewalMargin time.Duration
	now                  func() time.Time

	// writeMu serializes writes; the connection permits one writer at a time,
	// and a Flush or Cancel takes its number in the order it is written.
	writeMu sync.Mutex

	// mu guards everything below. It is never held across a call into the
	// host or the service.
	mu   sync.Mutex
	conn *wsutil.Conn
	// closed records that reading from conn failed, so the connection is gone
	// although it has not been disconnected yet.
	closed bool
	// receiveStop and receiveDone stop and wait for the receive loop.
	receiveStop context.CancelFunc
	receiveDone chan struct{}
	// keepaliveStop stops the keepalive loop.
	keepaliveStop context.CancelFunc

	// Session state reported by Begin. A session ID means Begin arrived.
	sessionID        string
	renewAt          time.Time
	maxGenerateChars int
	wordBoundaries   bool
	// reconnectDelay is the wait an Error asked for before the next connection
	// attempt.
	reconnectDelay time.Duration

	// nextFlushID is the number of the request text is being written to now.
	// The server numbers requests from 0 per session, and each Flush and Cancel
	// takes the next number.
	nextFlushID int
	// requests maps flush ID to context, for requests whose FlushDone is still
	// due.
	requests map[int]string
	contexts map[string]*ttsContextState
	// wordContexts maps flush ID to context, for the WordBoundaries that follows
	// a FlushDone.
	wordContexts map[int]string
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *synthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(""), VoiceID: s.live.Voice.Or("")}
}

// Start connects when the pipeline starts, implementing tts.Starter. Connecting
// here, rather than on the first sentence, keeps the connection setup off the
// first response's latency.
func (s *synthesizer) Start(ctx context.Context) {
	s.connect(ctx)
}

// Close disconnects, implementing tts.Closer.
func (s *synthesizer) Close() error {
	s.disconnect(context.Background())
	return nil
}

// buildURL builds the connect URL, which carries the whole session
// configuration.
func (s *synthesizer) buildURL() string {
	q := url.Values{}
	q.Set("sample_rate", strconv.Itoa(s.cfg.SampleRate))
	q.Set("encoding", defaultEncoding)
	q.Set("word_boundaries", "true")
	// Unset values are left to the server's defaults.
	if voice := s.live.Voice.Or(""); voice != "" {
		q.Set("voice_id", voice)
	}
	if lang := s.live.Language.Or(""); lang != "" {
		q.Set("language", lang)
	}
	if model := s.live.Model.Or(""); model != "" {
		q.Set("model_id", model)
	}
	return s.cfg.URL + "?" + q.Encode()
}

// connect connects to AssemblyAI and starts the receive and keepalive loops.
func (s *synthesizer) connect(ctx context.Context) {
	s.ws.Connect()
	s.mu.Lock()
	s.reconnectDelay = 0
	s.mu.Unlock()
	s.openSession(ctx)
	s.startKeepalive(ctx)
}

// disconnect disconnects from AssemblyAI and stops the receive and keepalive
// loops.
func (s *synthesizer) disconnect(ctx context.Context) {
	s.closeConnectionContexts()
	s.ws.Disconnect()
	s.stopKeepalive()
	s.closeSession(ctx)
}

// openSession opens the WebSocket session and starts the receive loop on it.
func (s *synthesizer) openSession(ctx context.Context) {
	_ = s.ConnectWebsocket(ctx)
	s.mu.Lock()
	if s.conn == nil || s.receiveStop != nil {
		s.mu.Unlock()
		return
	}
	// The loop outlives the call that opened the session: it runs until the
	// session is closed, not until this sentence is done.
	runCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	s.receiveStop, s.receiveDone = stop, done
	s.mu.Unlock()
	go func() {
		defer close(done)
		s.ws.ReceiveTaskHandler(runCtx, s.reportError)
		s.mu.Lock()
		if s.receiveDone == done {
			s.receiveStop, s.receiveDone = nil, nil
		}
		s.mu.Unlock()
	}()
}

// closeSession stops the receive loop and closes the session.
func (s *synthesizer) closeSession(ctx context.Context) {
	s.mu.Lock()
	stop, done := s.receiveStop, s.receiveDone
	s.receiveStop, s.receiveDone = nil, nil
	s.mu.Unlock()
	if stop != nil {
		stop()
		select {
		case <-done:
		case <-time.After(receiveStopTimeout):
			slog.Debug("assemblyai tts receive loop did not stop in time")
		}
	}
	s.disconnectWebsocket(ctx)
}

// renewSessionIfExpiring replaces the session ahead of its expiry, while
// nothing is in flight. Audio contexts stay open, so a turn in progress carries
// on in the new session.
func (s *synthesizer) renewSessionIfExpiring(ctx context.Context) {
	s.mu.Lock()
	due := !s.renewAt.IsZero() && !s.now().Before(s.renewAt)
	idle := len(s.requests) == 0 && len(s.wordContexts) == 0
	session := s.sessionID
	s.mu.Unlock()
	if !due || !idle {
		return
	}
	slog.Debug("assemblyai tts session expires soon; opening a new one", "session", session)
	s.closeSession(ctx)
	s.openSession(ctx)
}

// ConnectWebsocket opens the WebSocket session, implementing wsservice.Handler.
// A failure is reported here and leaves no connection behind, which the
// receive loop reads as a failed attempt.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	if s.Connected() {
		return nil
	}
	endpoint := s.buildURL()
	slog.Debug("connecting to assemblyai tts", "url", endpoint)
	header := authHeader(s.cfg.APIKey)
	header.Set("User-Agent", ttsUserAgent)
	conn, err := wsutil.Dial(ctx, endpoint, header, readLimit)
	if err != nil {
		s.svc.PushError(ctx, fmt.Sprintf("Error connecting to AssemblyAI TTS: %v", err), err, false)
		return nil
	}
	s.mu.Lock()
	s.conn, s.closed = conn, false
	s.mu.Unlock()
	return nil
}

// DisconnectWebsocket readies a redial, implementing wsservice.Handler. The
// receive loop calls it before every reconnect attempt: it first waits as long
// as the last Error asked, then ends the audio contexts of the connection being
// replaced, and closes the socket.
func (s *synthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.mu.Lock()
	delay := s.reconnectDelay
	s.reconnectDelay = 0
	s.mu.Unlock()
	if delay > 0 {
		slog.Debug("waiting before reconnecting to assemblyai tts, as the server asked", "delay", delay)
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
		case <-timer.C:
		}
		timer.Stop()
	}
	s.closeConnectionContexts()
	s.disconnectWebsocket(ctx)
	return nil
}

// disconnectWebsocket closes the WebSocket session and forgets its state.
//
// Closing discards synthesis still in flight, which is what a disconnect wants;
// by the end of a pipeline every reply has already been played.
func (s *synthesizer) disconnectWebsocket(ctx context.Context) {
	s.mu.Lock()
	conn, open := s.conn, s.conn != nil && !s.closed
	s.conn, s.closed = nil, false
	s.resetSessionState()
	s.mu.Unlock()
	if !open {
		// Nothing to close, or a connection that is gone already.
		return
	}
	slog.Debug("disconnecting from assemblyai tts")
	if err := conn.Close(websocket.StatusNormalClosure, ""); err != nil {
		s.svc.PushError(ctx, fmt.Sprintf("Error disconnecting: %v", err), err, false)
	}
}

// resetSessionState forgets per-session state, which a new session doesn't
// carry over. Callers hold mu.
func (s *synthesizer) resetSessionState() {
	s.nextFlushID = 0
	clear(s.requests)
	clear(s.wordContexts)
	s.sessionID = ""
	s.renewAt = time.Time{}
	s.maxGenerateChars = maxGenerateChars
	s.wordBoundaries = false
}

// closeConnectionContexts ends the audio contexts of a connection that is being
// replaced, forgetting their bookkeeping first.
func (s *synthesizer) closeConnectionContexts() {
	s.mu.Lock()
	ids := make([]string, 0, len(s.contexts))
	for id := range s.contexts {
		ids = append(ids, id)
	}
	s.mu.Unlock()
	for _, id := range ids {
		s.forgetContext(id)
	}
	if s.host != nil {
		s.host.ReplaceConnection()
	}
}

// ReceiveMessages receives and dispatches server frames until the session
// ends, implementing wsservice.Handler.
func (s *synthesizer) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return wsservice.ErrNotConnected
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
		var m ttsServerMessage
		if err := json.Unmarshal(data, &m); err != nil {
			slog.Warn("assemblyai tts sent an invalid JSON message", "message", string(data))
			continue
		}
		switch m.Type {
		case ttsMsgAudio:
			s.handleAudio(&m)
		case ttsMsgFlushDone:
			s.handleFlushDone(&m)
		case ttsMsgWordBoundaries:
			s.handleWordBoundaries(&m)
		case ttsMsgBegin:
			s.handleBegin(&m)
		case ttsMsgError:
			s.handleError(ctx, &m)
		default:
			// The protocol may add frame types, which are safe to ignore.
			slog.Debug("assemblyai tts sent an unhandled message", "message", string(data))
		}
	}
}

// Connected reports whether there is a live session, implementing
// wsservice.Handler.
func (s *synthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.closed
}

// Usable reports whether the service can still do its job, implementing
// wsservice.Handler.
func (s *synthesizer) Usable() bool { return s.svc.Usable() }

// reportError reports a connection error from the receive loop.
func (s *synthesizer) reportError(ctx context.Context, ef *frames.ErrorFrame, forceTreatAsPermanent bool) {
	s.svc.PushErrorFrame(ctx, ef, forceTreatAsPermanent)
}

// send writes one JSON message.
func (s *synthesizer) send(ctx context.Context, msg map[string]any) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.sendLocked(ctx, msg)
}

// sendLocked writes one JSON message. Callers hold writeMu.
func (s *synthesizer) sendLocked(ctx context.Context, msg map[string]any) error {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return wsservice.ErrNotConnected
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, payload)
}

// sendFlush concludes the request text is being written to, which takes the
// next number.
func (s *synthesizer) sendFlush(ctx context.Context) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	s.mu.Lock()
	s.nextFlushID++
	s.mu.Unlock()
	return s.sendLocked(ctx, map[string]any{fieldType: ttsMsgFlush})
}

// startKeepalive starts the keepalive loop if it isn't running.
func (s *synthesizer) startKeepalive(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.keepaliveStop != nil {
		return
	}
	runCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	s.keepaliveStop = stop
	go s.keepalive(runCtx, s.keepaliveInterval)
}

// stopKeepalive stops the keepalive loop if running.
func (s *synthesizer) stopKeepalive() {
	s.mu.Lock()
	stop := s.keepaliveStop
	s.keepaliveStop = nil
	s.mu.Unlock()
	if stop != nil {
		stop()
	}
}

// keepalive sends KeepAlive every interval while connected, so an idle session
// stays open. A failed send is logged and the loop keeps going: the receive
// loop handles a lost connection, and the next keepalive goes out on the
// reconnected socket.
func (s *synthesizer) keepalive(ctx context.Context, interval time.Duration) {
	tick := time.NewTicker(interval)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		if !s.Connected() {
			continue
		}
		if err := s.send(ctx, map[string]any{fieldType: ttsMsgKeepAlive}); err != nil && ctx.Err() == nil {
			slog.Warn("assemblyai tts keepalive failed", "err", err)
		}
	}
}

// contextState returns a context's bookkeeping, creating it on first use.
// Callers hold mu.
func (s *synthesizer) contextState(contextID string) *ttsContextState {
	st, ok := s.contexts[contextID]
	if !ok {
		st = &ttsContextState{awaitingWords: map[int]float64{}}
		s.contexts[contextID] = st
	}
	return st
}

// requestsFor reports whether a request for the context still awaits its
// FlushDone. Callers hold mu.
func (s *synthesizer) requestsFor(contextID string) bool {
	for _, owner := range s.requests {
		if owner == contextID {
			return true
		}
	}
	return false
}

// maybeCloseContext closes a context once its turn has ended and the server
// owes it nothing. That is: every Flush answered with FlushDone, and every
// WordBoundaries frame due arrived, or the grace period for one ran out.
func (s *synthesizer) maybeCloseContext(contextID string) {
	s.mu.Lock()
	st := s.contexts[contextID]
	if st == nil || !st.turnDone || s.requestsFor(contextID) {
		s.mu.Unlock()
		return
	}
	if len(st.awaitingWords) > 0 {
		if st.grace == nil {
			g := &graceTimer{}
			g.t = time.AfterFunc(s.wordBoundariesGrace, func() { s.wordBoundariesGraceExpired(contextID, g) })
			st.grace = g
		}
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()
	s.closeContext(contextID)
}

// wordBoundariesGraceExpired closes a context whose WordBoundaries never came.
func (s *synthesizer) wordBoundariesGraceExpired(contextID string, g *graceTimer) {
	s.mu.Lock()
	st := s.contexts[contextID]
	if st == nil || st.grace != g {
		s.mu.Unlock()
		return
	}
	st.grace = nil
	awaiting := maps.Clone(st.awaitingWords)
	s.mu.Unlock()
	slog.Debug("assemblyai tts sent no WordBoundaries", "flushes", awaiting)
	s.closeContext(contextID)
}

// forgetContext drops a context's bookkeeping, including WordBoundaries still
// due for it. It reports whether there was any.
func (s *synthesizer) forgetContext(contextID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	st, ok := s.contexts[contextID]
	delete(s.contexts, contextID)
	if ok && st.grace != nil {
		st.grace.t.Stop()
		st.grace = nil
	}
	maps.DeleteFunc(s.wordContexts, func(_ int, owner string) bool { return owner == contextID })
	return ok
}

// closeContext ends a context: its stop frame rides the queue behind its
// audio. A context already forgotten was closed by whoever forgot it.
func (s *synthesizer) closeContext(contextID string) {
	if !s.forgetContext(contextID) {
		return
	}
	if s.host != nil && s.host.AudioContextAvailable(contextID) {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		s.host.AppendToAudioContext(contextID, stopped)
		s.host.RemoveAudioContext(contextID)
	}
}

// FlushAudio concludes the turn's text so the context can close once it is
// spoken, implementing tts.AudioFlusher. In sentence aggregation each sentence
// is already flushed by RunTTS. When streaming tokens, the turn's text is
// flushed here.
func (s *synthesizer) FlushAudio(ctx context.Context, contextID string) {
	if contextID == "" {
		return
	}
	s.mu.Lock()
	pending := s.requests[s.nextFlushID] == contextID
	s.mu.Unlock()
	if pending {
		if err := s.sendFlush(ctx); err != nil {
			slog.Warn("assemblyai tts flush failed", "err", err)
		}
	}
	s.mu.Lock()
	s.contextState(contextID).turnDone = true
	s.mu.Unlock()
	s.maybeCloseContext(contextID)
}

// OnAudioContextInterrupted cancels the server's in-flight synthesis when the
// bot is interrupted, implementing tts.AudioContextInterrupter. Cancel discards
// everything the server has not yet delivered and keeps the session open, so
// the next response needs no reconnect.
func (s *synthesizer) OnAudioContextInterrupted(_ context.Context, contextID string) {
	s.mu.Lock()
	inFlight := len(s.requests) > 0
	// Audio still in transit for these requests is dropped by flush_id.
	clear(s.requests)
	s.mu.Unlock()
	if inFlight {
		ctx, cancel := context.WithTimeout(context.Background(), cancelWriteTimeout)
		if err := s.sendCancel(ctx); err != nil {
			slog.Warn("assemblyai tts cancel failed", "err", err)
		}
		cancel()
	}
	s.forgetContext(contextID)
}

// sendCancel discards the synthesis in flight, which takes the next number.
// Nothing is sent without a session.
func (s *synthesizer) sendCancel(ctx context.Context) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	s.mu.Lock()
	if s.conn == nil || s.closed {
		s.mu.Unlock()
		return nil
	}
	s.nextFlushID++
	s.mu.Unlock()
	return s.sendLocked(ctx, map[string]any{fieldType: ttsMsgCancel})
}

// OnAudioContextCompleted forgets a context's bookkeeping once it has finished
// playing, implementing tts.AudioContextCompleter.
//
// A context that goes quiet mid-turn ends on the base's timeout and is
// recreated when more audio arrives, with word times measured from its new
// first audio, so its offset starts again from zero.
func (s *synthesizer) OnAudioContextCompleted(_ context.Context, contextID string) {
	s.forgetContext(contextID)
}

// handleBegin records the session Begin describes.
func (s *synthesizer) handleBegin(m *ttsServerMessage) {
	s.mu.Lock()
	s.sessionID = m.ID
	if m.ExpiresAt > 0 {
		sec, frac := math.Modf(m.ExpiresAt)
		expires := time.Unix(int64(sec), int64(frac*float64(time.Second)))
		margin := min(s.sessionRenewalMargin, expires.Sub(s.now())/2)
		s.renewAt = expires.Add(-margin)
	}
	s.maxGenerateChars = maxGenerateChars
	s.wordBoundaries = false
	if c := m.Configuration; c != nil {
		if c.Limits != nil && c.Limits.MaxGenerateTextLength > 0 {
			s.maxGenerateChars = c.Limits.MaxGenerateTextLength
		}
		s.wordBoundaries = c.WordBoundaries
	}
	s.mu.Unlock()
	slog.Debug("assemblyai tts session started", "session", m.ID, "configuration", m.Configuration)
}

// handleAudio appends a request's audio to the context it belongs to.
func (s *synthesizer) handleAudio(m *ttsServerMessage) {
	s.mu.Lock()
	contextID := s.requests[m.FlushID]
	s.mu.Unlock()
	if contextID == "" || s.host == nil || !s.host.AudioContextAvailable(contextID) {
		return
	}
	pcm, err := base64.StdEncoding.DecodeString(m.Audio)
	if err != nil {
		slog.Warn("assemblyai tts sent audio that could not be decoded", "err", err)
		return
	}
	f := frames.NewTTSAudioRawFrame(pcm, s.cfg.SampleRate, 1)
	f.ContextID = contextID
	s.host.AppendToAudioContext(contextID, f)
}

// handleFlushDone settles a request: its audio is all in, and its words follow
// when the session reports them.
func (s *synthesizer) handleFlushDone(m *ttsServerMessage) {
	s.mu.Lock()
	contextID, ok := s.requests[m.FlushID]
	delete(s.requests, m.FlushID)
	if !ok || contextID == "" {
		s.mu.Unlock()
		return
	}
	st := s.contextState(contextID)
	if s.wordBoundaries && m.AudioDurationMs != 0 {
		s.wordContexts[m.FlushID] = contextID
		st.awaitingWords[m.FlushID] = st.audioMs
	}
	st.audioMs += m.AudioDurationMs
	s.mu.Unlock()
	s.maybeCloseContext(contextID)
}

// handleWordBoundaries reports a request's words, placed on the context's
// timeline.
func (s *synthesizer) handleWordBoundaries(m *ttsServerMessage) {
	s.mu.Lock()
	contextID, ok := s.wordContexts[m.FlushID]
	delete(s.wordContexts, m.FlushID)
	st := s.contexts[contextID]
	if !ok || st == nil {
		s.mu.Unlock()
		return
	}
	start, due := st.awaitingWords[m.FlushID]
	if !due {
		s.mu.Unlock()
		return
	}
	delete(st.awaitingWords, m.FlushID)
	s.mu.Unlock()

	// Word times are on the session's timeline; audio_start_ms is where this
	// request's audio starts on it.
	offsetMs := start - m.AudioStartMs
	words := make([]uctx.WordTiming, 0, len(m.Words))
	for _, w := range m.Words {
		words = append(words, uctx.WordTiming{Word: w.Word, Offset: (offsetMs + w.Start) / 1000})
	}
	if len(words) > 0 && s.host != nil {
		s.host.AddWordTimestamps(contextID, words, tts.WordTimingOptions{})
	}
	s.maybeCloseContext(contextID)
}

// handleError reports an Error frame. The server closes the session right
// after it.
//
// A refusal before Begin that isn't retryable is in the session's own
// configuration (the credential, voice, model or a parameter), so it leaves the
// service unusable. Retryable errors are retried by the receive loop's
// reconnect, after any wait the server asks for, and other errors after Begin
// concern the text sent.
func (s *synthesizer) handleError(ctx context.Context, m *ttsServerMessage) {
	label := strconv.Itoa(m.ErrorCode)
	if m.ErrorType != "" {
		label += " " + m.ErrorType
	}
	text := fmt.Sprintf("AssemblyAI TTS error %s: %s", label, m.Error)

	s.mu.Lock()
	begun := s.sessionID != ""
	if m.Retryable {
		s.reconnectDelay = time.Duration(m.RetryAfterSeconds * float64(time.Second))
	}
	s.mu.Unlock()

	var category errs.Category
	switch {
	case m.Retryable:
		if routineErrorTypes[m.ErrorType] {
			slog.Debug(text)
			return
		}
		category = errs.Server
		if m.ErrorCode == errorCodeRateLimited {
			category = errs.RateLimit
		}
	case !begun:
		category = errs.InvalidRequest
		if m.ErrorCode == errorCodeUnauthorized {
			category = errs.Authentication
		}
	default:
		category = errs.Unknown
	}
	s.svc.PushError(ctx, text, nil, false, processor.WithErrorCategory(category))
}

// RunTTS sends text to AssemblyAI for synthesis. It yields nothing: the audio
// arrives on the receive loop, which appends it to the context it belongs to.
// In sentence aggregation each call is flushed as its own request, since only a
// flush guarantees the server speaks all of the text.
func (s *synthesizer) RunTTS(
	ctx context.Context, text, contextID string, yield func(f frames.Frame) error,
) error {
	if s.ws.ReconnectInProgress() {
		// The receive loop owns the connection until it reconnects.
		s.yieldStopped(contextID, yield)
		return errReconnecting
	}

	if err := s.sendText(ctx, text, contextID); err != nil {
		s.yieldStopped(contextID, yield)
		s.disconnect(ctx)
		s.connect(ctx)
		return fmt.Errorf("assemblyai tts: %w", err)
	}
	return nil
}

// sendText sends one request's text, connecting first if there is no session
// and renewing the session if it is about to expire.
func (s *synthesizer) sendText(ctx context.Context, text, contextID string) error {
	if s.Connected() {
		s.renewSessionIfExpiring(ctx)
	} else {
		// A connection that is gone already is dropped first, along with the
		// state of the session it carried.
		s.disconnectWebsocket(ctx)
		s.connect(ctx)
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	// Recorded before sending, so audio that arrives while the messages are
	// still being sent is attributed to this context.
	s.mu.Lock()
	s.requests[s.nextFlushID] = contextID
	size := s.maxGenerateChars
	s.mu.Unlock()
	// Consecutive Generate texts are joined exactly as sent.
	for _, part := range splitRunes(text, size) {
		if err := s.sendLocked(ctx, map[string]any{fieldType: ttsMsgGenerate, "text": part}); err != nil {
			return err
		}
	}
	if s.svc.IsStreamingTokens() {
		return nil
	}
	s.mu.Lock()
	s.nextFlushID++
	s.mu.Unlock()
	return s.sendLocked(ctx, map[string]any{fieldType: ttsMsgFlush})
}

// yieldStopped ends the context of text that will not be spoken.
func (s *synthesizer) yieldStopped(contextID string, yield func(f frames.Frame) error) {
	if yield == nil {
		return
	}
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	_ = yield(stopped)
}

// splitRunes cuts text into pieces of at most size characters.
func splitRunes(text string, size int) []string {
	runes := []rune(text)
	if size <= 0 || len(runes) <= size {
		return []string{text}
	}
	parts := make([]string, 0, (len(runes)+size-1)/size)
	for start := 0; start < len(runes); start += size {
		parts = append(parts, string(runes[start:min(start+size, len(runes))]))
	}
	return parts
}

// RunTTSTimed sends the text like RunTTS, implementing tts.WordTimestamps. The
// timings arrive on the receive loop with the audio, so neither callback is
// used here; implementing the interface is what puts the base on the
// word-aligned path.
func (s *synthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	yield func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, yield)
}

// ttsServerMessage is the subset of a Streaming TTS server message we read.
type ttsServerMessage struct {
	Type string `json:"type"`

	// Begin.
	ID            string            `json:"id"`
	ExpiresAt     float64           `json:"expires_at"`
	Configuration *ttsConfiguration `json:"configuration"`

	// Audio, FlushDone and WordBoundaries.
	FlushID         int       `json:"flush_id"`
	Audio           string    `json:"audio"`
	AudioDurationMs float64   `json:"audio_duration_ms"`
	AudioStartMs    float64   `json:"audio_start_ms"`
	Words           []ttsWord `json:"words"`

	// Error.
	ErrorCode         int     `json:"error_code"`
	ErrorType         string  `json:"error_type"`
	Error             string  `json:"error"`
	Retryable         bool    `json:"retryable"`
	RetryAfterSeconds float64 `json:"retry_after_seconds"`
}

// ttsConfiguration is the part of a session's configuration Begin reports that
// the service reads.
type ttsConfiguration struct {
	WordBoundaries bool `json:"word_boundaries"`
	Limits         *struct {
		MaxGenerateTextLength int `json:"max_generate_text_length"`
	} `json:"limits"`
}

// ttsWord is one word a WordBoundaries frame places on the session timeline,
// with its start in milliseconds.
type ttsWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
}

// assemblyaiTTSLanguages are the languages the Streaming TTS API speaks, each
// under the full English name it takes, unlike the STT APIs, which take short
// codes.
//
//nolint:gochecknoglobals // lookup table, read-only
var assemblyaiTTSLanguages = map[language.Language]string{
	language.German:     "german",
	language.English:    "english",
	language.Spanish:    "spanish",
	language.French:     "french",
	language.Italian:    "italian",
	language.Portuguese: "portuguese",
}

// assemblyaiTTSLanguage maps a Language to the name the Streaming TTS API
// takes for it. The zero value maps to "", which leaves the language to the
// server.
func assemblyaiTTSLanguage(l language.Language) string {
	if l == "" {
		return ""
	}
	return language.Resolve(l, assemblyaiTTSLanguages, true)
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*synthesizer)(nil)
	_ tts.WordTimestamps          = (*synthesizer)(nil)
	_ tts.AudioFlusher            = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*synthesizer)(nil)
	_ tts.AudioContextCompleter   = (*synthesizer)(nil)
	_ tts.Starter                 = (*synthesizer)(nil)
	_ tts.Closer                  = (*synthesizer)(nil)
	_ tts.Describer               = (*synthesizer)(nil)
	_ wsservice.Handler           = (*synthesizer)(nil)
)
