package gradium

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

// errSTTProtocol is returned when Gradium reports an STT error or an unexpected
// handshake reply.
//
//nolint:gochecknoglobals // sentinel error
var errSTTProtocol = errors.New("gradium: stt protocol error")

// errSTTNotConnected is returned for a flush asked of a session with no
// connection to send it on.
//
//nolint:gochecknoglobals // sentinel error
var errSTTNotConnected = errors.New("gradium: stt not connected")

const (
	defaultSTTURL   = "wss://api.gradium.ai/api/speech/asr"
	defaultSTTModel = "default"
	// defaultDelayInFrames delays text generation by 12 frames (960 ms), trading
	// latency for context.
	defaultDelayInFrames = 12
	// sttChunkMS is the audio chunk size streamed to the server.
	sttChunkMS = 80

	// The turn detection defaults: the end-pointing horizon watched, the
	// inactivity probability that opens and closes a turn, and how many step
	// messages to let pass after a flush before the signal is trusted again.
	defaultEOTHorizonS             = 3.0
	defaultEOTThreshold            = 0.5
	defaultPostFlushCooldownFrames = 8

	// msgStep carries the server's end-pointing signal.
	msgStep = "step"
	// msgFlush asks the server to process the audio it is holding.
	msgFlush = "flush"
)

// Events a Gradium STT service raises with turn detection on, each with the
// service as its source.
const (
	// EventTurnStart fires when the end-pointing signal opens a turn.
	EventTurnStart = "on_turn_start"
	// EventTurnEnd fires when the end-pointing signal closes the turn.
	EventTurnEnd = "on_turn_end"
)

// turnDetectionFields are the settings read as step messages arrive; a change
// to any of them takes effect on the next step without reopening the session.
//
//nolint:gochecknoglobals // a fixed list
var turnDetectionFields = []string{"eot_horizon_s", "eot_threshold", "post_flush_cooldown_frames"}

// STTConfig configures the Gradium STT service.
type STTConfig struct {
	// APIKey is the Gradium API key, sent as the x-api-key header. Required.
	APIKey string `validate:"required"`
	// URL overrides the ASR WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the transcription model; empty uses a default.
	Model string
	// Encoding is the base audio encoding ("pcm", "wav", or "opus"); empty uses
	// "pcm". For PCM the sample rate is appended to form the input format (for
	// example "pcm_16000"). Audio at a rate Gradium does not accept is resampled
	// to one it does.
	Encoding string
	// Language grounds the transcription in one language; the zero value uses
	// English. Verified codes are de, en, es, fr, and pt; other languages fall
	// back to the base code. Set it to AnyLanguage to have Gradium detect the
	// language instead.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// DelayInFrames delays text generation by this many 80 ms frames; nil uses a
	// default. Allowed values: 7, 8, 10, 12, 14, 16, 20, 24, 36, 48.
	DelayInFrames *int `validate:"omitempty,oneof=7 8 10 12 14 16 20 24 36 48"`

	// EnableTurnDetection lets Gradium's end-pointing signal decide when user
	// turns start and end, instead of the pipeline's VAD. The service then
	// proposes each turn's start and end, flushes the server at the end of a
	// turn, and recommends the external turn strategies, which resolve the
	// proposals into the user's turn. Off by default.
	EnableTurnDetection bool
	// EOTHorizonS picks the end-pointing horizon that drives turn decisions:
	// the step entry whose horizon is closest to it is watched. Nil uses 3
	// seconds with turn detection on.
	EOTHorizonS *float64
	// EOTThreshold is the inactivity probability on that horizon at or above
	// which an open turn ends, and below which a turn opens once the signal has
	// read inactive. Nil uses 0.5 with turn detection on.
	EOTThreshold *float64
	// PostFlushCooldownFrames is how many step messages to ignore after a flush
	// before the next turn may start or end. A flush feeds the model silence,
	// and the signal is unreliable on the frames that follow: it can dip as if
	// speech resumed, then fire again on the first real frames. Nil uses 8 with
	// turn detection on.
	PostFlushCooldownFrames *int

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.GradiumTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// STTSettings is the part of the Gradium STT configuration that can change
// while the pipeline runs. The model, the language and the decoder delay are
// bound to the session's setup message, so a change to any of them reopens the
// session. The turn detection fields are read as step messages arrive and take
// effect on the next one.
type STTSettings struct {
	settings.STT

	// DelayInFrames delays text generation by this many 80 ms frames.
	DelayInFrames settings.Opt[int] `settings:"delay_in_frames"`
	// EOTHorizonS picks the end-pointing horizon that drives turn decisions.
	EOTHorizonS settings.Opt[float64] `settings:"eot_horizon_s"`
	// EOTThreshold is the inactivity probability that opens and ends a turn.
	EOTThreshold settings.Opt[float64] `settings:"eot_threshold"`
	// PostFlushCooldownFrames is how many step messages to ignore after a flush.
	PostFlushCooldownFrames settings.Opt[int] `settings:"post_flush_cooldown_frames"`
}

// newSTTSettings is the starting state, taken from what the service was built
// with. The turn detection fields hold no value unless turn detection is on,
// which gives them their defaults; a value the configuration gives wins over a
// default.
func newSTTSettings(cfg STTConfig) *STTSettings {
	s := &STTSettings{}
	s.Model = settings.Set(cfg.Model)
	if cfg.Language != "" {
		s.Language = settings.Set(cfg.Language.Code())
	}
	delay := defaultDelayInFrames
	if cfg.DelayInFrames != nil {
		delay = *cfg.DelayInFrames
	}
	s.DelayInFrames = settings.Set(delay)

	s.EOTHorizonS = settings.Cleared[float64]()
	s.EOTThreshold = settings.Cleared[float64]()
	s.PostFlushCooldownFrames = settings.Cleared[int]()
	if cfg.EnableTurnDetection {
		s.EOTHorizonS = settings.Set(defaultEOTHorizonS)
		s.EOTThreshold = settings.Set(defaultEOTThreshold)
		s.PostFlushCooldownFrames = settings.Set(defaultPostFlushCooldownFrames)
	}
	setIf(&s.EOTHorizonS, cfg.EOTHorizonS)
	setIf(&s.EOTThreshold, cfg.EOTThreshold)
	setIf(&s.PostFlushCooldownFrames, cfg.PostFlushCooldownFrames)
	return s
}

// setIf gives a setting the value v points to, when it points to one.
func setIf[T any](o *settings.Opt[T], v *T) {
	if v != nil {
		*o = settings.Set(*v)
	}
}

// NewSTT builds a Gradium streaming STT service.
//
// It transcribes English unless the configuration names another language. Set
// Language to AnyLanguage to have Gradium detect it instead.
//
// By default the pipeline's VAD closes each utterance: when it reports the user
// stopped, the server is flushed and the text gathered finalizes into the
// transcript. With EnableTurnDetection the server's end-pointing signal drives
// the turns instead, and the VAD is left to the turn strategies.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultSTTURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Encoding == "" {
		cfg.Encoding = encPCM
	}
	if cfg.Language == "" {
		cfg.Language = language.English
	}
	c := newSTTConnector(cfg)
	svc := stt.NewStream("GradiumSTT", c, cfg.SampleRate)
	svc.Events().Register(EventTurnStart, false)
	svc.Events().Register(EventTurnEnd, false)
	c.svc = svc
	return svc
}

// newSTTConnector builds the connector with its settings seeded.
func newSTTConnector(cfg STTConfig) *sttConnector {
	c := &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}
	c.refreshTurnParams()
	return c
}

type sttConnector struct {
	cfg STTConfig
	// live is what may change while the pipeline runs.
	live *STTSettings
	// svc is the service this connector drives, for raising its turn events.
	// It is nil for a connector built outside NewSTT.
	svc *stt.StreamService

	// mu guards what the sessions read from their own goroutine.
	mu sync.Mutex
	// turn is the turn detection settings as the sessions read them, copied
	// from live whenever it changes.
	turn turnParams
	// carryOpen records that a session dropped with a turn open, for the
	// session that replaces it to keep open.
	carryOpen bool
}

// turnParams is the turn detection settings a session reads on each step.
type turnParams struct {
	horizon, threshold float64
	// set reports whether both the horizon and the threshold have a value;
	// without them the signal decides nothing.
	set      bool
	cooldown int
}

// refreshTurnParams copies the turn detection settings for the sessions to
// read. It is called where the settings are held steady.
func (c *sttConnector) refreshTurnParams() {
	horizon, hasHorizon := c.live.EOTHorizonS.Value()
	threshold, hasThreshold := c.live.EOTThreshold.Value()
	cooldown, _ := c.live.PostFlushCooldownFrames.Value()
	c.mu.Lock()
	c.turn = turnParams{horizon: horizon, threshold: threshold, set: hasHorizon && hasThreshold, cooldown: cooldown}
	c.mu.Unlock()
}

// turnParams is the turn detection settings now in force.
func (c *sttConnector) turnParams() turnParams {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.turn
}

// keepTurnOpen records that the session dropped with a turn open. The user may
// still be speaking, so the session that replaces it carries the turn on, and
// its own end-pointing signal ends it.
func (c *sttConnector) keepTurnOpen() {
	c.mu.Lock()
	c.carryOpen = true
	c.mu.Unlock()
}

// takeOpenTurn reports whether the session being opened carries on a turn the
// one before it left open, and forgets it.
func (c *sttConnector) takeOpenTurn() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	open := c.carryOpen
	c.carryOpen = false
	return open
}

// raise fires one of the service's turn events.
func (c *sttConnector) raise(ctx context.Context, name string) {
	if c.svc != nil {
		c.svc.Events().Call(ctx, name, c.svc)
	}
}

// Metadata reports Gradium's time-to-final-segment latency to downstream
// processors. With turn detection on the service proposes the turn boundaries
// itself, so it recommends the external turn strategies, which resolve those
// proposals rather than running local detection, and a speech-end to
// transcript interval means nothing: the server defines where the turn ends.
func (c *sttConnector) Metadata() stt.Metadata {
	m := stt.Metadata{
		TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.GradiumTTFSP99),
		Model:   c.live.Model.Or(c.cfg.Model),
	}
	if c.cfg.EnableTurnDetection {
		noTTFS := false
		m.SupportsTTFS = &noTTFS
		m.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{})
	}
	return m
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *sttConnector) Settings() any { return c.live }

// UpdateSettings asks for the session to be reopened when anything bound to
// its setup message changed. The turn detection fields are read as step
// messages arrive, so a change to them alone reopens nothing.
func (c *sttConnector) UpdateSettings(_ context.Context, changed settings.Changed) (bool, error) {
	c.refreshTurnParams()
	return len(changed.Except(turnDetectionFields...)) > 0, nil
}

// ServiceLanguage names a language the way Gradium does.
func (c *sttConnector) ServiceLanguage(l language.Language) string { return gradiumLanguage(l) }

// gradiumPCMSampleRates are the PCM sample rates Gradium accepts, lowest first.
//
//nolint:gochecknoglobals // fixed lookup table
var gradiumPCMSampleRates = []int{8000, 16000, 24000}

// gradiumPCMSampleRate picks the PCM sample rate to send audio at: the lowest
// rate Gradium accepts at or above sampleRate, or the highest one it accepts.
func gradiumPCMSampleRate(sampleRate int) int {
	for _, rate := range gradiumPCMSampleRates {
		if rate >= sampleRate {
			return rate
		}
	}
	return gradiumPCMSampleRates[len(gradiumPCMSampleRates)-1]
}

// sendSampleRate is the rate audio is sent to Gradium at. PCM at a rate Gradium
// does not accept is resampled to one it does; other encodings are sent at the
// pipeline's rate.
func (c *sttConnector) sendSampleRate(sampleRate int) int {
	if c.cfg.Encoding != encPCM {
		return sampleRate
	}
	return gradiumPCMSampleRate(sampleRate)
}

// inputFormat builds Gradium's input_format from the encoding and sample rate.
// For PCM the sample rate is appended; other encodings are used as-is.
func (c *sttConnector) inputFormat(sampleRate int) string {
	if c.cfg.Encoding != encPCM {
		return c.cfg.Encoding
	}
	switch sampleRate {
	case 8000:
		return "pcm_8000"
	case 24000:
		return "pcm_24000"
	default:
		return "pcm_16000"
	}
}

// AnyLanguage asks Gradium to detect the language rather than being grounded in
// one. It is not one of the language constants because it names a mode rather
// than a language, and it is passed through as it is written.
const AnyLanguage language.Language = "any"

// gradiumSTTLanguages are the languages Gradium transcribes, each under the code
// it takes for it.
//
//nolint:gochecknoglobals // read-only lookup table
var gradiumSTTLanguages = map[language.Language]string{
	language.German:     "de",
	language.English:    "en",
	language.Spanish:    "es",
	language.French:     "fr",
	language.Portuguese: "pt",
}

// gradiumLanguage maps a Language to Gradium's language code. Gradium wants the
// base code; the empty language yields "" so the caller omits the hint, and the
// detect-the-language mode is passed through as it is.
func gradiumLanguage(l language.Language) string {
	switch l {
	case "":
		return ""
	case AnyLanguage:
		return string(AnyLanguage)
	default:
		return language.Resolve(l, gradiumSTTLanguages, true)
	}
}

// WebsocketOptions hands the stream base the connection settings the
// configuration gives, implementing wsservice.Optioner.
func (c *sttConnector) WebsocketOptions() wsservice.Options { return c.cfg.Websocket }

// Connect dials the WebSocket, sends the setup handshake, and waits for the
// server's ready acknowledgement. Audio is resampled from the pipeline's rate to
// the rate it is sent at.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	sendRate := c.sendSampleRate(sampleRate)
	rs, err := resample.New(sampleRate, sendRate, 1)
	if err != nil {
		return nil, err
	}

	header := http.Header{}
	header.Set("x-api-key", c.cfg.APIKey)

	conn, err := c.cfg.Websocket.Dial(ctx, c.cfg.URL, header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}

	if err := conn.Write(ctx, websocket.MessageText, c.setup(sendRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "setup failed")
		return nil, err
	}
	if err := awaitReady(ctx, conn); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "setup failed")
		return nil, err
	}

	chunkBytes := sttChunkMS * sendRate * 2 / 1000
	lang, _ := c.live.Language.Value()
	s := &sttStream{
		conn:          conn,
		ctx:           ctx,
		rs:            rs,
		chunkBytes:    chunkBytes,
		lang:          lang,
		owner:         c,
		turnDetection: c.cfg.EnableTurnDetection,
		reads:         make(chan sttRead, 8),
		done:          make(chan struct{}),
	}
	if c.takeOpenTurn() {
		s.phase = phaseOpen
	}
	go s.readLoop()
	return s, nil
}

// setup builds the initial setup message carrying the model, input format, and
// optional language and delay configuration.
func (c *sttConnector) setup(sampleRate int) []byte {
	msg := map[string]any{
		msgType:        "setup",
		"model_name":   c.live.Model.Or(c.cfg.Model),
		"input_format": c.inputFormat(sampleRate),
	}
	jsonConfig := map[string]any{}
	if code, ok := c.live.Language.Value(); ok {
		if lang := gradiumLanguage(language.Language(code)); lang != "" {
			jsonConfig["language"] = lang
		}
	}
	if delay, ok := c.live.DelayInFrames.Value(); ok && delay != 0 {
		jsonConfig["delay_in_frames"] = delay
	}
	if len(jsonConfig) > 0 {
		msg["json_config"] = jsonConfig
	}
	b, _ := json.Marshal(msg) //nolint:errchkjson // map of known-serializable values
	return b
}

// awaitReady reads the handshake reply and requires a "ready" message.
func awaitReady(ctx context.Context, conn *wsutil.Conn) error {
	_, data, err := conn.Read(ctx)
	if err != nil {
		return err
	}
	var m sttMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	switch m.Type {
	case "ready":
		return nil
	case "error":
		return fmt.Errorf("%w: %s", errSTTProtocol, m.Message)
	default:
		return fmt.Errorf("%w: unexpected first message %q", errSTTProtocol, m.Type)
	}
}

// transcriptAggregationDelay is how long the transcript stays open after a
// flush. The flush only says the buffered audio has been processed: the words at
// the end of it can still be on their way, so settling the transcript the moment
// it arrives drops the last of the utterance, and those words then turn up at
// the front of the next one.
const transcriptAggregationDelay = 100 * time.Millisecond

// turnPhase is where turn detection is in a turn's lifecycle.
type turnPhase int

const (
	// phaseIdle waits for the signal to read inactive.
	phaseIdle turnPhase = iota
	// phaseArmed opens a turn on the next dip of the signal.
	phaseArmed
	// phaseOpen is a turn in progress.
	phaseOpen
	// phaseEnding has proposed the stop and flushed the server, and ignores the
	// signal until the turn's transcript is out, then hands over to
	// phaseArmed.
	phaseEnding
)

type sttStream struct {
	conn *wsutil.Conn
	ctx  context.Context
	// rs converts the pipeline's audio to the rate it is sent at; it passes
	// the audio through when the two match.
	rs         *resample.Resampler
	chunkBytes int
	// lang is the configured language hint echoed on results, or "".
	lang string
	// owner is the connector that opened the session, which holds what outlives
	// it. It is nil for a stream built outside Connect.
	owner *sttConnector

	writeMu sync.Mutex
	buf     []byte
	// flushCounter numbers the flushes sent on this session.
	flushCounter int

	accumulated []string

	// turnDetection is whether the end-pointing signal drives the turns.
	turnDetection bool
	// phase is where turn detection is, and cooldown how many steps remain in
	// the post-flush cooldown, which outlives phaseEnding. Both are touched
	// only by the goroutine reading results.
	phase    turnPhase
	cooldown int

	// reads carries what the reader goroutine has read. Waiting for trailing
	// text on a timer needs a read that can be given up on, and giving up on a
	// read by canceling its context closes the connection under it, so the read
	// runs on its own and the wait happens here instead.
	reads chan sttRead
	// pending holds a message taken off reads while waiting for trailing text
	// that was not text itself, so it is acted on rather than dropped.
	pending *sttRead
	// done stops the reader when the stream is closed.
	done      chan struct{}
	closeOnce sync.Once
}

// sttRead is one message from the reader goroutine, or the failure that ended
// it.
type sttRead struct {
	msg sttMessage
	err error
}

// readLoop reads messages until the connection fails or the stream is closed. It
// reads under the stream's own context, which outlives any single wait.
func (s *sttStream) readLoop() {
	defer close(s.reads)
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			select {
			case s.reads <- sttRead{err: err}:
			case <-s.done:
			}
			return
		}
		var m sttMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		select {
		case s.reads <- sttRead{msg: m}:
		case <-s.done:
			return
		}
	}
}

// next returns the next message, preferring one already taken off the channel
// while waiting for trailing text.
func (s *sttStream) next() (sttRead, bool) {
	if p := s.pending; p != nil {
		s.pending = nil
		return *p, true
	}
	r, ok := <-s.reads
	return r, ok
}

// aggregateTrailing keeps the transcript open for a moment after a flush,
// folding in the words still arriving. A step message read in that window is
// handled where it stands, as it would be outside it: without turn detection it
// is ignored, and while a turn is ending it counts only against the cooldown.
// Anything else is held for the next call rather than dropped.
func (s *sttStream) aggregateTrailing() {
	timer := time.NewTimer(transcriptAggregationDelay)
	defer timer.Stop()
	for {
		select {
		case r, ok := <-s.reads:
			if !ok {
				return
			}
			switch {
			case r.err == nil && r.msg.Type == msgText:
				s.accumulated = append(s.accumulated, r.msg.Text)
			case r.err == nil && r.msg.Type == msgStep && !s.turnDetection:
			case r.err == nil && r.msg.Type == msgStep && s.phase == phaseEnding:
				s.handleStep(r.msg)
			default:
				s.pending = &r
				return
			}
		case <-timer.C:
			return
		}
	}
}

// sttMessage is the subset of a Gradium ASR message we use.
type sttMessage struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	Message string `json:"message"`
	// VAD is the end-pointing signal a step message carries, one entry per
	// horizon.
	VAD []vadEntry `json:"vad,omitempty"`
}

// vadEntry is the probability that speech stays inactive over the next
// HorizonS seconds.
type vadEntry struct {
	HorizonS       *float64 `json:"horizon_s,omitempty"`
	InactivityProb *float64 `json:"inactivity_prob,omitempty"`
}

// Send resamples the audio to the rate it is sent at, then buffers it and
// streams it in fixed 80 ms chunks, base64-encoded.
func (s *sttStream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	if s.rs != nil {
		audio = s.rs.Process(audio)
	}
	s.buf = append(s.buf, audio...)
	for len(s.buf) >= s.chunkBytes {
		chunk := s.buf[:s.chunkBytes]
		s.buf = s.buf[s.chunkBytes:]
		msg := map[string]any{
			msgType:  msgAudio,
			msgAudio: base64.StdEncoding.EncodeToString(chunk),
		}
		b, _ := json.Marshal(msg) //nolint:errchkjson // map of known-serializable values
		if err := s.conn.Write(s.ctx, websocket.MessageText, b); err != nil {
			return err
		}
	}
	return nil
}

// Finalize flushes the server when the pipeline's VAD reports the user stopped,
// implementing stt.Finalizer. The server answers with the text it was holding
// and a "flushed" acknowledgement, which finalizes the transcript. With turn
// detection on the end-pointing signal decides when a turn ends, so the VAD
// flushes nothing.
func (s *sttStream) Finalize() error {
	if s.turnDetection {
		return nil
	}
	return s.sendFlush()
}

// sendFlush asks the server to process the audio it is holding right away.
func (s *sttStream) sendFlush() error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.conn == nil {
		return errSTTNotConnected
	}
	s.flushCounter++
	msg := map[string]any{msgType: msgFlush, "flush_id": strconv.Itoa(s.flushCounter)}
	b, _ := json.Marshal(msg) //nolint:errchkjson // map of known-serializable values
	if err := s.conn.Write(s.ctx, websocket.MessageText, b); err != nil {
		slog.Warn("gradium stt: failed to send flush", "err", err)
		return err
	}
	return nil
}

// Recv reads the next message. Each "text" fragment accumulates and surfaces the
// running transcript as an interim result; a "flushed" acknowledgement
// finalizes the accumulated transcript with EndOfTurn set. With turn detection
// on, "step" messages open and end the user's turns.
//
// When the connection drops, the words already held are returned as a
// transcript before the failure is. The server's decoder state goes with the
// connection, so the session that replaces this one starts from a clean state,
// and the words would otherwise be lost. The transcript does not end the turn:
// the user may still be speaking, and a turn open stays open on the session
// that replaces this one.
func (s *sttStream) Recv() ([]stt.Result, error) {
	for {
		r, ok := s.next()
		if !ok {
			return nil, io.EOF
		}
		if r.err != nil {
			if res, ok := s.finalizeOnDrop(r); ok {
				return res, nil
			}
			return nil, r.err
		}
		res, err := s.handle(r.msg)
		if err != nil || len(res) > 0 {
			return res, err
		}
	}
}

// handle acts on one message, returning the results it gives.
func (s *sttStream) handle(m sttMessage) ([]stt.Result, error) {
	switch m.Type {
	case msgStep:
		if s.turnDetection {
			return s.handleStep(m), nil
		}
	case msgText:
		s.accumulated = append(s.accumulated, m.Text)
		text := strings.Join(s.accumulated, " ")
		return []stt.Result{{Text: text, Final: false, Language: s.lang}}, nil
	case msgFlushed:
		return s.flushed(), nil
	case msgEndStream:
		if len(s.accumulated) == 0 {
			return nil, nil
		}
		s.aggregateTrailing()
		return s.finalizeAccumulated(true), nil
	case msgError:
		return nil, fmt.Errorf("%w: %s", errSTTProtocol, m.Message)
	}
	return nil, nil
}

// flushed handles the server's acknowledgement of a flush. With turn detection
// on it starts the post-flush cooldown. The words that end the utterance can
// still be arriving, so the transcript stays open for a moment before it
// finalizes, and only then does a turn that was ending hand over to the next.
func (s *sttStream) flushed() []stt.Result {
	if s.turnDetection {
		s.cooldown = s.params().cooldown
	}
	s.aggregateTrailing()
	res := s.finalizeAccumulated(true)
	if s.phase == phaseEnding {
		s.phase = phaseArmed
	}
	return res
}

// finalizeAccumulated joins the text held into a transcript and clears it.
// Holding nothing, it gives nothing.
func (s *sttStream) finalizeAccumulated(endOfTurn bool) []stt.Result {
	if len(s.accumulated) == 0 {
		return nil
	}
	text := strings.Join(s.accumulated, " ")
	s.accumulated = nil
	return []stt.Result{{Text: text, Final: true, EndOfTurn: endOfTurn, Language: s.lang}}
}

// params is the turn detection settings now in force.
func (s *sttStream) params() turnParams {
	if s.owner == nil {
		return turnParams{}
	}
	return s.owner.turnParams()
}

// handleStep derives turn boundaries from the server's end-pointing signal.
//
// Each step carries, per horizon, the probability that speech stays inactive
// over the next horizon_s seconds; the entry closest to the configured horizon
// is watched. A step at or above the threshold arms the detector, and the next
// step below it opens a turn. At or above the threshold while a turn is open,
// the turn ends. A session's first turn opens on the signal falling below the
// threshold, not on it being there: the estimate reads low on the first steps,
// before the model has heard anything, then climbs as silence accumulates.
func (s *sttStream) handleStep(m sttMessage) []stt.Result {
	if len(m.VAD) == 0 {
		return nil
	}
	if s.cooldown > 0 {
		s.cooldown--
		return nil
	}
	p := s.params()
	if !p.set {
		return nil
	}
	entry := closestHorizon(m.VAD, p.horizon)
	if entry.InactivityProb == nil {
		return nil
	}
	inactive := *entry.InactivityProb >= p.threshold
	switch {
	case s.phase == phaseIdle && inactive:
		s.phase = phaseArmed
	case s.phase == phaseArmed && !inactive:
		return s.startTurn()
	case s.phase == phaseOpen && inactive:
		return s.endTurn()
	}
	// phaseEnding ignores the signal: the previous turn's transcript and stop
	// proposal must not arrive after the next turn's start proposal.
	return nil
}

// closestHorizon is the entry whose horizon is closest to horizon, the first of
// them on a tie. An entry naming no horizon is the farthest there is.
func closestHorizon(entries []vadEntry, horizon float64) vadEntry {
	best, bestDistance := entries[0], math.Inf(1)
	for i, e := range entries {
		distance := math.Inf(1)
		if e.HorizonS != nil {
			distance = math.Abs(*e.HorizonS - horizon)
		}
		if i == 0 || distance < bestDistance {
			best, bestDistance = e, distance
		}
	}
	return best
}

// startTurn opens a turn, proposing its start.
func (s *sttStream) startTurn() []stt.Result {
	slog.Debug("gradium stt turn detection: start of turn")
	s.phase = phaseOpen
	if s.owner != nil {
		s.owner.raise(s.ctx, EventTurnStart)
	}
	return []stt.Result{{Speech: stt.SpeechStarted}}
}

// endTurn ends the open turn, proposing its stop, and flushes the server. The
// flush pushes the decoder past its lookahead so the turn's tail arrives, and
// the transcript finalizes on the "flushed" acknowledgement. Without a flush no
// acknowledgement is coming, so the transcript finalizes on what has arrived.
func (s *sttStream) endTurn() []stt.Result {
	slog.Debug("gradium stt turn detection: end of turn")
	s.phase = phaseEnding
	if s.owner != nil {
		s.owner.raise(s.ctx, EventTurnEnd)
	}
	res := []stt.Result{{Speech: stt.SpeechStopped}}
	if s.sendFlush() == nil {
		return res
	}
	s.phase = phaseIdle
	return append(res, s.finalizeAccumulated(true)...)
}

// finalizeOnDrop closes out the transcript of a connection that dropped. It
// returns the words held so far as a transcript and holds the failure for the
// next call, so the session is still torn down and replaced. A turn open when
// the connection dropped stays open on the session that replaces it. A session
// being shut down on purpose has its context canceled, and its words are
// dropped with it rather than reported.
func (s *sttStream) finalizeOnDrop(r sttRead) ([]stt.Result, bool) {
	if s.ctx.Err() != nil {
		return nil, false
	}
	if s.phase == phaseOpen && s.owner != nil {
		s.owner.keepTurnOpen()
	}
	if len(s.accumulated) == 0 {
		return nil, false
	}
	res := s.finalizeAccumulated(false)
	s.pending = &r
	return res, true
}

// Close tears the session down, stops the reader and releases the resampler.
func (s *sttStream) Close() error {
	s.closeOnce.Do(func() { close(s.done) })
	if s.rs != nil {
		s.writeMu.Lock()
		s.rs.Close()
		s.writeMu.Unlock()
	}
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// Compile-time checks that the connector and its sessions answer to what the
// stream base asks of them.
var (
	_ stt.SettingsUpdater = (*sttConnector)(nil)
	_ stt.LanguageNamer   = (*sttConnector)(nil)
	_ stt.Describer       = (*sttConnector)(nil)
	_ stt.Finalizer       = (*sttStream)(nil)
)
