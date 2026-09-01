package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
	errs "github.com/gojargo/jargo/utils/errors"
)

const (
	// fluxListenURL is the streaming turn-aware transcription WebSocket.
	fluxListenURL = "wss://api.deepgram.com/v2/listen"
	// defaultFluxModel is the English conversational model.
	defaultFluxModel = "flux-general-en"
	// fluxMultilingualModel is the only model that honors language hints.
	fluxMultilingualModel = "flux-general-multi"

	// defaultWatchdogMinTimeout is the minimum idle period before silence is
	// injected to keep an active turn from dangling.
	defaultWatchdogMinTimeout = 500 * time.Millisecond
	// watchdogTick is how often the idle watchdog wakes.
	watchdogTick = 100 * time.Millisecond
	// watchdogSilenceSeconds is the duration of silence injected on a stall.
	watchdogSilenceSeconds = 0.5
	// pcmSampleWidth is the byte width of one linear16 PCM sample.
	pcmSampleWidth = 2

	// fluxConnectionTimeout bounds the wait for Flux to confirm a new
	// connection. An endpoint that rejects a connection setting sends neither a
	// Connected message nor an error, so the wait needs a bound or the session
	// opens forever with audio buffering behind it.
	fluxConnectionTimeout = 10 * time.Second
	// fluxConfigureAckTimeout is how long an in-flight Configure is trusted
	// before a new update supersedes it outright. Flux caps the number of
	// un-acked Configure messages, so at most one is ever in flight; this bounds
	// how long a missing ack can block later updates from ever being sent.
	fluxConfigureAckTimeout = 5 * time.Second
)

// Flux STT WebSocket message types (the top-level "type" field). The shared
// "Error" type (fluxMsgError) lives in deepgram.go.
const (
	fluxMsgConnected        = "Connected"
	fluxMsgTurnInfo         = "TurnInfo"
	fluxMsgConfigureSuccess = "ConfigureSuccess"
	fluxMsgConfigureFailure = "ConfigureFailure"
)

// Flux TurnInfo event types (the "event" field on a TurnInfo message).
const (
	fluxEventStartOfTurn     = "StartOfTurn"
	fluxEventTurnResumed     = "TurnResumed"
	fluxEventEndOfTurn       = "EndOfTurn"
	fluxEventEagerEndOfTurn  = "EagerEndOfTurn"
	fluxEventUpdate          = "Update"
	fluxDefaultLanguageForEn = "en"
)

// errFluxNotConfirmed is returned when Flux accepted the socket but never
// confirmed the session was ready. Flux stays silent rather than refusing the
// connection when a setting is unsupported, so an unconfirmed connection means
// the request was rejected, not that the network was slow.
//
//nolint:gochecknoglobals // sentinel error
var errFluxNotConfirmed = errors.New("deepgram flux: the connection was never confirmed")

// fluxFatalError is a fatal error Flux reported over the connection. It carries
// the code Flux named so the failure can be classified: reported in the protocol
// rather than as an HTTP status, it leaves the shared classification nothing to
// read.
type fluxFatalError struct {
	code string
	text string
}

func (e *fluxFatalError) Error() string { return e.text }

// Unwrap reports the server-error sentinel, so a caller matching on that still
// matches a fatal error carrying a code.
func (e *fluxFatalError) Unwrap() error { return errFluxServer }

// fluxPermanentErrorCodes are the Flux error codes whose cause a retry cannot
// clear. Rejected credentials are not among them: those fail the HTTP handshake
// and are classified from its status code, never reaching a Flux error message.
// A code not listed falls back to the shared classification, leaving recovery to
// the service's own reconnect handling.
//
//nolint:gochecknoglobals // a fixed table
var fluxPermanentErrorCodes = map[string]errs.Category{
	"UNPARSABLE_CLIENT_MESSAGE": errs.InvalidRequest,
}

// The Flux setting names, as an update spells them.
const (
	fluxFieldKeyterm           = "keyterm"
	fluxFieldEOTThreshold      = "eot_threshold"
	fluxFieldEagerEOTThreshold = "eager_eot_threshold"
	fluxFieldEOTTimeoutMs      = "eot_timeout_ms"
	fluxFieldLanguageHints     = "language_hints"
	fluxFieldNumerals          = "numerals"
	fluxFieldMinConfidence     = "min_confidence"
	fluxFieldModel             = "model"
)

// The Flux settings, grouped by how a change to one reaches Flux.
//
//nolint:gochecknoglobals // fixed tables
var (
	// fluxConfigureFields are the settings a live connection takes, through a
	// Configure message.
	fluxConfigureFields = []string{
		fluxFieldKeyterm, fluxFieldEOTThreshold, fluxFieldEagerEOTThreshold,
		fluxFieldEOTTimeoutMs, fluxFieldLanguageHints,
	}
	// fluxConnectionFields are the settings Flux only reads from the connection
	// URL, so a change to one is applied by opening another connection.
	fluxConnectionFields = []string{fluxFieldModel, fluxFieldNumerals}
	// fluxLocalFields are the settings applied to results as they arrive, so a
	// change to one needs no connection change at all.
	fluxLocalFields = []string{fluxFieldMinConfidence}
)

// FluxSettings is the Flux configuration a caller may change while the pipeline
// runs. Which ones can be applied to a session already running is Flux's own
// division: some are sent to it, some are only read from the URL the connection
// opened with, and one is applied here to the results as they arrive.
type FluxSettings struct {
	// STT carries the model, which Flux reads only from the connection URL, so
	// a change to it is applied by opening another connection. It also carries a
	// language, which Flux has no parameter for: language_hints covers
	// multilingual input instead, so an update naming a language is reported as
	// unsupported.
	settings.STT

	// Numerals writes spoken numbers as digits. Read only from the connection
	// URL, so a change is applied by opening another connection.
	Numerals settings.Opt[bool] `settings:"numerals"`
	// Keyterm boosts recognition of the given terms.
	Keyterm settings.Opt[[]string] `settings:"keyterm"`
	// EOTThreshold is the end-of-turn confidence required to finish a turn.
	EOTThreshold settings.Opt[float64] `settings:"eot_threshold"`
	// EagerEOTThreshold is the confidence at which an eager end-of-turn is
	// predicted, ahead of the turn being confirmed.
	EagerEOTThreshold settings.Opt[float64] `settings:"eager_eot_threshold"`
	// EOTTimeoutMs is the time in ms after speech to finish a turn regardless of
	// end-of-turn confidence.
	EOTTimeoutMs settings.Opt[int] `settings:"eot_timeout_ms"`
	// LanguageHints biases detection toward the given languages, as the base
	// codes Flux takes. Only the multilingual model honors them. An empty list
	// clears whatever hints are in force.
	LanguageHints settings.Opt[[]string] `settings:"language_hints"`
	// MinConfidence drops a finalized turn whose average word confidence does
	// not exceed it. Applied to results as they arrive.
	MinConfidence settings.Opt[float64] `settings:"min_confidence"`
}

// newFluxSettings is the starting state, taken from what the service was built
// with.
func newFluxSettings(cfg FluxConfig) *FluxSettings {
	s := &FluxSettings{}
	s.Model = settings.Set(cfg.Model)
	setOptBool(&s.Numerals, cfg.Numerals)
	setOptSlice(&s.Keyterm, cfg.Keyterm)
	setOptFloat(&s.EOTThreshold, cfg.EOTThreshold)
	setOptFloat(&s.EagerEOTThreshold, cfg.EagerEOTThreshold)
	setOptInt(&s.EOTTimeoutMs, cfg.EOTTimeoutMs)
	setOptFloat(&s.MinConfidence, cfg.MinConfidence)
	if hints := fluxLanguageHints(cfg.LanguageHints); len(hints) > 0 {
		s.LanguageHints = settings.Set(hints)
	}
	return s
}

// fluxLanguageHints renders languages as the base codes Flux takes, dropping any
// that has none.
func fluxLanguageHints(hints []language.Language) []string {
	out := make([]string, 0, len(hints))
	for _, l := range hints {
		if code := l.BaseCode(); code != "" {
			out = append(out, code)
		}
	}
	return out
}

// FluxConfig configures the Flux streaming turn-aware STT service. Fields left
// at their zero value fall back to the service defaults; optional tuning fields
// modeled as pointers or slices are omitted from the request when unset.
type FluxConfig struct {
	// ShouldInterrupt barges in when Flux detects that the user is speaking;
	// nil enables it. It is passed along to the user turn strategies this
	// service recommends, which own the interruption; strategies the application
	// configures itself override the recommendation and this setting with it.
	ShouldInterrupt *bool

	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// ListenURL overrides the transcription WebSocket endpoint; empty uses the
	// hosted endpoint.
	ListenURL string
	// Model is the Flux model; empty uses "flux-general-en". Set
	// "flux-general-multi" to enable language hints.
	Model string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int

	// EagerEOTThreshold enables eager end-of-turn predictions (interim
	// transcripts before a turn is confirmed). Off when nil.
	EagerEOTThreshold *float64
	// EOTThreshold is the end-of-turn confidence required to finish a turn;
	// nil uses the service default.
	EOTThreshold *float64
	// EOTTimeoutMs is the time in ms after speech to finish a turn regardless of
	// end-of-turn confidence; nil uses the service default.
	EOTTimeoutMs *int
	// MinConfidence drops a finalized turn whose average word confidence does
	// not exceed it; nil accepts every finalized turn.
	MinConfidence *float64
	// MipOptOut opts out of Deepgram's model-improvement program.
	MipOptOut *bool
	// Numerals writes spoken numbers as digits ("twenty three" becomes "23");
	// nil leaves the service default. Flux reads it only from the connection
	// URL, so changing it while the pipeline runs opens another connection.
	Numerals *bool
	// Keyterm boosts recognition of the given terms.
	Keyterm []string
	// Tag attaches billing tags to the request.
	Tag []string
	// LanguageHints biases detection toward the given languages. Only honored by
	// the multilingual model; ignored for other models.
	LanguageHints []language.Language
	// WatchdogMinTimeout is the minimum idle period before silence is injected
	// to keep an active turn from dangling; 0 uses 500ms.
	WatchdogMinTimeout time.Duration
	// ExtraQuery sets arbitrary additional query parameters; values override any
	// param of the same name set from other fields.
	ExtraQuery map[string]string
}

// Validate reports whether the configuration is usable.
func (cfg FluxConfig) Validate() error { return validate.Struct(cfg) }

// NewFluxSTT builds a Deepgram Flux streaming turn-aware STT service. Flux
// detects turn boundaries server-side, so the service recommends external user
// turns downstream.
func NewFluxSTT(cfg FluxConfig) *stt.StreamService {
	if cfg.Model == "" {
		cfg.Model = defaultFluxModel
	}
	if cfg.ListenURL == "" {
		cfg.ListenURL = fluxListenURL
	}
	if cfg.WatchdogMinTimeout == 0 {
		cfg.WatchdogMinTimeout = defaultWatchdogMinTimeout
	}
	c := newFluxConnector(cfg)
	svc := stt.NewStream("DeepgramFluxSTT", c, cfg.SampleRate)
	// The connector is handed the service it drives, so it can report a rejected
	// Configure on the pipeline. The rejection arrives on the read loop, long
	// after the update that caused it returned, so there is no call left to fail
	// and the service is the only way back.
	c.svc = svc
	return svc
}

// fluxQuery builds the transcription query string for the given sample rate,
// from the settings as they stand rather than as the service was built: a
// connection opened after a change to a connection-only setting is the one place
// that change takes effect.
func fluxQuery(cfg FluxConfig, live *FluxSettings, sampleRate int) url.Values {
	model := live.Model.Or(cfg.Model)

	q := url.Values{}
	q.Set("model", model)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("encoding", fluxEncoding)

	if v, ok := live.EagerEOTThreshold.Value(); ok {
		q.Set("eager_eot_threshold", strconv.FormatFloat(v, 'f', -1, 64))
	}
	if v, ok := live.EOTThreshold.Value(); ok {
		q.Set("eot_threshold", strconv.FormatFloat(v, 'f', -1, 64))
	}
	if v, ok := live.EOTTimeoutMs.Value(); ok {
		q.Set("eot_timeout_ms", strconv.Itoa(v))
	}
	if v, ok := live.Numerals.Value(); ok {
		q.Set("numerals", strconv.FormatBool(v))
	}
	query.SetBoolOpt(q, "mip_opt_out", cfg.MipOptOut)

	query.AddAll(q, "keyterm", live.Keyterm.Or(nil))
	query.AddAll(q, "tag", cfg.Tag)

	// Language hints are only meaningful on the multilingual model.
	if model == fluxMultilingualModel {
		for _, code := range live.LanguageHints.Or(nil) {
			q.Add("language_hint", code)
		}
	}

	for k, v := range cfg.ExtraQuery {
		q.Set(k, v)
	}
	return q
}

// newFluxConnector builds the connector driving a Flux session, with the
// settings store it starts from.
func newFluxConnector(cfg FluxConfig) *fluxConnector {
	return &fluxConnector{
		cfg:               cfg,
		live:              newFluxSettings(cfg),
		connectionTimeout: fluxConnectionTimeout,
	}
}

type fluxConnector struct {
	cfg FluxConfig
	// live is what may change while the pipeline runs. The service serializes
	// reading it here against applying an update, so a session is never opened
	// from a half-written change.
	live *FluxSettings
	// svc is the service this connector drives, for reporting a problem the
	// session survives.
	svc *stt.StreamService
	// connectionTimeout bounds the wait for Flux to confirm a session. Held here
	// rather than read from the constant so a test can shorten it.
	connectionTimeout time.Duration

	// mu guards the session the Configure messages go out on.
	mu     sync.Mutex
	stream *fluxStream
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *fluxConnector) Settings() any { return c.live }

// UpdateSettings applies a settings change the way Flux takes it. The fields a
// live connection accepts are sent to it in a Configure message; the ones Flux
// only reads from the connection URL ask for the session to be reopened, which
// waits until the user stops speaking; and the rest are already in force, since
// they are applied here to the results as they arrive.
func (c *fluxConnector) UpdateSettings(_ context.Context, changed settings.Changed) (bool, error) {
	stream := c.currentStream()

	if fields := fluxChangedIn(changed, fluxConfigureFields); len(fields) > 0 && stream != nil {
		stream.sendConfigure(fields, c.live)
	}
	if changed.Has(fluxFieldMinConfidence) && stream != nil {
		var floor *float64
		if v, ok := c.live.MinConfidence.Value(); ok {
			floor = &v
		}
		stream.setMinConfidence(floor)
	}

	handled := slices.Concat(fluxConfigureFields, fluxConnectionFields, fluxLocalFields)
	if rest := changed.Except(handled...); len(rest) > 0 {
		slog.Warn("runtime settings update is not supported by this service",
			"service", "DeepgramFluxSTT", "fields", strings.Join(rest, ", "))
	}

	return len(fluxChangedIn(changed, fluxConnectionFields)) > 0, nil
}

// fluxChangedIn reports which of fields the update changed.
func fluxChangedIn(changed settings.Changed, fields []string) []string {
	var out []string
	for _, f := range fields {
		if changed.Has(f) {
			out = append(out, f)
		}
	}
	return out
}

// currentStream is the session Configure messages go out on, or nil when there
// is none: a change made between sessions reaches Flux through the URL the next
// one opens with.
func (c *fluxConnector) currentStream() *fluxStream {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stream
}

// ClassifyError says what the failures Flux signals in its own protocol mean.
// Flux reports these over the connection rather than as an HTTP status, so they
// carry nothing the shared classification can read.
func (c *fluxConnector) ClassifyError(err error) errs.Category {
	if errors.Is(err, errFluxNotConfirmed) {
		return errs.InvalidRequest
	}
	var fatal *fluxFatalError
	if errors.As(err, &fatal) {
		return fluxPermanentErrorCodes[fatal.code]
	}
	return errs.Unset
}

// Metadata recommends external user turns: Flux emits its own turn boundaries.
func (c *fluxConnector) Metadata() stt.Metadata {
	noTTFS := false
	return stt.Metadata{
		UserTurnStrategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{
			EnableInterruptions: c.cfg.ShouldInterrupt,
		}),
		SupportsTTFS: &noTTFS,
		Model:        c.live.Model.Or(c.cfg.Model),
	}
}

// Connect dials the Flux transcription WebSocket for the given sample rate and
// waits for Flux to confirm the session is ready.
func (c *fluxConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := fluxQuery(c.cfg, c.live, sampleRate)

	header := http.Header{}
	header.Set("Authorization", "Token "+c.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, c.cfg.ListenURL+"?"+q.Encode(), header, 0)
	if err != nil {
		return nil, fmt.Errorf("deepgram flux: dial: %w", err)
	}
	s := &fluxStream{
		conn:        conn,
		ctx:         ctx,
		sampleRate:  sampleRate,
		model:       c.live.Model.Or(c.cfg.Model),
		live:        c.live,
		report:      c.reportProblem,
		watchdogMin: c.cfg.WatchdogMinTimeout,
		lastAudio:   time.Now(),
		connectWait: c.connectionTimeout,
	}
	if v, ok := c.live.MinConfidence.Value(); ok {
		s.minConf = &v
	}

	if err := s.awaitConnected(); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "not confirmed")
		return nil, err
	}

	s.wg.Go(s.watchdog)

	c.mu.Lock()
	c.stream = s
	c.mu.Unlock()
	return s, nil
}

// reportProblem puts a problem the session survived on the pipeline, so an
// application hears about it rather than only the log.
func (c *fluxConnector) reportProblem(ctx context.Context, msg string) {
	if c.svc == nil {
		return
	}
	c.svc.PushError(ctx, msg, nil, false)
}

type fluxStream struct {
	conn        *wsutil.Conn
	ctx         context.Context //nolint:containedctx // mirrors the STT stream lifetime
	sampleRate  int
	model       string
	watchdogMin time.Duration
	// live is the settings store, read only while the service holds it steady:
	// on the connection that opened this session, and when a settings update
	// hands its changed fields over. Never from the read loop, which runs on a
	// goroutine of its own.
	live *FluxSettings
	// report puts a problem the session survived on the pipeline.
	report func(ctx context.Context, msg string)
	// connectWait bounds the wait for Flux to confirm this session.
	connectWait time.Duration

	writeMu sync.Mutex
	wg      sync.WaitGroup

	stateMu   sync.Mutex
	speaking  bool
	lastAudio time.Time
	lastChunk time.Duration
	// minConf is the confidence floor as it currently stands. It is the one
	// setting applied to results rather than sent anywhere, so it is copied here
	// where the read loop can have it without racing the store.
	minConf *float64

	// pending holds anything read while waiting for Flux to confirm the
	// session, so a result that arrived first is still delivered.
	pending []stt.Result

	// cfgMu guards the Configure serialization below. A settings update arrives
	// on the frame goroutine and an ack on the read loop, so the two meet here.
	cfgMu sync.Mutex
	// configureInFlight records that a Configure is awaiting its ack. Flux caps
	// the number of un-acked Configure messages, so only one is ever sent at a
	// time and the rest are coalesced into configurePending.
	configureInFlight bool
	configureSentAt   time.Time
	// configurePending is the coalesced update waiting for the in-flight
	// Configure to be acked, as the field values that will go out. Only the
	// latest value of each field matters, so a later change to a field already
	// waiting simply replaces it.
	configurePending map[string]any
}

// awaitConnected reads until Flux confirms the session is ready. Anything read
// meanwhile is kept for the first Recv, so a message that arrived ahead of the
// confirmation is not lost.
//
// The wait is bounded because Flux answers a connection setting it will not
// accept with silence rather than a refusal: without a bound the session opens
// forever and audio piles up behind it.
func (s *fluxStream) awaitConnected() error {
	ctx, cancel := context.WithTimeout(s.ctx, s.connectWait)
	defer cancel()

	for {
		_, data, err := s.conn.Read(ctx)
		if err != nil {
			if ctx.Err() != nil && s.ctx.Err() == nil {
				return fmt.Errorf("%w within %s; the endpoint may not accept the current connection settings",
					errFluxNotConfirmed, s.connectWait)
			}
			return fmt.Errorf("deepgram flux: waiting for the connection: %w", err)
		}
		var m fluxMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		switch m.Type {
		case fluxMsgConnected:
			return nil
		case fluxMsgError:
			return &fluxFatalError{code: m.errCode(), text: "deepgram flux: " + m.errText()}
		case fluxMsgTurnInfo:
			s.trackTurn(m.Event)
			s.pending = append(s.pending, fluxResults(m, s.model, s.minConfidence())...)
		default:
		}
	}
}

// minConfidence is the confidence floor in force right now.
func (s *fluxStream) minConfidence() *float64 {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	return s.minConf
}

// setMinConfidence records a new confidence floor, which applies to the results
// that arrive from here on.
func (s *fluxStream) setMinConfidence(v *float64) {
	s.stateMu.Lock()
	s.minConf = v
	s.stateMu.Unlock()
}

// sendConfigure asks Flux to apply the given fields to the running session.
//
// At most one Configure is ever in flight, since Flux caps the number of
// un-acked ones. A send arriving while one is in flight is coalesced into the
// pending update and goes out once the in-flight one is acked, rather than being
// sent now; only the latest value of each field matters, so nothing is replayed.
// An in-flight Configure older than fluxConfigureAckTimeout is treated as lost,
// so a missing ack cannot block later updates for good.
//
// The values are read here, while the service holds the store steady, rather
// than when the coalesced update finally goes out: the read loop that flushes it
// cannot read the store safely, and a field queued twice keeps only its latest
// value either way.
func (s *fluxStream) sendConfigure(fields []string, live *FluxSettings) {
	values := make(map[string]any, len(fields))
	for _, f := range fields {
		switch f {
		case fluxFieldKeyterm:
			values[f] = live.Keyterm.Or(nil)
		case fluxFieldEOTThreshold:
			values[f] = live.EOTThreshold.Or(0)
		case fluxFieldEagerEOTThreshold:
			values[f] = live.EagerEOTThreshold.Or(0)
		case fluxFieldEOTTimeoutMs:
			values[f] = live.EOTTimeoutMs.Or(0)
		case fluxFieldLanguageHints:
			values[f] = live.LanguageHints.Or([]string{})
		}
	}

	s.cfgMu.Lock()
	if s.configureInFlight {
		if time.Since(s.configureSentAt) < fluxConfigureAckTimeout {
			if s.configurePending == nil {
				s.configurePending = map[string]any{}
			}
			maps.Copy(s.configurePending, values)
			s.cfgMu.Unlock()
			return
		}
		slog.Warn("no ack for the last configure message; sending the next one anyway",
			"service", "DeepgramFluxSTT", "timeout", fluxConfigureAckTimeout)
	}
	s.configureInFlight = true
	s.configureSentAt = time.Now()
	s.cfgMu.Unlock()

	s.writeConfigure(values)
}

// onConfigureAcked marks the in-flight Configure as acked and sends whatever was
// coalesced behind it. It is safe to call with nothing in flight, which is what a
// stray ack looks like.
func (s *fluxStream) onConfigureAcked() {
	s.cfgMu.Lock()
	s.configureInFlight = false
	s.configureSentAt = time.Time{}
	values := s.configurePending
	s.configurePending = nil
	if len(values) == 0 {
		s.cfgMu.Unlock()
		return
	}
	s.configureInFlight = true
	s.configureSentAt = time.Now()
	s.cfgMu.Unlock()

	s.writeConfigure(values)
}

// writeConfigure renders and sends a Configure message for the given values.
func (s *fluxStream) writeConfigure(values map[string]any) {
	message := map[string]any{"type": "Configure"}

	if v, ok := values[fluxFieldKeyterm]; ok {
		message["keyterms"] = v
	}

	thresholds := map[string]any{}
	for _, f := range []string{fluxFieldEOTThreshold, fluxFieldEagerEOTThreshold, fluxFieldEOTTimeoutMs} {
		if v, ok := values[f]; ok {
			thresholds[f] = v
		}
	}
	if len(thresholds) > 0 {
		message["thresholds"] = thresholds
	}

	if v, ok := values[fluxFieldLanguageHints]; ok {
		if s.model != fluxMultilingualModel {
			slog.Warn("language hints are only honored by the multilingual model, so the update is skipped",
				"service", "DeepgramFluxSTT", "model", s.model, "supported", fluxMultilingualModel)
		} else {
			message["language_hints"] = v
		}
	}

	payload, err := json.Marshal(message)
	if err != nil {
		slog.Error("encoding a deepgram flux configure message failed", "err", err)
		return
	}
	s.writeMu.Lock()
	err = s.conn.Write(s.ctx, websocket.MessageText, payload)
	s.writeMu.Unlock()
	if err != nil {
		slog.Warn("sending a deepgram flux configure message failed", "err", err)
	}
}

// fluxWord is one recognized word; confidence is absent on some events.
type fluxWord struct {
	Confidence *float64 `json:"confidence"`
}

// fluxMessage is the subset of a Flux message we consume.
type fluxMessage struct {
	Type       string `json:"type"`
	Event      string `json:"event"`
	Transcript string `json:"transcript"`
	// Code names the failure on a fatal Error message.
	Code any `json:"code"`
	// ErrorCode names it on a ConfigureFailure, which spells the same thing
	// differently.
	ErrorCode   any        `json:"error_code"` //nolint:tagliatelle // Deepgram wire field
	Description string     `json:"description"`
	Languages   []string   `json:"languages"`
	Words       []fluxWord `json:"words"`
}

// errCode is the code naming the failure, from whichever field carries it.
func (m fluxMessage) errCode() string {
	code := m.Code
	if code == nil {
		code = m.ErrorCode
	}
	if code == nil {
		return "unknown"
	}
	return fmt.Sprintf("%v", code)
}

// errText renders a fatal Error message. Flux names the failure with a code and
// a description, and reports neither under an "error" key, so both are read and
// a missing one is named rather than left blank.
func (m fluxMessage) errText() string {
	description := "no description"
	if m.Description != "" {
		description = m.Description
	}
	return fmt.Sprintf("[%s] %s", m.errCode(), description)
}

// Send writes a chunk of PCM audio as a binary frame and records the send time
// so the watchdog can detect a stalled turn.
func (s *fluxStream) Send(audio []byte) error {
	s.stateMu.Lock()
	s.lastAudio = time.Now()
	if s.sampleRate > 0 {
		secs := float64(len(audio)) / float64(s.sampleRate*pcmSampleWidth)
		s.lastChunk = time.Duration(secs * float64(time.Second))
	}
	s.stateMu.Unlock()

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.conn.Write(s.ctx, websocket.MessageBinary, audio); err != nil {
		return fmt.Errorf("deepgram flux: send: %w", err)
	}
	return nil
}

// Recv reads the next transcription result. TurnInfo Update/EagerEndOfTurn/
// StartOfTurn events become interim results; EndOfTurn becomes a finalized,
// end-of-turn result.
func (s *fluxStream) Recv() ([]stt.Result, error) {
	// Anything that arrived while the session was still being confirmed is
	// delivered first, in the order it came.
	if len(s.pending) > 0 {
		res := s.pending
		s.pending = nil
		return res, nil
	}
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, fmt.Errorf("deepgram flux: recv: %w", err)
		}
		var m fluxMessage
		if err := json.Unmarshal(data, &m); err != nil {
			// Skipped rather than fatal, unlike the listen session: Flux reads
			// its own stream and keeps the turn state, so one message that does
			// not parse is not worth the turn in progress.
			slog.Error("decoding a deepgram flux message failed", "err", err)
			continue
		}
		switch m.Type {
		case fluxMsgError:
			return nil, &fluxFatalError{code: m.errCode(), text: "deepgram flux: " + m.errText()}
		case fluxMsgTurnInfo:
			s.trackTurn(m.Event)
			if res := fluxResults(m, s.model, s.minConfidence()); len(res) > 0 {
				return res, nil
			}
		case fluxMsgConfigureSuccess:
			slog.Info("deepgram flux accepted the configure message")
			s.onConfigureAcked()
		case fluxMsgConfigureFailure:
			// Reported as well as logged: the update was made by the
			// application and it is the application that has to know the
			// session is not running with what it asked for. The session itself
			// is unharmed, so it carries on.
			msg := "configure rejected: " + m.errText()
			slog.Warn("deepgram flux rejected the configure message", "err", msg)
			s.onConfigureAcked()
			if s.report != nil {
				s.report(s.ctx, msg)
			}
		default:
			// Connected and other control messages carry no text.
		}
	}
}

// trackTurn records whether the user is currently speaking so the watchdog only
// injects silence during an active turn.
func (s *fluxStream) trackTurn(event string) {
	switch event {
	case fluxEventStartOfTurn:
		s.stateMu.Lock()
		s.speaking = true
		s.stateMu.Unlock()
	case fluxEventEndOfTurn:
		s.stateMu.Lock()
		s.speaking = false
		s.stateMu.Unlock()
	}
}

// fluxResults maps a Flux message to zero or one transcription result.
//
// Flux runs its own turn detection, so the boundaries it reports are carried on
// the results as proposals: the pipeline's strategies decide what to make of
// them. StartOfTurn proposes the start and carries no text, since the few words
// it may come with are a preview of a turn whose transcripts follow. EndOfTurn
// proposes the stop, and carries the turn's final transcript unless the words
// came back below the confidence floor, in which case the boundary is still
// reported and only the text is dropped.
func fluxResults(m fluxMessage, model string, minConf *float64) []stt.Result {
	if m.Type != fluxMsgTurnInfo {
		return nil
	}
	lang := primaryLanguage(m, model)
	switch m.Event {
	case fluxEventStartOfTurn:
		return []stt.Result{{Speech: stt.SpeechStarted}}
	case fluxEventUpdate, fluxEventEagerEndOfTurn:
		if m.Transcript == "" {
			return nil
		}
		return []stt.Result{{Text: m.Transcript, Final: false, Language: lang}}
	case fluxEventEndOfTurn:
		if !confidenceOK(m.Words, minConf) {
			slog.Warn("transcription confidence is below the configured floor, dropping the text",
				"service", "DeepgramFluxSTT")
			return []stt.Result{{Speech: stt.SpeechStopped}}
		}
		return []stt.Result{{
			Text: m.Transcript, Final: true, EndOfTurn: true,
			Language: lang, Speech: stt.SpeechStopped,
		}}
	case fluxEventTurnResumed:
		return nil
	default:
		return nil
	}
}

// primaryLanguage reports the detected language of a turn, falling back to
// English on the English-only model where the field is absent.
func primaryLanguage(m fluxMessage, model string) string {
	if len(m.Languages) > 0 {
		return m.Languages[0]
	}
	if model == defaultFluxModel {
		return fluxDefaultLanguageForEn
	}
	return ""
}

// confidenceOK reports whether a finalized turn clears the minimum-confidence
// threshold. With no threshold every turn passes; with a threshold set a turn
// missing confidence data is dropped.
func confidenceOK(words []fluxWord, minConf *float64) bool {
	if minConf == nil || *minConf <= 0 {
		return true
	}
	avg, ok := averageConfidence(words)
	if !ok {
		return false
	}
	return avg > *minConf
}

// averageConfidence averages the confidences of the words that carry one.
func averageConfidence(words []fluxWord) (float64, bool) {
	sum := 0.0
	n := 0
	for _, w := range words {
		if w.Confidence != nil {
			sum += *w.Confidence
			n++
		}
	}
	if n == 0 {
		return 0, false
	}
	return sum / float64(n), true
}

// watchdog injects a short block of silence when audio stops flowing mid-turn,
// so an active turn does not dangle without an end-of-turn event.
func (s *fluxStream) watchdog() {
	ticker := time.NewTicker(watchdogTick)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.stateMu.Lock()
			speaking := s.speaking
			last := s.lastAudio
			threshold := max(s.watchdogMin, s.lastChunk*2)
			s.stateMu.Unlock()

			if speaking && !last.IsZero() && time.Since(last) > threshold {
				if err := s.sendSilence(watchdogSilenceSeconds); err != nil {
					return
				}
			}
		}
	}
}

// sendSilence writes durationSecs of PCM silence at the session's sample rate.
func (s *fluxStream) sendSilence(durationSecs float64) error {
	samples := int(float64(s.sampleRate) * durationSecs)
	if samples <= 0 {
		return nil
	}
	silence := make([]byte, samples*pcmSampleWidth)
	if err := s.Send(silence); err != nil {
		return err
	}
	return nil
}

// Close asks Flux to finish the stream and then closes the socket.
func (s *fluxStream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"CloseStream"}`))
	s.writeMu.Unlock()
	s.wg.Wait()
	if err := s.conn.Close(websocket.StatusNormalClosure, ""); err != nil {
		return fmt.Errorf("deepgram flux: close: %w", err)
	}
	return nil
}
