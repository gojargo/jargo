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
	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
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
	fluxMsgWarning          = "Warning"
)

// fluxForceEndTurn asks Flux to finalize the audio sent so far, ending the turn
// in progress.
const fluxForceEndTurn = `{"type":"ForceEndTurn"}`

// fluxTriggerManual is the trigger of an EndOfTurn that answers a ForceEndTurn.
const fluxTriggerManual = "manual"

// fluxWarningNoActiveTurn is the warning Flux sends for a ForceEndTurn that
// arrived after it had already ended the turn.
const fluxWarningNoActiveTurn = "FORCE_END_TURN_NO_ACTIVE_TURN"

// Flux TurnInfo event types (the "event" field on a TurnInfo message).
const (
	fluxEventStartOfTurn     = "StartOfTurn"
	fluxEventTurnResumed     = "TurnResumed"
	fluxEventEndOfTurn       = "EndOfTurn"
	fluxEventEagerEndOfTurn  = "EagerEndOfTurn"
	fluxEventUpdate          = "Update"
	fluxDefaultLanguageForEn = "en"
)

// The events the Flux STT service fires as Flux reports on the turn, each after
// the frames the report produced have been pushed. All but EventFluxTurnResumed
// carry the transcript Flux sent with the report.
//
// The turn events follow Flux's turns. With FluxConfig.VADForceTurnEndpoint
// those are not the pipeline's turns: EventFluxEndOfTurn fires on every Flux
// finalization, so possibly several times per user turn. The user aggregator's
// own events report the pipeline's turns:
//
//	events.On(svc.Events(), deepgram.EventFluxUpdate,
//	    func(ctx context.Context, transcript string) { … })
//	events.OnSignal(svc.Events(), deepgram.EventFluxTurnResumed,
//	    func(ctx context.Context) { … })
const (
	// EventFluxStartOfTurn fires when Flux detects the user started a turn. The
	// transcript may hold the turn's first few words, or nothing.
	EventFluxStartOfTurn = "on_start_of_turn"
	// EventFluxTurnResumed fires when the user resumes speaking after a pause
	// Flux had predicted to be the end of the turn.
	EventFluxTurnResumed = "on_turn_resumed"
	// EventFluxEndOfTurn fires when Flux commits the end of the turn, with its
	// final transcript.
	EventFluxEndOfTurn = "on_end_of_turn"
	// EventFluxEagerEndOfTurn fires when Flux predicts the turn has ended
	// without committing to it, with the predicted transcript.
	EventFluxEagerEndOfTurn = "on_eager_end_of_turn"
	// EventFluxUpdate fires on each interim transcript of the turn in progress,
	// with the turn's transcript so far. An update with no text fires nothing.
	EventFluxUpdate = "on_update"
)

// fluxEvents are the events NewFluxSTT registers on the service.
//
//nolint:gochecknoglobals // a fixed table
var fluxEvents = []string{
	EventFluxStartOfTurn, EventFluxTurnResumed, EventFluxEndOfTurn,
	EventFluxEagerEndOfTurn, EventFluxUpdate,
}

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
var fluxPermanentErrorCodes = map[string]errclass.Category{
	"UNPARSABLE_CLIENT_MESSAGE": errclass.InvalidRequest,
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
		fluxFieldEOTTimeoutMs, fluxFieldLanguageHints, fluxFieldNumerals,
	}
	// fluxConnectionFields are the settings Flux only reads from the connection
	// URL, so a change to one is applied by opening another connection.
	fluxConnectionFields = []string{fluxFieldModel}
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

	// Numerals writes spoken numbers as digits. A change is sent to the
	// running connection.
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
	// Ignored with VADForceTurnEndpoint.
	ShouldInterrupt *bool
	// VADForceTurnEndpoint hands the decision of when a user turn ends to the
	// pipeline's own user turn strategies (a smart-turn analyzer, an LLM
	// completion gate), instead of Flux. Flux keeps transcribing: each
	// VADUserStoppedSpeakingFrame asks it to finalize the audio sent so far
	// with a ForceEndTurn, so a VAD analyzer is required, and one user turn can
	// produce several finalized transcripts, which the user aggregator joins
	// until the stop strategy ends the turn. The service then proposes neither
	// turn edge and recommends no strategies. Flux still finalizes on its own
	// confidence unless EOTThreshold is 1.0. It cannot be combined with
	// EagerEOTThreshold: nothing answers Flux's predictions when the pipeline
	// owns the turn. Off by default.
	VADForceTurnEndpoint bool `validate:"excluded_with=EagerEOTThreshold"`
	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.DeepgramFluxTTFSP99, and zero means no
	// wait. It only applies with VADForceTurnEndpoint, where the transcript
	// trails the VAD stop by a ForceEndTurn round trip; otherwise Flux marks
	// the end of speech itself.
	TTFSP99 *time.Duration

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
	// nil uses the service default. At 1.0 Flux ends a turn only on
	// EOTTimeoutMs or a ForceEndTurn.
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
	// nil leaves the service default. Changing it while the pipeline runs sends
	// the change to the running connection.
	Numerals *bool
	// Keyterm boosts recognition of the given terms.
	Keyterm []string
	// Tag attaches billing tags to the request.
	Tag []string
	// LanguageHints biases detection toward the given languages. Only honored by
	// the multilingual model; ignored for other models.
	LanguageHints []language.Language
	// WatchdogMinTimeout is the minimum idle period before silence is injected
	// to keep an active turn from dangling. Nil uses 500ms; zero leaves only the
	// floor of twice the last chunk's duration.
	WatchdogMinTimeout *time.Duration
	// ExtraQuery sets arbitrary additional query parameters; values override any
	// param of the same name set from other fields.
	ExtraQuery map[string]string
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (cfg FluxConfig) Validate() error { return validate.Struct(cfg) }

// NewFluxSTT builds a Deepgram Flux streaming turn-aware STT service. Flux
// detects turn boundaries server-side, so the service recommends external user
// turns downstream, unless VADForceTurnEndpoint hands the turns to the
// pipeline's own strategies.
func NewFluxSTT(cfg FluxConfig) *stt.StreamService {
	if cfg.Model == "" {
		cfg.Model = defaultFluxModel
	}
	if cfg.ListenURL == "" {
		cfg.ListenURL = fluxListenURL
	}
	if cfg.VADForceTurnEndpoint && cfg.ShouldInterrupt != nil && !*cfg.ShouldInterrupt {
		slog.Warn("ShouldInterrupt is ignored with VADForceTurnEndpoint; "+
			"configure interruptions on the user turn start strategy instead", "service", "DeepgramFluxSTT")
	}
	c := newFluxConnector(cfg)
	svc := stt.NewStream("DeepgramFluxSTT", c, cfg.SampleRate)
	for _, name := range fluxEvents {
		svc.Events().Register(name, false)
	}
	// The connector is handed the service it drives, so it can report a rejected
	// Configure on the pipeline and fire the turn events. The rejection arrives
	// on the read loop, long after the update that caused it returned, so there
	// is no call left to fail and the service is the only way back.
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
		cfg:                cfg,
		live:               newFluxSettings(cfg),
		connectionTimeout:  fluxConnectionTimeout,
		closeStreamTimeout: wsutil.DefaultCloseTimeout,
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
	// closeStreamTimeout bounds telling Flux a session is ending, for the same
	// reason held here.
	closeStreamTimeout time.Duration

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
func (c *fluxConnector) ClassifyError(err error) errclass.Category {
	if fatal, ok := errors.AsType[*fluxFatalError](err); ok {
		return fluxPermanentErrorCodes[fatal.code]
	}
	return errclass.Unset
}

// EagerEndOfTurn reports whether Flux was asked to predict the end of a turn
// before committing to it, which is what the eager threshold turns on.
func (c *fluxConnector) EagerEndOfTurn() bool {
	if c.live != nil {
		if _, ok := c.live.EagerEOTThreshold.Value(); ok {
			return true
		}
	}
	return c.cfg.EagerEOTThreshold != nil
}

// Metadata recommends external user turns: Flux emits its own turn boundaries.
// With the eager threshold set it recommends the eager ones instead, which
// answer a predicted end of turn while the turn is still open.
//
// With VADForceTurnEndpoint it recommends nothing, leaving the turns to the
// pipeline's own strategies, and TTFS applies: the VAD, not Flux, marks the end
// of speech, and the transcript trails it by a ForceEndTurn round trip.
func (c *fluxConnector) Metadata() stt.Metadata {
	m := stt.Metadata{Model: c.live.Model.Or(c.cfg.Model)}
	if c.cfg.VADForceTurnEndpoint {
		m.TTFSP99 = stt.TTFSP99Or(c.cfg.TTFSP99, stt.DeepgramFluxTTFSP99)
		return m
	}
	noTTFS := false
	m.SupportsTTFS = &noTTFS
	external := turns.ExternalStrategiesConfig{EnableInterruptions: c.cfg.ShouldInterrupt}
	m.UserTurnStrategies = turns.ExternalStrategies(external)
	if eager := turns.EagerUserTurnStrategies(c.EagerEndOfTurn(), turns.EagerStrategiesConfig{
		ExternalStrategiesConfig: external,
	}); eager != nil {
		m.UserTurnStrategies = *eager
	}
	return m
}

// errNotFlux is returned by ForceEndTurn for a service that is not a Flux STT
// service.
//
//nolint:gochecknoglobals // sentinel error
var errNotFlux = errors.New("deepgram flux: not a Flux STT service")

// ForceEndTurn asks Flux to finalize the audio svc has sent so far, ending the
// turn in progress, on a signal of the application's own such as a
// push-to-talk release. With VADForceTurnEndpoint each VAD stop does the same.
// Without it, the EndOfTurn Flux answers with ends the pipeline's turn as any
// other does.
//
// It does nothing while no turn is in progress or no session is open, and fails
// when svc is not a Flux STT service or the request could not be sent.
func ForceEndTurn(svc *stt.StreamService) error {
	c, ok := svc.Connector().(*fluxConnector)
	if !ok {
		return errNotFlux
	}
	s := c.currentStream()
	if s == nil {
		return nil
	}
	return s.forceEndTurn()
}

// Connect dials the Flux transcription WebSocket for the given sample rate.
//
// Flux confirms the session with a Connected message, which the read loop waits
// for (see Recv): a dial does not hold up whatever triggered it, the audio
// path included, for the length of that wait.
// WebsocketOptions hands the stream base the connection settings the
// configuration gives, implementing wsservice.Optioner.
func (c *fluxConnector) WebsocketOptions() wsservice.Options { return c.cfg.Websocket }

func (c *fluxConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := fluxQuery(c.cfg, c.live, sampleRate)

	header := http.Header{}
	header.Set("Authorization", "Token "+c.cfg.APIKey)

	conn, err := c.cfg.Websocket.Dial(ctx, c.cfg.ListenURL+"?"+q.Encode(), header, 0)
	if err != nil {
		return nil, dialError("deepgram flux", err)
	}
	watchdogMin := defaultWatchdogMinTimeout
	if c.cfg.WatchdogMinTimeout != nil {
		watchdogMin = *c.cfg.WatchdogMinTimeout
	}
	s := &fluxStream{
		conn:        conn,
		ctx:         ctx,
		forceTurns:  c.cfg.VADForceTurnEndpoint,
		sampleRate:  sampleRate,
		model:       c.live.Model.Or(c.cfg.Model),
		live:        c.live,
		report:      c.reportProblem,
		raise:       c.raise,
		watchdogMin: watchdogMin,
		lastAudio:   time.Now(),
		connectWait: c.connectionTimeout,
		closeWait:   c.closeStreamTimeout,
		stop:        make(chan struct{}),
		fail:        c.reportFailure,
		// The finalizes a ForceEndTurn asks for are recorded with the service
		// the connector drives.
		requestFinalize: c.requestFinalize,
	}
	if v, ok := c.live.MinConfidence.Value(); ok {
		s.minConf = &v
	}

	c.mu.Lock()
	c.stream = s
	c.mu.Unlock()
	return s, nil
}

// reportFailure puts a failure that ends the session on the pipeline, with the
// error behind it, so it is classified: a failure Flux names as permanent costs
// the service its usability, and the read loop does not reconnect into it.
func (c *fluxConnector) reportFailure(ctx context.Context, err error) {
	if c.svc == nil {
		return
	}
	c.svc.PushError(ctx, err.Error(), err, false)
}

// reportProblem puts a problem the session survived on the pipeline, so an
// application hears about it rather than only the log.
func (c *fluxConnector) reportProblem(ctx context.Context, msg string) {
	if c.svc == nil {
		return
	}
	c.svc.PushError(ctx, msg, nil, false)
}

// requestFinalize records with the service that Flux was asked to finalize,
// so the EndOfTurn answering it confirms it.
func (c *fluxConnector) requestFinalize() {
	if c.svc != nil {
		c.svc.RequestFinalize()
	}
}

// raise fires one of the service's turn events.
func (c *fluxConnector) raise(ctx context.Context, ev fluxTurnEvent) {
	if c.svc == nil {
		return
	}
	if ev.name == EventFluxTurnResumed {
		c.svc.Events().Call(ctx, ev.name, c.svc)
		return
	}
	c.svc.Events().Call(ctx, ev.name, c.svc, ev.transcript)
}

// fluxTurnEvent is a turn event waiting to be fired.
type fluxTurnEvent struct {
	name       string
	transcript string
}

// fluxTurnEventFor reports the turn event a TurnInfo message fires, if any.
func fluxTurnEventFor(m fluxMessage) (fluxTurnEvent, bool) {
	var name string
	switch m.Event {
	case fluxEventStartOfTurn:
		name = EventFluxStartOfTurn
	case fluxEventTurnResumed:
		name = EventFluxTurnResumed
	case fluxEventEndOfTurn:
		name = EventFluxEndOfTurn
	case fluxEventEagerEndOfTurn:
		name = EventFluxEagerEndOfTurn
	case fluxEventUpdate:
		if m.Transcript == "" {
			return fluxTurnEvent{}, false
		}
		name = EventFluxUpdate
	default:
		return fluxTurnEvent{}, false
	}
	return fluxTurnEvent{name: name, transcript: m.Transcript}, true
}

type fluxStream struct {
	conn        *wsutil.Conn
	ctx         context.Context //nolint:containedctx // mirrors the STT stream lifetime
	sampleRate  int
	model       string
	watchdogMin time.Duration
	// forceTurns is whether the pipeline's strategies decide where turns end:
	// the VAD stop asks Flux to finalize, and Flux's turn edges are not
	// proposed.
	forceTurns bool
	// requestFinalize records a finalize asked for with the service; nil for a
	// stream built outside Connect.
	requestFinalize func()
	// live is the settings store, read only while the service holds it steady:
	// on the connection that opened this session, and when a settings update
	// hands its changed fields over. Never from the read loop, which runs on a
	// goroutine of its own.
	live *FluxSettings
	// report puts a problem the session survived on the pipeline.
	report func(ctx context.Context, msg string)
	// raise fires one of the service's turn events.
	raise func(ctx context.Context, ev fluxTurnEvent)
	// turnEvents are the turn events of the messages whose results Recv last
	// returned, fired at the start of the next Recv: the read loop pushes a
	// Recv's results before it calls again, so each event fires after the
	// frames its message produced. Read loop only.
	turnEvents []fluxTurnEvent
	// fail puts a failure that ends the session on the pipeline.
	fail func(ctx context.Context, err error)
	// connectWait bounds the wait for Flux to confirm this session.
	connectWait time.Duration
	// closeWait bounds telling Flux this session is ending.
	closeWait time.Duration
	// confirmed records that Flux has confirmed the session. Read loop only.
	confirmed bool
	// lifeMu guards closed, which Close sets so a confirmation arriving as the
	// session closes does not start a watchdog Close is no longer waiting for.
	lifeMu sync.Mutex
	closed bool
	// stop is closed by Close, which ends the watchdog: Close runs while the
	// session's context is still live.
	stop chan struct{}

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

	// turnText is what Flux has transcribed of the turn in progress, from its
	// latest Update, and turnLanguage the language that Update was in. A turn
	// lives in one connection, so these are what end it when the connection
	// closes before Flux sends EndOfTurn. Read loop only.
	turnText     string
	turnLanguage string

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

// awaitConnected handles messages until Flux confirms the session is ready.
// Anything read meanwhile is kept for the first Recv, so a message that arrived
// ahead of the confirmation is not lost.
//
// Flux refuses unsupported settings with an HTTP 400 during the handshake, so a
// connection that opens but never confirms is retried like a dropped one: the
// wait ends with an error wrapping wsservice.ErrNotConfirmed, which the read
// loop counts as failing right after connecting, giving up after a few in a row.
func (s *fluxStream) awaitConnected() error {
	read := func(ctx context.Context) ([]byte, error) {
		_, data, err := s.conn.Read(ctx)
		return data, err
	}
	err := wsservice.ReceiveUntilConfirmed(s.ctx, read, s.handleUnconfirmed, s.connectWait)
	if errors.Is(err, wsservice.ErrNotConfirmed) {
		return fmt.Errorf("deepgram flux: %w", err)
	}
	return err
}

// handleUnconfirmed handles one message read while the session is waiting to be
// confirmed, and reports whether it was the confirmation.
func (s *fluxStream) handleUnconfirmed(data []byte) (bool, error) {
	var m fluxMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return false, nil //nolint:nilerr // a message that does not parse is skipped, not fatal
	}
	switch m.Type {
	case fluxMsgConnected:
		return true, nil
	case fluxMsgError:
		return false, s.fatal(m)
	case fluxMsgTurnInfo:
		s.trackTurn(m)
		s.queueTurnEvent(m)
		s.pending = append(s.pending, s.results(m)...)
	case fluxMsgWarning:
		s.warn(m)
	default:
	}
	return false, nil
}

// startWatchdog starts the watchdog once the session is confirmed: it runs for a
// confirmed session only, and not for one already closing.
func (s *fluxStream) startWatchdog() {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	if !s.closed {
		s.wg.Go(s.watchdog)
	}
}

// fatal reports the fatal error Flux sent and returns it, ending the session.
func (s *fluxStream) fatal(m fluxMessage) error {
	err := &fluxFatalError{code: m.errCode(), text: "deepgram flux: " + m.errText()}
	if s.fail != nil {
		s.fail(s.ctx, err)
	}
	return err
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
		case fluxFieldNumerals:
			// A value cleared goes out as null, leaving Flux its default.
			var numerals any
			if v, ok := live.Numerals.Value(); ok {
				numerals = v
			}
			values[f] = numerals
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
	if v, ok := values[fluxFieldNumerals]; ok {
		message["numerals"] = v
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
	// Code names the failure on a fatal Error message, and the problem on a
	// Warning.
	Code any `json:"code"`
	// Trigger says what ended the turn on an EndOfTurn: "manual" for one that
	// answers a ForceEndTurn.
	Trigger string `json:"trigger"`
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
	s.fireTurnEvents()
	if !s.confirmed {
		if err := s.awaitConnected(); err != nil {
			return nil, err
		}
		s.confirmed = true
		s.startWatchdog()
	}
	// Anything that arrived while the session was still being confirmed is
	// delivered first, in the order it came.
	if len(s.pending) > 0 {
		res := s.pending
		s.pending = nil
		return res, nil
	}
	s.fireTurnEvents()
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
			return nil, s.fatal(m)
		case fluxMsgTurnInfo:
			s.trackTurn(m)
			s.queueTurnEvent(m)
			if res := s.results(m); len(res) > 0 {
				return res, nil
			}
			// Nothing to push first, so the event fires now.
			s.fireTurnEvents()
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
		case fluxMsgWarning:
			s.warn(m)
		default:
			// Connected and other control messages carry no text.
		}
	}
}

// warn logs a Warning from Flux. Warnings never interrupt the stream, so none
// is reported as an error. A ForceEndTurn that found no turn to end raced an
// EndOfTurn Flux had already sent, which is routine.
func (s *fluxStream) warn(m fluxMessage) {
	if m.errCode() == fluxWarningNoActiveTurn {
		slog.Debug("deepgram flux: ForceEndTurn arrived with no active turn")
		return
	}
	description := "no description"
	if m.Description != "" {
		description = m.Description
	}
	slog.Warn("deepgram flux warning", "code", m.errCode(), "description", description)
}

// results maps a Flux message to its results. With the turns left to the
// pipeline's strategies, the turn edges Flux reports are not proposed: what
// carried only a proposal is dropped, and the transcripts go out alone.
func (s *fluxStream) results(m fluxMessage) []stt.Result {
	res := fluxResults(m, s.model, s.minConfidence())
	if !s.forceTurns {
		return res
	}
	return withoutTurnProposals(res)
}

// withoutTurnProposals strips the speech boundaries from res, dropping a result
// left with nothing to report.
func withoutTurnProposals(res []stt.Result) []stt.Result {
	out := res[:0]
	for _, r := range res {
		r.Speech = stt.SpeechUnknown
		if r != (stt.Result{}) {
			out = append(out, r)
		}
	}
	return out
}

// Finalize asks Flux to finalize the turn in progress when the pipeline's VAD
// reports the user stopped, implementing stt.Finalizer. Only with
// VADForceTurnEndpoint: otherwise Flux decides where a turn ends, and the VAD
// stop asks nothing of it.
func (s *fluxStream) Finalize() error {
	if !s.forceTurns {
		return nil
	}
	return s.forceEndTurn()
}

// forceEndTurn asks Flux to finalize the audio sent so far, recording the
// finalize with the service so the EndOfTurn answering it confirms it. It does
// nothing while no turn is in progress, or once the session is closing.
func (s *fluxStream) forceEndTurn() error {
	if !s.turnInProgress() {
		return nil
	}
	if s.requestFinalize != nil {
		s.requestFinalize()
	}
	return s.sendForceEndTurn()
}

// turnInProgress reports whether Flux has a turn open on a session still
// running, which is the only time a ForceEndTurn has anything to finalize.
func (s *fluxStream) turnInProgress() bool {
	s.lifeMu.Lock()
	closed := s.closed
	s.lifeMu.Unlock()
	s.stateMu.Lock()
	speaking := s.speaking
	s.stateMu.Unlock()
	return speaking && !closed && s.ctx.Err() == nil
}

// sendForceEndTurn asks Flux to finalize the audio sent so far. Flux answers
// with an EndOfTurn whose trigger is "manual".
func (s *fluxStream) sendForceEndTurn() error {
	slog.Debug("deepgram flux: sending ForceEndTurn")
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.conn.Write(s.ctx, websocket.MessageText, []byte(fluxForceEndTurn)); err != nil {
		return fmt.Errorf("deepgram flux: sending ForceEndTurn: %w", err)
	}
	return nil
}

// queueTurnEvent queues the turn event a TurnInfo message fires, to be fired
// once the results it produced have been pushed.
func (s *fluxStream) queueTurnEvent(m fluxMessage) {
	if ev, ok := fluxTurnEventFor(m); ok {
		s.turnEvents = append(s.turnEvents, ev)
	}
}

// fireTurnEvents fires the queued turn events, in the order their messages
// arrived.
func (s *fluxStream) fireTurnEvents() {
	evs := s.turnEvents
	s.turnEvents = nil
	if s.raise == nil {
		return
	}
	for _, ev := range evs {
		s.raise(s.ctx, ev)
	}
}

// trackTurn records whether the user is currently speaking, so the watchdog only
// injects silence during an active turn, and what Flux has transcribed of the
// turn so far, so a connection closing mid-turn can end it.
func (s *fluxStream) trackTurn(m fluxMessage) {
	switch m.Event {
	case fluxEventStartOfTurn:
		s.stateMu.Lock()
		s.speaking = true
		s.stateMu.Unlock()
		s.turnText, s.turnLanguage = m.Transcript, ""
	case fluxEventUpdate:
		if m.Transcript != "" {
			s.turnText, s.turnLanguage = m.Transcript, primaryLanguage(m, s.model)
		}
	case fluxEventEndOfTurn:
		s.stateMu.Lock()
		s.speaking = false
		s.stateMu.Unlock()
		s.turnText, s.turnLanguage = "", ""
	}
}

// EndSession ends a turn the connection closed in the middle of, implementing
// stt.SessionEnder. A Flux turn lives in one connection, so a new connection
// never ends the old one's turn. The transcript received so far is reported as
// the turn's transcription, unfinalized since Flux never committed it, and the
// turn's end is proposed unless the turns are left to the pipeline's
// strategies. If the user keeps talking on a new connection, what follows is
// transcribed as a turn of its own.
func (s *fluxStream) EndSession() []stt.Result {
	s.stateMu.Lock()
	speaking := s.speaking
	s.speaking = false
	s.stateMu.Unlock()
	if !speaking {
		return nil
	}
	slog.Debug("deepgram flux connection closed mid-turn, ending the turn")
	text, lang := s.turnText, s.turnLanguage
	s.turnText, s.turnLanguage = "", ""
	res := []stt.Result{{Text: text, Final: text != "", Speech: stt.SpeechStopped, Language: lang}}
	if s.forceTurns {
		return withoutTurnProposals(res)
	}
	return res
}

// fluxResults maps a Flux message to zero or one transcription result.
//
// Flux runs its own turn detection, so the boundaries it reports are carried on
// the results as proposals: the pipeline's strategies decide what to make of
// them. StartOfTurn proposes the start and carries no text, since the few words
// it may come with are a preview of a turn whose transcripts follow. EndOfTurn
// proposes the stop, and carries the turn's final transcript unless the words
// came back below the confidence floor, in which case the boundary is still
// reported and only the text is dropped. Only an EndOfTurn with the "manual"
// trigger answers a ForceEndTurn, and so confirms the finalize asked for.
func fluxResults(m fluxMessage, model string, minConf *float64) []stt.Result {
	if m.Type != fluxMsgTurnInfo {
		return nil
	}
	lang := primaryLanguage(m, model)
	switch m.Event {
	case fluxEventStartOfTurn:
		return []stt.Result{{Speech: stt.SpeechStarted}}
	case fluxEventUpdate:
		if m.Transcript == "" {
			return nil
		}
		return []stt.Result{{Text: m.Transcript, Final: false, Language: lang}}
	case fluxEventEagerEndOfTurn:
		if m.Transcript == "" {
			return nil
		}
		// Flux predicts the turn has ended without committing to it. That is not
		// an interim transcript: a reply may be generated from it, which nothing
		// would do for a partial, and it is withdrawn on TurnResumed.
		return []stt.Result{{Text: m.Transcript, EagerEndOfTurn: true, Language: lang}}
	case fluxEventEndOfTurn:
		if !confidenceOK(m.Words, minConf) {
			slog.Warn("transcription confidence is below the configured floor, dropping the text",
				"service", "DeepgramFluxSTT")
			// The turn still ended, and still answers a ForceEndTurn: the
			// confirmation stands with no text, and marks the next transcript.
			return []stt.Result{{
				Final: true, FromFinalize: m.Trigger == fluxTriggerManual, Speech: stt.SpeechStopped,
			}}
		}
		return []stt.Result{{
			Text: m.Transcript, Final: true, EndOfTurn: true,
			FromFinalize: m.Trigger == fluxTriggerManual,
			Language:     lang, Speech: stt.SpeechStopped,
		}}
	case fluxEventTurnResumed:
		// The user was not finished after all, so any prediction Flux made about
		// this turn is void.
		return []stt.Result{{EagerEndOfTurnWithdrawn: true}}
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
		case <-s.stop:
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

// Close asks Flux to finish the stream and then closes the socket. An open
// session holds one of the account's concurrent Flux connections, so the socket
// is closed whether or not the ask went out.
func (s *fluxStream) Close() error {
	s.lifeMu.Lock()
	already := s.closed
	s.closed = true
	s.lifeMu.Unlock()
	if !already {
		// The watchdog stops with the session, which may still be running.
		close(s.stop)
	}
	open := sendCloseStream(s.ctx, s.conn, &s.writeMu, s.closeWait)
	s.wg.Wait()
	if !open {
		return nil
	}
	if err := s.conn.Close(websocket.StatusNormalClosure, ""); err != nil {
		return fmt.Errorf("deepgram flux: close: %w", err)
	}
	return nil
}
