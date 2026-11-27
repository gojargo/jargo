package eval

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"strings"
	"time"

	"github.com/gojargo/jargo/event"
)

// The eval session: one conversation with a bot, driven to a result.
//
// A session connects to a running bot's eval transport as an RTVI client, runs
// the handshake, lets its driver converse, tears down, and returns the driver's
// result, a failed connect or a harness error included. SessionParams is how the
// run behaves, whichever kind of scenario it is, and SessionFromScenario builds
// the session of whichever kind a scenario is.

const (
	// DefaultEventTimeoutMS is the latency budget for an expectation that sets
	// no within_ms. It is generous so an expectation without one waits long
	// enough for a slow LLM or TTS response, and a function-call round trip,
	// rather than failing on latency. Set within_ms to assert on timing.
	DefaultEventTimeoutMS = 60_000
	// DefaultBotReadyTimeout is how long the bot has, once connected, to answer
	// the handshake with bot-ready. A bot builds its pipeline when the harness
	// connects, so the wait covers its startup: connecting its services and
	// loading any local models.
	DefaultBotReadyTimeout = 60 * time.Second
	// DefaultConnectTimeout is how long the harness waits for the bot to accept
	// connections at all.
	DefaultConnectTimeout = 5 * time.Second
)

// The values SessionParams.AudioSupportCheck takes.
const (
	// AudioSupportCheckOff runs every bot unchecked.
	AudioSupportCheckOff = "off"
	// AudioSupportCheckWhenAdvertised refuses a bot whose bot-ready reports its
	// transport off for audio the run uses.
	AudioSupportCheckWhenAdvertised = "when_advertised"
)

// EventProgress fires with a progress record as the conversation advances: a
// scripted scenario's turns and expectations as they resolve (a
// ScriptTurnProgress), a simulation's lines as they are spoken (a
// SimulationProgress). Records are delivered in order, and Run waits for every
// handler before it returns:
//
//	event.On(&session.Registry, eval.EventProgress, func(ctx context.Context, p eval.Progress) { … })
const EventProgress = "on_progress"

// errAudioSupportCheck is an AudioSupportCheck of no known value.
var errAudioSupportCheck = errors.New("eval: audio_support_check must be off or when_advertised")

// SessionParams is how a run behaves, whichever kind of scenario it is:
// timeouts, recording, caching, teardown.
//
// It is plain configuration, so one value serves many runs; the services a run
// uses are passed to the session separately. The zero value uses the defaults.
type SessionParams struct {
	// ConnectTimeout is how long to wait for the bot to accept the connection
	// before giving up; zero uses DefaultConnectTimeout.
	ConnectTimeout time.Duration
	// BotReadyTimeout is how long to wait, once connected, for the bot to
	// answer the RTVI handshake with bot-ready; zero uses
	// DefaultBotReadyTimeout.
	BotReadyTimeout time.Duration
	// DefaultTimeoutMS is, for scripted scenarios, the latency budget of
	// expectations without their own within_ms (a turn's expectations share one
	// deadline anchored at the send); zero uses DefaultEventTimeoutMS.
	DefaultTimeoutMS int
	// RecordPath is where to save the conversation audio, or "". Only an audio
	// run records.
	RecordPath string
	// CacheDir is the directory for cached synthesized user audio, or "" for
	// the default (see DefaultCacheDir).
	CacheDir string
	// UseCache, when false, ignores cached user audio and forces fresh
	// synthesis, with no cache reads or writes; nil is true.
	UseCache *bool
	// AudioSupportCheck is whether an audio run checks that the bot's transport
	// carries the audio it uses, as the bot reports in bot-ready. With
	// AudioSupportCheckWhenAdvertised, a bot that reports a direction the run
	// uses as off is refused before the conversation (failure kind
	// audio_unsupported); a bot that reports nothing runs unchecked. "" is
	// AudioSupportCheckOff.
	AudioSupportCheck string
}

// withDefaults is the params with every zero value replaced by its default.
func (p SessionParams) withDefaults() SessionParams {
	if p.ConnectTimeout <= 0 {
		p.ConnectTimeout = DefaultConnectTimeout
	}
	if p.BotReadyTimeout <= 0 {
		p.BotReadyTimeout = DefaultBotReadyTimeout
	}
	if p.DefaultTimeoutMS <= 0 {
		p.DefaultTimeoutMS = DefaultEventTimeoutMS
	}
	if p.AudioSupportCheck == "" {
		p.AudioSupportCheck = AudioSupportCheckOff
	}
	return p
}

// useCache reports whether cached user audio is read and written.
func (p SessionParams) useCache() bool { return p.UseCache == nil || *p.UseCache }

// validate checks the params are well-formed, and warns about the deprecated
// ones that ask for something.
func (p SessionParams) validate() error {
	switch p.AudioSupportCheck {
	case "", AudioSupportCheckOff, AudioSupportCheckWhenAdvertised:
	default:
		return fmt.Errorf("%w, not %q", errAudioSupportCheck, p.AudioSupportCheck)
	}
	return nil
}

// Session is one conversation with a bot, driven to a result R.
//
// Connect, run the handshake, let the driver converse, tear down, and turn what
// happened, a failed connect or a harness error included, into the driver's
// result. ScriptSession and SimulationSession build the client and the driver
// for their kind of scenario; SessionFromScenario builds whichever kind a
// scenario is. It raises EventProgress.
type Session[R any] struct {
	event.Registry

	kind   Kind
	name   string
	botURL string
	params SessionParams
	// trace is a timestamped trace of the harness's own decisions, for
	// diagnosing flakes.
	trace *Trace
	// Built by the kind of session: the bot's output as events, the
	// connection to the bot, and what drives the conversation.
	stream *EventStream
	client *Client
	driver driver
	// assemble has the driver build the run's result.
	assemble func(failures []AssertionFailure, durationMS int, eventsSeen []map[string]any,
		debugLog []string, skipped string) R
	// describe is the run's config summary, one line per section, for the
	// trace.
	describe func() string
	// skipReason is why the run cannot be driven at all, or "" to run it.
	skipReason func() string
}

// newSession initializes the runtime of a session of kind.
func newSession[R any](kind Kind, name, botURL string, params SessionParams) *Session[R] {
	s := &Session[R]{kind: kind, name: name, botURL: botURL, params: params.withDefaults(), trace: NewTrace()}
	s.Register(EventProgress, false)
	return s
}

// Trace is the run's trace.
func (s *Session[R]) Trace() *Trace { return s.trace }

func (*Session[R]) scenarioSession() {}

// Run connects, drives the conversation, and returns the result.
//
// What goes wrong with the run itself is a failure of the result rather than an
// error: a bot that never accepts the connection is a <connect> failure, one
// that never becomes ready a <bot-ready> failure, and a harness that fails
// mid-run an <error> or <harness> one, scored against the turn it interrupted.
// Only ctx ending is returned as an error.
func (s *Session[R]) Run(ctx context.Context) (R, error) {
	started := time.Now()
	s.trace.Start()
	s.trace.Log(fmt.Sprintf("run: %s %q -> %s", s.kind, s.name, s.botURL))
	// Record which speech, transcription and judge services and models were
	// used, so a saved trace is self-describing.
	for line := range strings.Lines(s.describe()) {
		s.trace.Log(strings.TrimRight(line, "\n"))
	}

	// The judge is closed however the run ends, including the runs that are
	// skipped or never reach the bot.
	defer s.driver.close(context.WithoutCancel(ctx))

	if skipped := s.skipReason(); skipped != "" {
		slog.WarnContext(ctx, fmt.Sprintf("eval %q: %s; skipping", s.name, skipped))
		return s.result(started, nil, skipped), nil
	}

	// A bot that never accepts is a clean <connect> failure.
	if err := s.client.WaitForBot(ctx); err != nil {
		if ctx.Err() != nil {
			return s.result(started, nil, ""), ctx.Err()
		}
		reason := fmt.Sprintf("failed to connect to %s: %v", s.botURL, err)
		return s.result(started, []AssertionFailure{s.failure(RunEventConnect, reason, FailureConnectFailed)}, ""), nil
	}

	failures, err := s.drive(ctx)
	if err != nil {
		return s.result(started, failures, ""), err
	}
	verdict := "PASS"
	if len(failures) > 0 {
		verdict = "FAIL"
	}
	s.trace.Log(fmt.Sprintf("done: %s (%d failure(s))", verdict, len(failures)))
	return s.result(started, failures, ""), nil
}

// drive starts the client, converses, and tears down.
//
// An error in the harness itself, a panic included, is recorded as a failure
// with its stack in the trace, so the run still yields a result.
func (s *Session[R]) drive(ctx context.Context) (failures []AssertionFailure, err error) {
	if startErr := s.client.Start(ctx); startErr != nil {
		s.client.Stop(ctx)
		return []AssertionFailure{s.errorFailure(startErr.Error(), nil)}, nil
	}
	defer func() {
		s.client.Stop(ctx)
		// Progress handlers run on their own goroutine; wait them out so every
		// record is delivered before the caller has the result in hand.
		s.Cleanup(context.WithoutCancel(ctx))
		if err != nil {
			return
		}
		// A harness service can fail once nothing is waiting on it, such as
		// the caller's voice on a last turn that expects nothing. Stopping the
		// client lets the failure surface; a run that otherwise passed is then
		// no evidence.
		if reason, failed := s.stream.HarnessError(); failed && len(failures) == 0 {
			s.trace.Log("FAIL: harness: " + reason)
			f := s.failure(RunEventHarness, reason, FailureError)
			s.driver.recordFailure(f)
			failures = []AssertionFailure{f}
		}
	}()
	defer func() {
		if r := recover(); r != nil {
			failures, err = []AssertionFailure{s.errorFailure(fmt.Sprintf("panic: %v", r), debug.Stack())}, nil
		}
	}()

	failures, err = s.converse(ctx)
	if err != nil {
		if ctx.Err() != nil {
			return failures, ctx.Err()
		}
		return []AssertionFailure{s.errorFailure(err.Error(), nil)}, nil
	}
	return failures, nil
}

// converse runs the handshake, then the driver; a bot that never says ready is
// a failure.
func (s *Session[R]) converse(ctx context.Context) ([]AssertionFailure, error) {
	s.trace.Log("connected")
	err := s.client.Handshake(ctx)
	var (
		unavailable *BotUnavailableError
		timeout     *handshakeTimeoutError
	)
	switch {
	case errors.As(err, &unavailable):
		s.trace.Log(fmt.Sprintf("handshake: failed (%v)", err))
		where := RunEventBotReady
		switch unavailable.Kind {
		case FailureConnectFailed:
			where = RunEventConnect
		case FailureError:
			where = RunEventHarness
		}
		return []AssertionFailure{s.failure(where, unavailable.Reason, unavailable.Kind)}, nil
	case errors.As(err, &timeout):
		s.trace.Log("handshake: failed (bot-ready not received)")
		return []AssertionFailure{s.failure(RunEventBotReady, err.Error(), FailureHandshakeTimeout)}, nil
	case err != nil:
		return nil, err
	}
	s.trace.Log("handshake: ok (bot-ready)")
	return s.driver.run(ctx)
}

// result has the driver assemble the run's result.
func (s *Session[R]) result(started time.Time, failures []AssertionFailure, skipped string) R {
	return s.assemble(failures, int(time.Since(started).Milliseconds()), s.stream.EventsSeen(), s.trace.Lines(), skipped)
}

// failure is a failure of the run itself rather than of an expectation, scored
// against the trace's current turn (-1 before any).
func (s *Session[R]) failure(eventName, reason, kind string) AssertionFailure {
	return AssertionFailure{
		TurnIndex: s.trace.Turn(), ExpectationIndex: -1, EventName: eventName, Reason: reason, Kind: kind,
	}
}

// errorFailure traces an unexpected error with its stack, when there is one,
// and lets the driver score it.
func (s *Session[R]) errorFailure(reason string, stack []byte) AssertionFailure {
	s.trace.Log("error: " + reason)
	for line := range strings.Lines(string(stack)) {
		s.trace.Log(strings.TrimRight(line, "\n"))
	}
	f := s.failure(RunEventError, reason, FailureError)
	s.driver.recordFailure(f)
	return f
}

// progress emits a progress record, a ScriptTurnProgress or a
// SimulationProgress, to the EventProgress handlers.
func (s *Session[R]) progress(record any) {
	s.Call(context.Background(), EventProgress, s, record)
}

// errNotRunnable is a scenario SessionFromScenario cannot build a session for.
var errNotRunnable = errors.New("eval: expected a scripted scenario or a simulation")

// errPersonaForScript is a persona LLM given for a scripted scenario.
var errPersonaForScript = errors.New("eval: a persona LLM applies to simulations only")

// errNotARunJudge is a judge given for a simulation that cannot judge a whole
// run.
var errNotARunJudge = errors.New("eval: a simulation's judge must judge whole runs (a RunJudge)")

// ScenarioSession is a session of either kind, as SessionFromScenario builds
// it: a *ScriptSession or a *SimulationSession. Switch on its type to run it.
type ScenarioSession interface {
	// Trace is the run's trace.
	Trace() *Trace
	scenarioSession()
}

// SessionServices are the services a session can be given rather than build
// from its scenario. A nil field is built from the scenario when the run needs
// it.
type SessionServices struct {
	// PersonaLLM plays a simulation's caller; it applies to simulations only.
	PersonaLLM PersonaLLM
	// Judge decides the scenario's eval: assertions, or a simulation's goal and
	// criteria, which takes a judge that is also a RunJudge (ClassifierJudge is
	// both).
	Judge Judge
	// UserTTS speaks the user's turns in audio mode.
	UserTTS *CachingTTSService
	// BotSTT transcribes the bot's audio when the run transcribes it.
	BotSTT SegmentedSTT
}

// SessionFromScenario builds a ready-to-run session for a scenario of either
// kind: a *ScriptSession for a *ScriptScenario, a *SimulationSession for a
// *SimulationScenario, each constructing the services it needs that services
// does not give. Then run it:
//
//	url := "ws://localhost:7860/ws"
//	session, err := eval.SessionFromScenario(scenario, url, eval.SessionParams{}, eval.SessionServices{})
//	if err != nil { … }
//	switch s := session.(type) {
//	case *eval.ScriptSession:
//		result, err := s.Run(ctx)
//		…
//	case *eval.SimulationSession:
//		result, err := s.Run(ctx)
//		…
//	}
//
// It fails when a persona LLM is given for a scripted scenario, which has no
// persona, when a simulation's judge cannot judge a whole run, when scenario is
// neither kind (a rubric has nothing to run), or when a service the scenario
// names cannot be built.
func SessionFromScenario(
	scenario LoadedScenario, botURL string, params SessionParams, services SessionServices,
) (ScenarioSession, error) {
	switch sc := scenario.(type) {
	case *SimulationScenario:
		var judge RunJudge
		if services.Judge != nil {
			runJudge, ok := services.Judge.(RunJudge)
			if !ok {
				return nil, fmt.Errorf("%w; %q got %T", errNotARunJudge, sc.Name, services.Judge)
			}
			judge = runJudge
		}
		return SimulationSessionFromScenario(sc, botURL, params, services.PersonaLLM, judge,
			services.UserTTS, services.BotSTT)
	case *ScriptScenario:
		if services.PersonaLLM != nil {
			return nil, fmt.Errorf("%w; %q is scripted", errPersonaForScript, sc.Name)
		}
		return ScriptSessionFromScenario(sc, botURL, params, services.Judge, services.UserTTS, services.BotSTT)
	}
	return nil, fmt.Errorf("%w, got %T", errNotRunnable, scenario)
}
