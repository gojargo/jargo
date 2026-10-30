package turns

import (
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// strategyEnv is the shared environment the controller hands each strategy: the
// single mutex that serializes ALL turn state (controller + every strategy), the
// session context, and the callbacks a strategy uses to signal decisions and
// push frames. Every callback and every Trigger* method runs with mu held — the
// controller's Process and all timer callbacks acquire mu first — so strategies
// need no locking of their own.
type strategyEnv struct {
	mu *sync.Mutex

	// Each decision callback carries the strategy that made it, so whatever the
	// controller reports the decision to can say which strategy decided. A turn
	// opened by a wake phrase and one opened by voice activity are the same
	// event otherwise, and they are not the same thing.
	started          func(s StartStrategy, params UserTurnStartedParams)
	resetAggregation func(s StartStrategy)
	// inferenceTriggered carries the speculation the inference answers, or nil
	// on an ordinary one against a turn that has ended.
	inferenceTriggered  func(s StopStrategy, speculation *UserTurnSpeculation)
	speculationCanceled func(s StopStrategy)
	stopped             func(s StopStrategy, params UserTurnStoppedParams)
	push                func(f frames.Frame, dir processor.Direction)
	broadcast           func(build func() frames.Frame)
}

// locked runs fn with the shared mutex held. A strategy the controller never
// attached has no mutex to take, and nothing to signal through either, so fn
// simply runs.
func (e strategyEnv) locked(fn func()) {
	if e.mu == nil {
		fn()
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	fn()
}

// after schedules fn to run after d with the shared mutex held, returning a
// cancel func. cancel must be called with the mutex held (from Process or
// another timer callback); calling it after the timer has fired is a no-op.
//
// A strategy the controller never attached has nothing to signal, so nothing is
// scheduled and the cancel is a no-op.
func (e strategyEnv) after(d time.Duration, fn func()) (cancel func()) {
	if e.mu == nil {
		return func() {}
	}
	stopped := false
	mu := e.mu
	timer := time.AfterFunc(d, func() {
		mu.Lock()
		defer mu.Unlock()
		if stopped {
			return
		}
		fn()
	})
	return func() {
		stopped = true
		timer.Stop()
	}
}

// boolOr returns *p, or configured when p is nil. A nil override is what "leave
// the strategy's own setting alone" is written as.
func boolOr(p *bool, configured bool) bool {
	if p == nil {
		return configured
	}
	return *p
}

// StartStrategy decides when the user's turn begins. Concrete strategies embed
// StartStrategyBase and implement Process.
type StartStrategy interface {
	// Process examines one frame; returning Stop short-circuits the start chain
	// for that frame. It runs with the shared mutex held.
	Process(f frames.Frame) ProcessFrameResult
	// TurnStarted readies per-turn state when a turn begins.
	TurnStarted()
	// TurnStopped clears per-turn state when a turn ends.
	TurnStopped()
	// Setup hands the strategy the pipeline's configuration, which it knows
	// before any frame arrives.
	Setup(s processor.Setup) error
	// Cleanup releases resources (timers).
	Cleanup()
	// ResolvesProposedTurnStartFrames reports whether this strategy resolves
	// proposals into turn starts.
	ResolvesProposedTurnStartFrames() bool
	attach(self StartStrategy, env strategyEnv)
}

// StartStrategyBase is embedded by every start strategy. It carries the
// open-turn flags and the trigger helpers.
type StartStrategyBase struct {
	// EnableInterruptions broadcasts an InterruptionFrame on turn start.
	EnableInterruptions bool
	// EnableUserSpeakingFrames broadcasts a UserStartedSpeakingFrame on turn start.
	EnableUserSpeakingFrames bool
	env                      strategyEnv
	// self is the concrete strategy this base belongs to, so a decision the base
	// signals is attributed to the strategy that made it rather than to the base.
	self StartStrategy
}

func (b *StartStrategyBase) attach(self StartStrategy, env strategyEnv) {
	b.self, b.env = self, env
}

// TurnStarted is the default no-op.
func (b *StartStrategyBase) TurnStarted() {}

// TurnStopped is the default no-op.
func (b *StartStrategyBase) TurnStopped() {}

// Setup is the default no-op.
func (b *StartStrategyBase) Setup(processor.Setup) error { return nil }

// Cleanup is the default no-op.
func (b *StartStrategyBase) Cleanup() {}

// ResolvesProposedTurnStartFrames reports whether this strategy resolves
// proposals into turn starts.
//
// A ProposedUserStartedSpeakingFrame is a request for a decision, so a strategy
// that acts on one consumes it: the frame stops traveling, and no resolver
// further along decides the same turn a second time. Override to true in a
// strategy that handles that frame.
func (b *StartStrategyBase) ResolvesProposedTurnStartFrames() bool { return false }

// StartedOverrides overrides, for one turn, the flags the strategy was built
// with. A nil field keeps the configured value.
type StartedOverrides struct {
	// EnableInterruptions overrides whether an interruption is broadcast for
	// this turn. Set it false when something else in the pipeline has already
	// broadcast one.
	EnableInterruptions *bool
	// EnableUserSpeakingFrames overrides whether a UserStartedSpeakingFrame is
	// emitted for this turn. Set it false when something else in the pipeline
	// has already emitted it.
	EnableUserSpeakingFrames *bool
}

// ReportStarted reports that the user started, with the flags the strategy was
// built with.
//
// A strategy calls it when it decides the user's turn has started. The
// controller starts the turn on the first decision, or once every start
// strategy has decided in a StrategyModeAll list.
func (b *StartStrategyBase) ReportStarted() {
	b.ReportStartedOverriding(StartedOverrides{})
}

// ReportStartedOverriding reports that the user started, overriding the
// strategy's configured flags for this turn.
func (b *StartStrategyBase) ReportStartedOverriding(o StartedOverrides) {
	if b.env.started == nil {
		return
	}
	b.env.started(b.self, UserTurnStartedParams{
		EnableInterruptions:      boolOr(o.EnableInterruptions, b.EnableInterruptions),
		EnableUserSpeakingFrames: boolOr(o.EnableUserSpeakingFrames, b.EnableUserSpeakingFrames),
	})
}

// TriggerStarted reports that the user started.
//
// Deprecated: use ReportStarted. A strategy reports its decision, and the
// controller decides when the turn starts.
func (b *StartStrategyBase) TriggerStarted() { b.ReportStarted() }

// TriggerStartedOverriding reports that the user started, overriding the
// strategy's configured flags for this turn.
//
// Deprecated: use ReportStartedOverriding.
func (b *StartStrategyBase) TriggerStartedOverriding(o StartedOverrides) {
	b.ReportStartedOverriding(o)
}

// TriggerResetAggregation asks the user aggregator to drop the in-progress
// aggregation (e.g. pre-wake-phrase speech).
func (b *StartStrategyBase) TriggerResetAggregation() {
	if b.env.resetAggregation != nil {
		b.env.resetAggregation(b.self)
	}
}

// after schedules a mutex-guarded timeout for the strategy.
func (b *StartStrategyBase) after(d time.Duration, fn func()) func() { return b.env.after(d, fn) }

// StopStrategy decides when the user's turn ends. Concrete strategies embed
// StopStrategyBase and implement Process.
type StopStrategy interface {
	// Process examines one frame; returning Stop short-circuits the stop chain.
	// Stop strategies usually return Continue and signal via Trigger*. It runs
	// with the shared mutex held.
	Process(f frames.Frame) ProcessFrameResult
	// TurnStarted readies per-turn state when a turn begins.
	TurnStarted()
	// TurnStopped clears per-turn state, including any buffered speech, when a
	// turn ends.
	TurnStopped()
	// Setup hands the strategy the pipeline's configuration, which it knows
	// before any frame arrives.
	Setup(s processor.Setup) error
	// Cleanup releases resources (timers).
	Cleanup()
	// ResolvesProposedTurnStopFrames reports whether this strategy resolves
	// proposals into turn stops.
	ResolvesProposedTurnStopFrames() bool
	// WaitsForInference reports whether this strategy waits for the turn's
	// inference before deciding.
	WaitsForInference() bool
	// DecisionScheduled reports whether this strategy has a decision of its own
	// scheduled, which the stop timeout then waits for.
	DecisionScheduled() bool
	attach(self StopStrategy, env strategyEnv)
}

// TranscriptWaiter is an optional interface a stop strategy implements when it
// can be told whether to hold a turn open until a transcript arrives.
//
// A pipeline driven by a speech-to-speech service takes transcripts off the
// critical path: the service is answering the audio, so waiting on a transcript
// only delays the turn. The user aggregator flips this on the strategies that
// expose it when such a service announces itself.
type TranscriptWaiter interface {
	StopStrategy
	// WaitForTranscript reports whether the turn is held open for a transcript.
	WaitForTranscript() bool
	// SetWaitForTranscript changes it for the turns that follow.
	SetWaitForTranscript(wait bool)
}

// StopStrategyBase is embedded by every stop strategy.
type StopStrategyBase struct {
	// EnableUserSpeakingFrames broadcasts a UserStoppedSpeakingFrame on turn stop.
	EnableUserSpeakingFrames bool
	env                      strategyEnv
	// self is the concrete strategy this base belongs to. See StartStrategyBase.
	self StopStrategy
	// warnedInferenceWithoutSpeculation records that the deprecated
	// TriggerInferenceTriggered was called, so it warns once.
	warnedInferenceWithoutSpeculation bool
}

func (b *StopStrategyBase) attach(self StopStrategy, env strategyEnv) {
	b.self, b.env = self, env
}

// TurnStarted is the default no-op.
func (b *StopStrategyBase) TurnStarted() {}

// TurnStopped is the default no-op.
func (b *StopStrategyBase) TurnStopped() {}

// Setup is the default no-op.
func (b *StopStrategyBase) Setup(processor.Setup) error { return nil }

// Cleanup is the default no-op.
func (b *StopStrategyBase) Cleanup() {}

// ResolvesProposedTurnStopFrames reports whether this strategy resolves
// proposals into turn stops.
//
// A ProposedUserStoppedSpeakingFrame is a request for a decision, so a strategy
// that acts on one consumes it: the frame stops traveling, and no resolver
// further along decides the same turn a second time. Override to true in a
// strategy that handles that frame, including one that holds the proposal for a
// while before deciding.
func (b *StopStrategyBase) ResolvesProposedTurnStopFrames() bool { return false }

// WaitsForInference reports whether this strategy waits for the turn's
// inference before deciding.
//
// In a StrategyModeAll list, the turn's inference runs once every other stop
// strategy has decided, and this strategy then decides on it. Override to true
// in a strategy that decides from the turn's answer, such as one reading the
// LLM's turn-completion marker.
func (b *StopStrategyBase) WaitsForInference() bool { return false }

// DecisionScheduled reports whether this strategy has a decision of its own
// scheduled.
//
// While a stop strategy has one, the controller's stop timeout waits for it
// instead of ending the turn. Override to return true only while the strategy
// is sure to decide soon on its own, as when a timer of its own will fire, so
// a turn whose decision never comes still times out.
func (b *StopStrategyBase) DecisionScheduled() bool { return false }

// StoppedOverrides overrides, for one turn, the flags the strategy was built
// with. A nil field keeps the configured value.
type StoppedOverrides struct {
	// EnableUserSpeakingFrames overrides whether a UserStoppedSpeakingFrame is
	// emitted for this turn. Set it false when something else in the pipeline
	// has already emitted it.
	EnableUserSpeakingFrames *bool
	// ConfirmsSpeculation reports that this turn end confirms a speculative
	// reply that has already been generated, so the model must not run again.
	ConfirmsSpeculation bool
}

// ReportStopped reports that the user is done, with the flags the strategy was
// built with.
//
// A strategy calls it when it decides the user's turn has ended. The
// controller combines the stop strategies' decisions into the turn end, and
// runs the turn's inference itself once they have decided.
func (b *StopStrategyBase) ReportStopped() {
	b.ReportStoppedOverriding(StoppedOverrides{})
}

// ReportStoppedOverriding reports that the user is done, overriding the
// strategy's configured flags for this turn.
func (b *StopStrategyBase) ReportStoppedOverriding(o StoppedOverrides) {
	if b.env.stopped == nil {
		return
	}
	b.env.stopped(b.self, UserTurnStoppedParams{
		EnableUserSpeakingFrames: boolOr(o.EnableUserSpeakingFrames, b.EnableUserSpeakingFrames),
		ConfirmsSpeculation:      o.ConfirmsSpeculation,
	})
}

// StartSpeculation starts a speculative inference against a provisional
// conversation carrying the turn text the speculation names, for a turn that
// has not ended yet. The reply it produces is held until the turn is confirmed.
//
// The controller runs a turn's inference itself once the stop strategies have
// decided, so a strategy only calls this to answer ahead of that, as an eager
// end of turn does.
func (b *StopStrategyBase) StartSpeculation(s UserTurnSpeculation) {
	if b.env.inferenceTriggered != nil {
		b.env.inferenceTriggered(b.self, &s)
	}
}

// CancelSpeculation withdraws the speculation this strategy has in flight,
// whatever voided it: the service withdrawing its prediction, a committed
// transcript that does not match, or a turn that never resolved the speculation
// at all.
//
// It is broadcast the same way the turn end is. A withdrawal that took the
// queued path would reach a downstream gate after the turn end that follows it,
// and the gate would release the reply the withdrawal was meant to discard.
func (b *StopStrategyBase) CancelSpeculation() {
	if b.env.speculationCanceled != nil {
		b.env.speculationCanceled(b.self)
	}
}

// TriggerStopped reports that the user is done. It no longer starts the turn's
// inference, which the controller now runs once the stop strategies have
// decided.
//
// Deprecated: use ReportStopped.
func (b *StopStrategyBase) TriggerStopped() { b.ReportStopped() }

// TriggerStoppedOverriding reports that the user is done, overriding the
// strategy's configured flags for this turn.
//
// Deprecated: use ReportStoppedOverriding.
func (b *StopStrategyBase) TriggerStoppedOverriding(o StoppedOverrides) {
	b.ReportStoppedOverriding(o)
}

// TriggerInferenceTriggered used to start the turn's inference. It does
// nothing now, with a warning the first time, since the controller runs the
// turn's inference once the stop strategies have decided.
//
// Deprecated: call ReportStopped when the user is done, and StartSpeculation to
// answer ahead of that.
func (b *StopStrategyBase) TriggerInferenceTriggered() {
	if b.warnedInferenceWithoutSpeculation {
		return
	}
	b.warnedInferenceWithoutSpeculation = true
	slog.Warn("turns: TriggerInferenceTriggered without a speculation does nothing, since "+
		"the controller runs the turn's inference once the stop strategies have decided; "+
		"call ReportStopped when the user is done", "strategy", fmt.Sprintf("%T", b.self))
}

// TriggerInferenceTriggeredSpeculating starts a speculative inference.
//
// Deprecated: use StartSpeculation.
func (b *StopStrategyBase) TriggerInferenceTriggeredSpeculating(s UserTurnSpeculation) {
	b.StartSpeculation(s)
}

// TriggerSpeculationCanceled withdraws the speculation this strategy has in
// flight.
//
// Deprecated: use CancelSpeculation.
func (b *StopStrategyBase) TriggerSpeculationCanceled() { b.CancelSpeculation() }

// TriggerFinalized reports that the user is done.
//
// Deprecated: use ReportStopped, which reports the same decision.
func (b *StopStrategyBase) TriggerFinalized() { b.ReportStopped() }

// TriggerFinalizedOverriding reports that the user is done, overriding the
// strategy's configured flags for this turn.
//
// Deprecated: use ReportStoppedOverriding, which reports the same decision.
func (b *StopStrategyBase) TriggerFinalizedOverriding(o StoppedOverrides) {
	b.ReportStoppedOverriding(o)
}

// Push sends a frame to the neighbor in dir.
func (b *StopStrategyBase) Push(f frames.Frame, dir processor.Direction) {
	if b.env.push != nil {
		b.env.push(f, dir)
	}
}

// Broadcast builds one frame per direction and sends them both ways. It takes a
// constructor so the two directions never share an instance.
func (b *StopStrategyBase) Broadcast(build func() frames.Frame) {
	if b.env.broadcast != nil {
		b.env.broadcast(build)
	}
}

// after schedules a mutex-guarded timeout for the strategy.
func (b *StopStrategyBase) after(d time.Duration, fn func()) func() { return b.env.after(d, fn) }
