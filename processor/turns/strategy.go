package turns

import (
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
	started            func(s StartStrategy, params UserTurnStartedParams)
	resetAggregation   func(s StartStrategy)
	inferenceTriggered func(s StopStrategy)
	stopped            func(s StopStrategy, params UserTurnStoppedParams)
	push               func(f frames.Frame, dir processor.Direction)
	broadcast          func(build func() frames.Frame)
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

// TriggerStarted signals that the user's turn has begun, with the flags the
// strategy was built with.
func (b *StartStrategyBase) TriggerStarted() {
	b.TriggerStartedOverriding(StartedOverrides{})
}

// TriggerStartedOverriding signals that the user's turn has begun, overriding
// the strategy's configured flags for this turn.
func (b *StartStrategyBase) TriggerStartedOverriding(o StartedOverrides) {
	if b.env.started == nil {
		return
	}
	b.env.started(b.self, UserTurnStartedParams{
		EnableInterruptions:      boolOr(o.EnableInterruptions, b.EnableInterruptions),
		EnableUserSpeakingFrames: boolOr(o.EnableUserSpeakingFrames, b.EnableUserSpeakingFrames),
	})
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

// StoppedOverrides overrides, for one turn, the flags the strategy was built
// with. A nil field keeps the configured value.
type StoppedOverrides struct {
	// EnableUserSpeakingFrames overrides whether a UserStoppedSpeakingFrame is
	// emitted for this turn. Set it false when something else in the pipeline
	// has already emitted it.
	EnableUserSpeakingFrames *bool
}

// TriggerStopped fires inference-triggered then finalized, the usual "turn is
// over" signal.
//
// To leave finalization to another strategy, so this one fires only the
// inference trigger, wrap it with Deferred rather than changing the call.
func (b *StopStrategyBase) TriggerStopped() {
	b.TriggerStoppedOverriding(StoppedOverrides{})
}

// TriggerStoppedOverriding fires inference-triggered then finalized, overriding
// the strategy's configured flags for this turn.
func (b *StopStrategyBase) TriggerStoppedOverriding(o StoppedOverrides) {
	b.TriggerInferenceTriggered()
	b.TriggerFinalizedOverriding(o)
}

// TriggerInferenceTriggered signals that there is enough evidence to start LLM
// inference, without finalizing the turn.
func (b *StopStrategyBase) TriggerInferenceTriggered() {
	if b.env.inferenceTriggered != nil {
		b.env.inferenceTriggered(b.self)
	}
}

// TriggerFinalized signals that the turn is semantically final.
func (b *StopStrategyBase) TriggerFinalized() {
	b.TriggerFinalizedOverriding(StoppedOverrides{})
}

// TriggerFinalizedOverriding signals that the turn is semantically final,
// overriding the strategy's configured flags for this turn.
func (b *StopStrategyBase) TriggerFinalizedOverriding(o StoppedOverrides) {
	if b.env.stopped == nil {
		return
	}
	b.env.stopped(b.self, UserTurnStoppedParams{
		EnableUserSpeakingFrames: boolOr(o.EnableUserSpeakingFrames, b.EnableUserSpeakingFrames),
	})
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
