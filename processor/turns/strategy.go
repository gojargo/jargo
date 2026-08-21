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

// after schedules fn to run after d with the shared mutex held, returning a
// cancel func. cancel must be called with the mutex held (from Process or
// another timer callback); calling it after the timer has fired is a no-op.
func (e strategyEnv) after(d time.Duration, fn func()) (cancel func()) {
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

// TriggerStarted signals that the user's turn has begun.
func (b *StartStrategyBase) TriggerStarted() {
	if b.env.started != nil {
		b.env.started(b.self, UserTurnStartedParams{
			EnableInterruptions:      b.EnableInterruptions,
			EnableUserSpeakingFrames: b.EnableUserSpeakingFrames,
		})
	}
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
	attach(self StopStrategy, env strategyEnv)
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

// TriggerStopped fires inference-triggered then finalized — the usual "turn is
// over" signal.
func (b *StopStrategyBase) TriggerStopped() {
	b.TriggerInferenceTriggered()
	b.TriggerFinalized()
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
	if b.env.stopped != nil {
		b.env.stopped(b.self, UserTurnStoppedParams{EnableUserSpeakingFrames: b.EnableUserSpeakingFrames})
	}
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
