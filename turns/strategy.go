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

	started            func(params UserTurnStartedParams)
	resetAggregation   func()
	inferenceTriggered func()
	stopped            func(params UserTurnStoppedParams)
	push               func(f frames.Frame, dir processor.Direction)
	broadcast          func(f frames.Frame)
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
	// Reset clears per-turn state at the start of each turn.
	Reset()
	// Cleanup releases resources (timers).
	Cleanup()
	attach(env strategyEnv)
}

// StartStrategyBase is embedded by every start strategy. It carries the
// open-turn flags and the trigger helpers.
type StartStrategyBase struct {
	// EnableInterruptions broadcasts an InterruptionFrame on turn start.
	EnableInterruptions bool
	// EnableUserSpeakingFrames broadcasts a UserStartedSpeakingFrame on turn start.
	EnableUserSpeakingFrames bool
	env                      strategyEnv
}

func (b *StartStrategyBase) attach(env strategyEnv) { b.env = env }

// Reset is the default no-op.
func (b *StartStrategyBase) Reset() {}

// Cleanup is the default no-op.
func (b *StartStrategyBase) Cleanup() {}

// TriggerStarted signals that the user's turn has begun.
func (b *StartStrategyBase) TriggerStarted() {
	if b.env.started != nil {
		b.env.started(UserTurnStartedParams{
			EnableInterruptions:      b.EnableInterruptions,
			EnableUserSpeakingFrames: b.EnableUserSpeakingFrames,
		})
	}
}

// TriggerResetAggregation asks the user aggregator to drop the in-progress
// aggregation (e.g. pre-wake-phrase speech).
func (b *StartStrategyBase) TriggerResetAggregation() {
	if b.env.resetAggregation != nil {
		b.env.resetAggregation()
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
	// Reset clears per-turn state.
	Reset()
	// Cleanup releases resources (timers).
	Cleanup()
	attach(env strategyEnv)
}

// StopStrategyBase is embedded by every stop strategy.
type StopStrategyBase struct {
	// EnableUserSpeakingFrames broadcasts a UserStoppedSpeakingFrame on turn stop.
	EnableUserSpeakingFrames bool
	env                      strategyEnv
}

func (b *StopStrategyBase) attach(env strategyEnv) { b.env = env }

// Reset is the default no-op.
func (b *StopStrategyBase) Reset() {}

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
		b.env.inferenceTriggered()
	}
}

// TriggerFinalized signals that the turn is semantically final.
func (b *StopStrategyBase) TriggerFinalized() {
	if b.env.stopped != nil {
		b.env.stopped(UserTurnStoppedParams{EnableUserSpeakingFrames: b.EnableUserSpeakingFrames})
	}
}

// Push sends a frame to the neighbor in dir.
func (b *StopStrategyBase) Push(f frames.Frame, dir processor.Direction) {
	if b.env.push != nil {
		b.env.push(f, dir)
	}
}

// Broadcast sends a frame both downstream and upstream.
func (b *StopStrategyBase) Broadcast(f frames.Frame) {
	if b.env.broadcast != nil {
		b.env.broadcast(f)
	}
}

// after schedules a mutex-guarded timeout for the strategy.
func (b *StopStrategyBase) after(d time.Duration, fn func()) func() { return b.env.after(d, fn) }
