package turns

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// defaultStopTimeout is the watchdog that force-stops a turn stuck open with no
// strategy firing.
const defaultStopTimeout = 5 * time.Second

// ControllerHooks are the callbacks the controller invokes upward (to the
// UserTurnProcessor). They all run with the controller's mutex held.
type ControllerHooks struct {
	Started            func(ctx context.Context, params UserTurnStartedParams)
	Stopped            func(ctx context.Context, params UserTurnStoppedParams)
	InferenceTriggered func(ctx context.Context)
	StopTimeout        func(ctx context.Context)
	ResetAggregation   func(ctx context.Context)
	Push               func(ctx context.Context, f frames.Frame, dir processor.Direction)
	Broadcast          func(ctx context.Context, f frames.Frame)
}

// UserTurnController runs the start and stop strategy chains and owns the
// user-turn state machine: double-start/stop guards and a stop-timeout watchdog.
// A single mutex serializes every state mutation — Process and all strategy
// timer callbacks acquire it — so strategies need no locking of their own.
type UserTurnController struct {
	strategies  UserTurnStrategies
	stopTimeout time.Duration
	hooks       ControllerHooks

	mu  sync.Mutex
	ctx context.Context

	userSpeaking   bool
	userTurn       bool
	watchdogCancel func()
}

// NewUserTurnController builds a controller. A zero stopTimeout uses 5s; empty
// strategy lists fall back to the defaults.
func NewUserTurnController(strategies UserTurnStrategies, stopTimeout time.Duration) *UserTurnController {
	if stopTimeout == 0 {
		stopTimeout = defaultStopTimeout
	}
	strategies.fillDefaults()
	return &UserTurnController{strategies: strategies, stopTimeout: stopTimeout}
}

// SetHooks installs the upward callbacks. Call before Setup.
func (c *UserTurnController) SetHooks(h ControllerHooks) { c.hooks = h }

// Setup records the session context and binds each strategy to the shared
// environment.
func (c *UserTurnController) Setup(ctx context.Context) {
	c.mu.Lock()
	c.ctx = ctx
	c.mu.Unlock()
	for _, s := range c.strategies.Start {
		s.attach(c.startEnv())
	}
	for _, s := range c.strategies.Stop {
		s.attach(c.stopEnv())
	}
}

// Cleanup stops the watchdog and cleans up the strategies.
func (c *UserTurnController) Cleanup() {
	c.mu.Lock()
	if c.watchdogCancel != nil {
		c.watchdogCancel()
		c.watchdogCancel = nil
	}
	c.mu.Unlock()
	for _, s := range c.strategies.Start {
		s.Cleanup()
	}
	for _, s := range c.strategies.Stop {
		s.Cleanup()
	}
}

// Process taps one frame: it updates the speaking flag, re-arms the watchdog, and
// runs the start then stop strategy chains. It holds the mutex throughout, so the
// strategies' synchronous Trigger* callbacks run safely without re-locking.
func (c *UserTurnController) Process(f frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch f.(type) {
	case *frames.UserStartedSpeakingFrame, *frames.VADUserStartedSpeakingFrame:
		c.userSpeaking = true
		c.rearmWatchdog()
	case *frames.UserStoppedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame:
		c.userSpeaking = false
		c.rearmWatchdog()
	case *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame:
		c.rearmWatchdog()
	}

	for _, s := range c.strategies.Start {
		if s.Process(f) == Stop {
			break
		}
	}
	for _, s := range c.strategies.Stop {
		if s.Process(f) == Stop {
			break
		}
	}
}

// startEnv builds the environment shared with start strategies.
func (c *UserTurnController) startEnv() strategyEnv {
	return strategyEnv{
		mu:               &c.mu,
		started:          c.onStartTriggered,
		resetAggregation: func() { c.fire(c.hooks.ResetAggregation) },
		push:             c.pushHook(),
		broadcast:        c.broadcastHook(),
	}
}

// stopEnv builds the environment shared with stop strategies.
func (c *UserTurnController) stopEnv() strategyEnv {
	return strategyEnv{
		mu:                 &c.mu,
		inferenceTriggered: c.onInferenceTriggered,
		stopped:            c.onStopTriggered,
		push:               c.pushHook(),
		broadcast:          c.broadcastHook(),
	}
}

func (c *UserTurnController) pushHook() func(frames.Frame, processor.Direction) {
	return func(f frames.Frame, dir processor.Direction) {
		if c.hooks.Push != nil {
			c.hooks.Push(c.ctx, f, dir)
		}
	}
}

func (c *UserTurnController) broadcastHook() func(frames.Frame) {
	return func(f frames.Frame) {
		if c.hooks.Broadcast != nil {
			c.hooks.Broadcast(c.ctx, f)
		}
	}
}

func (c *UserTurnController) fire(fn func(context.Context)) {
	if fn != nil {
		fn(c.ctx)
	}
}

// onStartTriggered opens a turn (guarded against double-start), resets all
// strategies for the fresh turn, and notifies upward.
func (c *UserTurnController) onStartTriggered(params UserTurnStartedParams) {
	if c.userTurn {
		return
	}
	c.userTurn = true
	c.rearmWatchdog()
	c.resetStrategies(true)
	if c.hooks.Started != nil {
		c.hooks.Started(c.ctx, params)
	}
}

// onInferenceTriggered fires only during an active turn.
func (c *UserTurnController) onInferenceTriggered() {
	if !c.userTurn {
		return
	}
	c.rearmWatchdog()
	if c.hooks.InferenceTriggered != nil {
		c.hooks.InferenceTriggered(c.ctx)
	}
}

// onStopTriggered closes a turn (guarded against double-stop), resets the stop
// strategies, and notifies upward.
func (c *UserTurnController) onStopTriggered(params UserTurnStoppedParams) {
	if !c.userTurn {
		return
	}
	c.userTurn = false
	c.rearmWatchdog()
	c.resetStrategies(false)
	if c.hooks.Stopped != nil {
		c.hooks.Stopped(c.ctx, params)
	}
}

func (c *UserTurnController) resetStrategies(includeStart bool) {
	if includeStart {
		for _, s := range c.strategies.Start {
			s.Reset()
		}
	}
	for _, s := range c.strategies.Stop {
		s.Reset()
	}
}

// rearmWatchdog restarts the stop-timeout timer. It runs with the mutex held.
func (c *UserTurnController) rearmWatchdog() {
	if c.watchdogCancel != nil {
		c.watchdogCancel()
	}
	stopped := false
	timer := time.AfterFunc(c.stopTimeout, func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		if stopped {
			return
		}
		c.onWatchdog()
	})
	c.watchdogCancel = func() {
		stopped = true
		timer.Stop()
	}
}

// onWatchdog force-stops a turn that has been stuck open with the user silent.
func (c *UserTurnController) onWatchdog() {
	if c.userTurn && !c.userSpeaking {
		if c.hooks.StopTimeout != nil {
			c.hooks.StopTimeout(c.ctx)
		}
		c.onStopTriggered(DefaultStoppedParams())
	}
}
