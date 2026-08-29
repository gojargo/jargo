package turns

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// defaultStopTimeout is the watchdog that force-stops a turn stuck open with no
// strategy firing.
const defaultStopTimeout = 5 * time.Second

// errNotSetUp is returned when strategies are applied to a controller that has
// not been given the pipeline configuration yet.
//
//nolint:gochecknoglobals // sentinel error
var errNotSetUp = errors.New("turns: the controller was not set up")

// ControllerHooks are the callbacks the controller invokes upward (to the
// UserTurnProcessor). They all run with the controller's mutex held.
type ControllerHooks struct {
	// Started, Stopped, InferenceTriggered and ResetAggregation each carry the
	// strategy that made the decision, so what the hooks report can name it.
	Started            func(ctx context.Context, s StartStrategy, params UserTurnStartedParams)
	Stopped            func(ctx context.Context, s StopStrategy, params UserTurnStoppedParams)
	InferenceTriggered func(ctx context.Context, s StopStrategy)
	StopTimeout        func(ctx context.Context)
	ResetAggregation   func(ctx context.Context, s StartStrategy)
	Push               func(ctx context.Context, f frames.Frame, dir processor.Direction)
	Broadcast          func(ctx context.Context, build func() frames.Frame)
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
	// setup is the pipeline configuration the strategies were set up with, so
	// a strategy added later is given the same one. setupDone records that it
	// was actually handed over, since the zero Setup is a legitimate value.
	setup     processor.Setup
	setupDone bool

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
func (c *UserTurnController) Setup(ctx context.Context, st processor.Setup) error {
	c.mu.Lock()
	c.ctx = ctx
	c.setup = st
	c.setupDone = true
	c.mu.Unlock()
	return c.setupStrategies()
}

// Stop tears the turn watchdog down, leaving the strategies alone. Its owner
// calls it at the end of the session: left running the watchdog reports what
// ending looks like rather than a turn that really stalled, since no turn
// finishes once the session is over. The strategies may be shared, so cleaning
// them up waits for Cleanup.
func (c *UserTurnController) Stop() {
	c.mu.Lock()
	if c.watchdogCancel != nil {
		c.watchdogCancel()
		c.watchdogCancel = nil
	}
	c.mu.Unlock()
}

// Cleanup stops the watchdog and cleans up the strategies.
func (c *UserTurnController) Cleanup() {
	c.Stop()
	c.cleanupStrategies()
}

// Strategies are the chains the controller is currently running.
func (c *UserTurnController) Strategies() UserTurnStrategies {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.strategies
}

// UpdateStrategies replaces the current strategies with the given ones. The
// chains that go are cleaned up and the ones that arrive are set up with the
// same pipeline configuration the controller was given, so a caller swapping
// them does not have to hand it over again.
//
// Empty chains fall back to the defaults, as they do at construction.
func (c *UserTurnController) UpdateStrategies(strategies UserTurnStrategies) error {
	c.cleanupStrategies()
	strategies.fillDefaults()
	c.mu.Lock()
	c.strategies = strategies
	c.mu.Unlock()
	return c.setupStrategies()
}

// setupStrategies binds every strategy to the shared environment and hands it
// the pipeline configuration.
func (c *UserTurnController) setupStrategies() error {
	c.mu.Lock()
	strategies, st, done := c.strategies, c.setup, c.setupDone
	c.mu.Unlock()
	if !done {
		return errNotSetUp
	}
	for _, s := range strategies.Start {
		s.attach(s, c.startEnv())
		if err := s.Setup(st); err != nil {
			return err
		}
	}
	for _, s := range strategies.Stop {
		s.attach(s, c.stopEnv())
		if err := s.Setup(st); err != nil {
			return err
		}
	}
	return nil
}

// cleanupStrategies releases what every strategy holds.
func (c *UserTurnController) cleanupStrategies() {
	c.mu.Lock()
	strategies := c.strategies
	c.mu.Unlock()
	for _, s := range strategies.Start {
		s.Cleanup()
	}
	for _, s := range strategies.Stop {
		s.Cleanup()
	}
}

// ResolvesProposedTurnStartFrames reports whether any active start strategy
// resolves proposed turn starts.
//
// A proposal is resolved once, so a caller holding this controller stops
// forwarding a ProposedUserStartedSpeakingFrame when this is true: passing it
// along would let a resolver further down the pipeline decide the same turn a
// second time.
func (c *UserTurnController) ResolvesProposedTurnStartFrames() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, s := range c.strategies.Start {
		if s.ResolvesProposedTurnStartFrames() {
			return true
		}
	}
	return false
}

// ResolvesProposedTurnStopFrames is the end-of-turn counterpart to
// ResolvesProposedTurnStartFrames.
func (c *UserTurnController) ResolvesProposedTurnStopFrames() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, s := range c.strategies.Stop {
		if s.ResolvesProposedTurnStopFrames() {
			return true
		}
	}
	return false
}

// Locked runs fn with the controller's lock held, so a caller can keep its own
// turn-scoped state in step with the decisions the strategies make.
//
// A strategy decides a turn is over from a timer, and the whole stop sequence
// runs from that timer under this lock: the inference trigger and the
// finalization that follows it are one indivisible step. State a caller updates
// through here therefore cannot land in the middle of that step, only before or
// after it, which is what keeps a turn's end from being observed half-made.
//
// fn must not push frames or re-enter the controller.
func (c *UserTurnController) Locked(fn func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	fn()
}

// Process taps one frame: it updates the speaking flag, re-arms the watchdog, and
// runs the start then stop strategy chains. It holds the mutex throughout, so the
// strategies' synchronous Trigger* callbacks run safely without re-locking.
func (c *UserTurnController) Process(f frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch f.(type) {
	case *frames.UserStartedSpeakingFrame, *frames.ProposedUserStartedSpeakingFrame,
		*frames.VADUserStartedSpeakingFrame:
		c.userSpeaking = true
		c.rearmWatchdog()
	case *frames.UserStoppedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame,
		*frames.VADUserStoppedSpeakingFrame:
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
		mu:      &c.mu,
		started: c.onStartTriggered,
		resetAggregation: func(s StartStrategy) {
			if c.hooks.ResetAggregation != nil {
				c.hooks.ResetAggregation(c.ctx, s)
			}
		},
		push:      c.pushHook(),
		broadcast: c.broadcastHook(),
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

func (c *UserTurnController) broadcastHook() func(func() frames.Frame) {
	return func(build func() frames.Frame) {
		if c.hooks.Broadcast != nil {
			c.hooks.Broadcast(c.ctx, build)
		}
	}
}

// onStartTriggered opens a turn (guarded against double-start), resets all
// strategies for the fresh turn, and notifies upward.
func (c *UserTurnController) onStartTriggered(s StartStrategy, params UserTurnStartedParams) {
	if c.userTurn {
		return
	}
	c.userTurn = true
	c.rearmWatchdog()
	c.notifyTurnStarted()
	if c.hooks.Started != nil {
		c.hooks.Started(c.ctx, s, params)
	}
}

// onInferenceTriggered fires only during an active turn.
func (c *UserTurnController) onInferenceTriggered(s StopStrategy) {
	if !c.userTurn {
		return
	}
	c.rearmWatchdog()
	if c.hooks.InferenceTriggered != nil {
		c.hooks.InferenceTriggered(c.ctx, s)
	}
}

// onStopTriggered closes a turn (guarded against double-stop), resets the stop
// strategies, and notifies upward.
func (c *UserTurnController) onStopTriggered(s StopStrategy, params UserTurnStoppedParams) {
	if !c.userTurn {
		return
	}
	// Never finalize while the user is audibly speaking. A stop strategy can
	// finalize on a latent signal, an LLM verdict that resolves only after the
	// user resumed, which is stale by the time it lands. Keeping the turn open
	// lets the next inference re-evaluate, and the watchdog still finalizes if the
	// user then falls silent. Detector strategies finalize only once the user has
	// stopped, so this costs them nothing.
	if c.userSpeaking {
		return
	}
	c.userTurn = false
	c.rearmWatchdog()
	c.notifyTurnStopped()
	if c.hooks.Stopped != nil {
		c.hooks.Stopped(c.ctx, s, params)
	}
}

// notifyTurnStarted readies every strategy for the turn now beginning.
func (c *UserTurnController) notifyTurnStarted() {
	for _, s := range c.strategies.Start {
		s.TurnStarted()
	}
	for _, s := range c.strategies.Stop {
		s.TurnStarted()
	}
}

// notifyTurnStopped tells every strategy the turn ended.
func (c *UserTurnController) notifyTurnStopped() {
	for _, s := range c.strategies.Start {
		s.TurnStopped()
	}
	for _, s := range c.strategies.Stop {
		s.TurnStopped()
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
		// No strategy decided this one: the turn is being closed because none
		// did, so there is nothing to attribute it to.
		c.onStopTriggered(nil, DefaultStoppedParams())
	}
}
