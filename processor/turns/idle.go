package turns

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// IdleCallback runs when the conversation has stayed quiet past the configured
// timeout. It receives the controller so it can Push or Broadcast frames (a
// reminder, or an EndFrame to hang up). It runs off the frame path. Following
// Pipecat, it fires once per arming; escalation/retry, if wanted, is the
// caller's responsibility.
type IdleCallback func(ctx context.Context, c *UserIdleController) error

// IdleConfig configures a UserIdleController.
type IdleConfig struct {
	// Timeout is how long the conversation may stay quiet after the bot stops
	// speaking before Callback fires. A value <= 0 disables idle detection.
	Timeout time.Duration
	// Callback fires on idle. A nil callback disables idle detection.
	Callback IdleCallback
}

// UserIdleController fires IdleConfig.Callback when the conversation stays quiet
// after the bot finishes speaking. It is owned by a UserTurnProcessor, which
// feeds it every frame plus synthetic user-speaking frames on its turn
// decisions. The timer arms on BotStoppedSpeakingFrame (only when no user turn
// is in progress and no tool calls are pending) and is canceled by bot/user
// speech onset or a tool call.
//
// A UserIdleTimeoutUpdateFrame applies at once: it restarts a running timer with
// the new duration and, while the bot waits for the user to speak, arms the
// timer even when idle detection was disabled before.
type UserIdleController struct {
	cfg  IdleConfig
	emit Emitter

	mu      sync.Mutex
	ctx     context.Context
	timeout time.Duration
	// waitingForUser records the window where the bot has finished and nothing
	// has happened since, which is when a timeout update may arm the timer
	// without waiting for the next bot turn. It is tracked even while the
	// timeout is <= 0 and no timer runs.
	waitingForUser     bool
	userTurnInProgress bool
	functionCalls      int
	timerCancel        func()
}

// NewUserIdleController builds a user-idle controller.
func NewUserIdleController(cfg IdleConfig) *UserIdleController {
	return &UserIdleController{cfg: cfg, timeout: cfg.Timeout}
}

// Setup records the session context and the emitter used by the callback.
func (c *UserIdleController) Setup(ctx context.Context, emit Emitter) {
	c.mu.Lock()
	c.ctx = ctx
	c.emit = emit
	c.mu.Unlock()
}

// Stop cancels any pending timer. Its owner calls it at the end of the session,
// so the timer cannot report an idleness that only means the session is over.
//
// It has no Start to pair with: the timer is created in response to speech
// rather than at the start of the session, so there is nothing to bring up.
func (c *UserIdleController) Stop() {
	c.mu.Lock()
	c.cancelTimer()
	c.mu.Unlock()
}

// Cleanup cancels any pending timer.
func (c *UserIdleController) Cleanup() { c.Stop() }

// Push sends a frame to the neighbor in dir (for use from the callback).
func (c *UserIdleController) Push(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	c.mu.Lock()
	emit := c.emit
	c.mu.Unlock()
	if emit == nil {
		return nil
	}
	return emit.Push(ctx, f, dir)
}

// Broadcast sends a frame both downstream and upstream (for use from the
// callback), building a separate instance for each direction.
func (c *UserIdleController) Broadcast(ctx context.Context, build func() frames.Frame) error {
	c.mu.Lock()
	emit := c.emit
	c.mu.Unlock()
	if emit == nil {
		return nil
	}
	return emit.Broadcast(ctx, build)
}

// Process updates idle state from one frame and arms/cancels the timer.
func (c *UserIdleController) Process(f frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()
	switch fr := f.(type) {
	case *frames.UserIdleTimeoutUpdateFrame:
		c.timeout = fr.Timeout
		switch {
		case c.timeout <= 0:
			c.cancelTimer()
		case c.waitingForUser:
			// Apply the new timeout now: restart a running timer with it, or
			// arm one when idle detection was disabled before.
			c.startTimer()
		}
	case *frames.BotStoppedSpeakingFrame:
		if !c.userTurnInProgress && c.functionCalls == 0 {
			c.waitingForUser = true
			c.startTimer()
		}
	case *frames.BotStartedSpeakingFrame:
		c.waitingForUser = false
		c.cancelTimer()
	case *frames.UserStartedSpeakingFrame:
		c.waitingForUser = false
		c.userTurnInProgress = true
		c.cancelTimer()
	case *frames.UserStoppedSpeakingFrame:
		c.userTurnInProgress = false
	case *frames.FunctionCallsStartedFrame:
		c.waitingForUser = false
		c.functionCalls += len(fr.Calls)
		c.cancelTimer()
	case *frames.FunctionCallResultFrame, *frames.FunctionCallCancelFrame:
		if c.functionCalls > 0 {
			c.functionCalls--
		}
	}
}

// startTimer arms the one-shot idle timer. It runs with the mutex held.
func (c *UserIdleController) startTimer() {
	if c.timeout <= 0 || c.cfg.Callback == nil {
		return
	}
	c.cancelTimer()
	stopped := false
	timer := time.AfterFunc(c.timeout, func() {
		c.mu.Lock()
		if stopped {
			c.mu.Unlock()
			return
		}
		c.timerCancel = nil
		ctx := c.ctx
		c.mu.Unlock()
		_ = c.cfg.Callback(ctx, c) // off the lock: the callback may push frames
	})
	c.timerCancel = func() {
		stopped = true
		timer.Stop()
	}
}

func (c *UserIdleController) cancelTimer() {
	if c.timerCancel != nil {
		c.timerCancel()
		c.timerCancel = nil
	}
}
