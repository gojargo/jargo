package turns

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// IdleCallback runs when the conversation has stayed quiet past one of the
// configured timeouts. It receives the controller so it can Push or Broadcast
// frames (a reminder, or an EndFrame to hang up). It runs off the frame path. It
// fires once per arming; escalation/retry, if wanted, is the caller's
// responsibility.
type IdleCallback func(ctx context.Context, c *UserIdleController) error

// IdleConfig configures a UserIdleController.
type IdleConfig struct {
	// Timeout is how long the conversation may stay quiet after the bot stops
	// speaking, with no background work in progress, before Callback fires. A
	// value <= 0 disables idle detection.
	Timeout time.Duration
	// Callback fires when the user is idle (the on_user_turn_idle event). A nil
	// callback disables idle detection.
	Callback IdleCallback
	// WaitingTimeout is how long the conversation may stay quiet, with
	// background work in progress, before WaitingCallback fires. A value <= 0,
	// the default, disables waiting detection.
	WaitingTimeout time.Duration
	// WaitingCallback fires when the user has waited quietly on background work
	// (the on_user_waiting event). A nil callback disables waiting detection.
	WaitingCallback IdleCallback
}

// UserIdleController detects a conversation that has gone quiet: the user idle,
// or waiting. It is owned by a UserTurnProcessor or a user aggregator, which
// feeds it every frame plus synthetic user-speaking frames on its turn
// decisions.
//
// The controller times the silence after the bot finishes speaking and reports
// it as one of two events, told apart by whether background work is in
// progress. Background work is what a BackgroundWorkStartedFrame announced and
// no BackgroundWorkFinishedFrame has ended yet: an asynchronous function call,
// or a task delegated to an LLMWithBackend backend, for instance.
//
//   - With no background work, the user is idle: it is their turn, and they
//     have said nothing. After IdleConfig.Timeout, IdleConfig.Callback runs.
//   - With background work in progress, the user is waiting on it. After
//     IdleConfig.WaitingTimeout, IdleConfig.WaitingCallback runs.
//
// A synchronous function call is not background work: it is part of the bot's
// turn, whose result the LLM waits for before it can say anything, so the idle
// timer does not run while one is pending.
//
// Either timer starts when a BotStoppedSpeakingFrame is received and is canceled
// when someone starts speaking again (UserStartedSpeakingFrame or
// BotStartedSpeakingFrame). Each fires once per start; the bot speaking again,
// the usual answer to either event, starts it over. A callback that answers with
// nothing starts it over by hand, with WaitForUser. Background work starting
// while the conversation is already quiet starts the waiting timer, and the last
// of it finishing hands over to the idle one.
//
// The timers are suppressed while a user turn is in progress to avoid false
// triggers during interruptions (where BotStoppedSpeakingFrame arrives while the
// user is still speaking).
//
// A UserIdleTimeoutUpdateFrame applies at once: it restarts a running idle timer
// with the new duration and, while the bot waits for the user, arms the timer
// even when idle detection was disabled before. A UserWaitingTimeoutUpdateFrame
// does the same for the waiting timer.
type UserIdleController struct {
	cfg  IdleConfig
	emit Emitter

	mu             sync.Mutex
	ctx            context.Context
	idleTimeout    time.Duration
	waitingTimeout time.Duration
	// waitingForUser records the window where the bot has finished and nothing
	// has happened since, which is when a timeout update may arm the timer
	// without waiting for the next bot turn. It is tracked even while the
	// timeout is <= 0 and no timer runs.
	waitingForUser     bool
	botSpeaking        bool
	userTurnInProgress bool
	// functionCalls are the function calls seen starting and not yet finished,
	// by tool call id.
	functionCalls map[string]struct{}
	// backgroundWork is the work announced as started and not yet finished, by
	// work id.
	backgroundWork map[string]struct{}
	timerCancel    func()
	// quietSince is when the conversation last went quiet, while it is. It is
	// zero while someone is speaking.
	quietSince time.Time
}

// NewUserIdleController builds a user-idle controller.
func NewUserIdleController(cfg IdleConfig) *UserIdleController {
	return &UserIdleController{
		cfg:            cfg,
		idleTimeout:    cfg.Timeout,
		waitingTimeout: cfg.WaitingTimeout,
		functionCalls:  make(map[string]struct{}),
		backgroundWork: make(map[string]struct{}),
	}
}

// QuietSecs is how long nothing has been said, in seconds; 0 while someone is
// speaking.
func (c *UserIdleController) QuietSecs() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.quietSince.IsZero() {
		return 0
	}
	return time.Since(c.quietSince).Seconds()
}

// Setup records the session context and the emitter used by the callback.
func (c *UserIdleController) Setup(ctx context.Context, emit Emitter) {
	c.mu.Lock()
	c.ctx = ctx
	c.emit = emit
	c.mu.Unlock()
}

// Stop cancels any pending timer. Its owner calls it at the end of the session,
// so the timer cannot report a silence that only means the session is over.
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

// Process updates the controller's state from one frame and arms or cancels the
// timers.
func (c *UserIdleController) Process(f frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()
	switch f.(type) {
	case *frames.UserIdleTimeoutUpdateFrame, *frames.UserWaitingTimeoutUpdateFrame:
		c.processTimeoutUpdate(f)
	case *frames.BotStoppedSpeakingFrame, *frames.BotStartedSpeakingFrame,
		*frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame,
		*frames.LLMMarkerResponseFrame:
		c.processSpeech(f)
	default:
		c.processWork(f)
	}
}

// processTimeoutUpdate applies a timeout update at once. Callers hold c.mu.
func (c *UserIdleController) processTimeoutUpdate(f frames.Frame) {
	switch fr := f.(type) {
	case *frames.UserIdleTimeoutUpdateFrame:
		c.idleTimeout = fr.Timeout
		switch {
		case c.idleTimeout <= 0:
			if len(c.backgroundWork) == 0 {
				c.cancelTimer()
			}
		case c.waitingForUser && len(c.backgroundWork) == 0:
			// Apply the new timeout now: restart a running timer with it, or
			// arm one when idle detection was disabled before.
			c.startIdleTimer()
		}
	case *frames.UserWaitingTimeoutUpdateFrame:
		c.waitingTimeout = fr.Timeout
		switch {
		case c.waitingTimeout <= 0:
			if len(c.backgroundWork) > 0 {
				c.cancelTimer()
			}
		case c.quiet() && len(c.backgroundWork) > 0:
			c.startWaitingTimer()
		}
	}
}

// processSpeech follows who is speaking, which starts and cancels the timers.
// Callers hold c.mu.
func (c *UserIdleController) processSpeech(f frames.Frame) {
	switch fr := f.(type) {
	case *frames.BotStoppedSpeakingFrame:
		c.botSpeaking = false
		// Interruption case: the frame order is UserStartedSpeaking →
		// BotStoppedSpeaking → (user keeps talking) → UserStoppedSpeaking.
		// Without the user-turn guard the timer would start while the user is
		// still speaking.
		if c.userTurnInProgress {
			return
		}
		c.quietSince = time.Now()
		c.startAfterBotTurn()
	case *frames.BotStartedSpeakingFrame:
		c.botSpeaking = true
		c.waitingForUser = false
		c.quietSince = time.Time{}
		c.cancelTimer()
	case *frames.UserStartedSpeakingFrame:
		c.waitingForUser = false
		c.userTurnInProgress = true
		c.quietSince = time.Time{}
		c.cancelTimer()
	case *frames.UserStoppedSpeakingFrame:
		c.userTurnInProgress = false
		if !c.botSpeaking {
			c.quietSince = time.Now()
		}
	case *frames.LLMMarkerResponseFrame:
		// A reply the LLM chose to leave unspoken ends its turn like speech
		// would, with the silence going on: the next interval gets a fresh
		// decision.
		if fr.Kind == frames.LLMMarkerKindSilent && c.quiet() {
			c.startAfterBotTurn()
		}
	}
}

// processWork follows the function calls and the background work in progress.
// Callers hold c.mu.
func (c *UserIdleController) processWork(f frames.Frame) {
	switch fr := f.(type) {
	case *frames.FunctionCallsStartedFrame:
		c.waitingForUser = false
		for _, call := range fr.Calls {
			c.functionCalls[call.ID] = struct{}{}
		}
		c.cancelTimer()
	case *frames.FunctionCallResultFrame:
		// An async tool can report progress with intermediate results while it
		// keeps running; only its final result settles the call.
		if fr.Properties.Final() {
			c.settleFunctionCall(fr.ToolCallID)
		}
	case *frames.FunctionCallCancelFrame:
		c.settleFunctionCall(fr.ToolCallID)
	case *frames.BackgroundWorkStartedFrame:
		first := len(c.backgroundWork) == 0
		c.backgroundWork[fr.WorkID] = struct{}{}
		// Background work starting in a quiet conversation: the user is waiting
		// from now on, so an idle timer running at this point is replaced.
		if first && c.quiet() {
			c.cancelTimer()
			c.startWaitingTimer()
		}
	case *frames.BackgroundWorkFinishedFrame:
		delete(c.backgroundWork, fr.WorkID)
		if len(c.backgroundWork) == 0 && c.quiet() {
			// Nothing left to wait on: the user is idle from here.
			if len(c.functionCalls) == 0 {
				c.waitingForUser = true
			}
			c.startTimer()
		}
	}
}

// WaitingForUser reports whether the bot has finished responding and is waiting
// for the user. It is false while the bot is thinking, speaking or running a
// function call, and during a user turn.
func (c *UserIdleController) WaitingForUser() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.waitingForUser
}

// FunctionCallsInProgress reports whether function calls have started and not
// finished.
func (c *UserIdleController) FunctionCallsInProgress() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.functionCalls) > 0
}

// WaitForUser starts waiting for the user after a turn the bot will not answer.
//
// The timer normally starts when the bot stops speaking. A user turn that ends
// with nothing for the bot to answer (no transcript, say) cancels the timer
// without the bot speaking again afterwards, so the caller re-arms it here: the
// waiting timer with background work in progress, the idle timer otherwise. It
// does nothing while the bot is speaking or a user turn is in progress, and
// starts no idle timer while function calls are pending.
func (c *UserIdleController) WaitForUser() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.quiet() {
		c.startAfterBotTurn()
	}
}

// startAfterBotTurn starts the timer for the silence after the bot's turn.
// Callers hold c.mu.
//
// Function call case: normally FunctionCallsStarted arrives after
// BotStoppedSpeaking and cancels the timer directly. But a race can make
// FunctionCallsStarted arrive before BotStoppedSpeaking when a TTSSpeakFrame is
// pushed from the function-calls-started handler, so the guard on the calls in
// progress keeps the idle timer from starting while a function call is pending.
func (c *UserIdleController) startAfterBotTurn() {
	if len(c.functionCalls) == 0 {
		// Track the waiting-for-user window even when the timeout is currently
		// <= 0 (no timer), so a later timeout update can arm the timer without
		// waiting for the next bot turn.
		c.waitingForUser = true
	}
	c.startTimer()
}

// settleFunctionCall counts a function call as finished. Callers hold c.mu.
//
// The call's start canceled the timer. A reply that speaks first and calls the
// function after, and then says nothing on its result, leaves the conversation
// quiet with the background work still in flight, so the last call settling
// starts the waiting timer over; the bot speaking on the result restarts it as
// usual.
//
// A call the controller did not see start settles nothing: a built-in tool whose
// calls are not announced still reports a result, and a bus bridge can deliver a
// result a second time.
func (c *UserIdleController) settleFunctionCall(toolCallID string) {
	if _, ok := c.functionCalls[toolCallID]; !ok {
		return
	}
	delete(c.functionCalls, toolCallID)
	if len(c.functionCalls) == 0 && len(c.backgroundWork) > 0 && c.quiet() {
		c.startWaitingTimer()
	}
}

// quiet reports whether nobody is speaking: no bot speech and no open user turn.
// Callers hold c.mu.
func (c *UserIdleController) quiet() bool {
	return !c.botSpeaking && !c.userTurnInProgress
}

// startTimer starts the timer for the kind of silence this is. Callers hold
// c.mu.
//
// It is the waiting timer when background work is in progress, and the idle
// timer otherwise, unless a function call is pending, whose result will run the
// bot. A timer of the other kind that is still running is canceled either way,
// since it was timing a silence that is over.
func (c *UserIdleController) startTimer() {
	c.cancelTimer()
	if len(c.backgroundWork) > 0 {
		c.startWaitingTimer()
	} else if len(c.functionCalls) == 0 {
		c.startIdleTimer()
	}
}

// startIdleTimer starts (or restarts) the idle timer. Callers hold c.mu.
func (c *UserIdleController) startIdleTimer() {
	if c.idleTimeout <= 0 || c.cfg.Callback == nil {
		return
	}
	c.armTimer(c.idleTimeout, c.cfg.Callback)
}

// startWaitingTimer starts (or restarts) the waiting timer. Callers hold c.mu.
func (c *UserIdleController) startWaitingTimer() {
	if c.waitingTimeout <= 0 || c.cfg.WaitingCallback == nil {
		return
	}
	c.armTimer(c.waitingTimeout, c.cfg.WaitingCallback)
}

// armTimer replaces whichever timer is running with a one-shot one that runs
// callback after timeout. Callers hold c.mu.
func (c *UserIdleController) armTimer(timeout time.Duration, callback IdleCallback) {
	c.cancelTimer()
	stopped := false
	timer := time.AfterFunc(timeout, func() {
		c.mu.Lock()
		if stopped {
			c.mu.Unlock()
			return
		}
		c.timerCancel = nil
		ctx := c.ctx
		c.mu.Unlock()
		_ = callback(ctx, c) // off the lock: the callback may push frames
	})
	c.timerCancel = func() {
		stopped = true
		timer.Stop()
	}
}

// cancelTimer cancels whichever timer is running. Callers hold c.mu.
func (c *UserIdleController) cancelTimer() {
	if c.timerCancel != nil {
		c.timerCancel()
		c.timerCancel = nil
	}
}
