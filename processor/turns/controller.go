package turns

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// defaultStopTimeout is how long an open turn may go without user activity
// before the controller ends it.
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
	Started func(ctx context.Context, s StartStrategy, params UserTurnStartedParams)
	Stopped func(ctx context.Context, s StopStrategy, params UserTurnStoppedParams)
	// InferenceTriggered carries the speculation the inference answers, or nil
	// on an ordinary one. The controller runs the turn's inference once the
	// stop strategies have decided; a strategy only starts one to speculate.
	InferenceTriggered func(ctx context.Context, s StopStrategy, speculation *UserTurnSpeculation)
	// SpeculationCanceled withdraws a speculative reply that is no longer the
	// right answer.
	SpeculationCanceled func(ctx context.Context)
	// Resumed fires when the user starts speaking again in an open turn, after
	// they stopped or after the turn's inference was triggered. The new speech
	// voids the answer that inference started, which the params say.
	Resumed func(ctx context.Context, params UserTurnResumedParams)
	// StopTimeout fires when the stop timeout ends a turn its stop strategies
	// did not end.
	StopTimeout      func(ctx context.Context)
	ResetAggregation func(ctx context.Context, s StartStrategy)
	Push             func(ctx context.Context, f frames.Frame, dir processor.Direction)
	Broadcast        func(ctx context.Context, build func() frames.Frame)
}

// userTurnInference is the inference triggered for the stop decisions made so
// far in a user turn.
type userTurnInference int

const (
	inferenceNone userTurnInference = iota
	inferenceTriggered
	inferenceSpeculative
)

// startDecision is a start strategy's decision that the user started.
type startDecision struct {
	strategy StartStrategy
	params   UserTurnStartedParams
}

// stopDecision is a stop strategy's decision that the user is done.
type stopDecision struct {
	strategy StopStrategy
	params   UserTurnStoppedParams
}

// UserTurnController runs the start and stop strategy chains and owns the
// user-turn state machine: the decisions the strategies have made, the turn's
// inference, and a stop-timeout watchdog.
//
// How a turn starts and ends follows the StartMode and StopMode of its
// UserTurnStrategiesParams. With StrategyModeFirst the first strategy to decide
// starts or ends the turn. With StrategyModeAll the turn starts or ends once
// every strategy in the list has decided. Turns start on the first start
// strategy and end once every stop strategy agrees by default. A stop strategy
// that waits for the turn's inference gets its say in either mode: in a
// StrategyModeFirst list, the first stop strategy to decide runs the inference,
// and the turn ends once a strategy waiting for that inference decides. A
// canceled answer voids the stop decisions made on it.
//
// In a StrategyModeAll list the controller starts the turn's inference itself,
// once every stop strategy that listens to the user has decided. A stop
// strategy that waits for the inference instead, such as one reading the LLM's
// turn-completion marker, then decides on that inference. An eager end of turn
// still starts its speculation at once.
//
// A single mutex serializes every state mutation. Process and all strategy
// timer callbacks acquire it, so strategies need no locking of their own.
type UserTurnController struct {
	strategies UserTurnStrategies
	// stopStrategies are the stop strategies the controller runs: the stop
	// chain, with a deferred wrapper replaced by the strategy it wraps, since
	// the controller runs the turn's inference itself.
	stopStrategies []StopStrategy
	params         UserTurnStrategiesParams
	stopTimeout    time.Duration
	hooks          ControllerHooks

	mu  sync.Mutex
	ctx context.Context
	// setup is the pipeline configuration the strategies were set up with, so
	// a strategy added later is given the same one. setupDone records that it
	// was actually handed over, since the zero Setup is a legitimate value.
	setup     processor.Setup
	setupDone bool

	userSpeaking bool
	userTurn     bool
	// startDecisions and stopDecisions are the decisions made so far, in the
	// order the strategies first made them.
	startDecisions []startDecision
	stopDecisions  []stopDecision
	// inference is the inference triggered for the stop decisions made so far.
	inference userTurnInference
	// enableInterruptions is whether the open turn interrupts the bot, as its
	// start decided.
	enableInterruptions bool
	// warnedProcessFrameResult records that a deprecated Stop was honored.
	warnedProcessFrameResult bool
	watchdogCancel           func()
}

// NewUserTurnController builds a controller with the default
// UserTurnStrategiesParams. A zero stopTimeout uses 5s; empty strategy lists
// fall back to the defaults.
func NewUserTurnController(strategies UserTurnStrategies, stopTimeout time.Duration) *UserTurnController {
	return NewUserTurnControllerWithParams(strategies, UserTurnStrategiesParams{}, stopTimeout)
}

// NewUserTurnControllerWithParams builds a controller whose strategies decide
// as params says. A zero stopTimeout uses 5s; empty strategy lists fall back to
// the defaults.
func NewUserTurnControllerWithParams(
	strategies UserTurnStrategies, params UserTurnStrategiesParams, stopTimeout time.Duration,
) *UserTurnController {
	if stopTimeout == 0 {
		stopTimeout = defaultStopTimeout
	}
	strategies.fillDefaults()
	return &UserTurnController{
		strategies:     strategies,
		stopStrategies: stopStrategiesOf(strategies),
		params:         params,
		stopTimeout:    stopTimeout,
	}
}

// stopStrategiesOf is the stop chain the controller runs. The controller runs
// the turn's inference, so a deferred strategy is just the strategy it wraps.
func stopStrategiesOf(strategies UserTurnStrategies) []StopStrategy {
	out := make([]StopStrategy, 0, len(strategies.Stop))
	for _, s := range strategies.Stop {
		if d, ok := s.(*deferredStop); ok {
			s = d.inner
		}
		out = append(out, s)
	}
	return out
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

// StrategiesParams is how the controller's strategies decide.
func (c *UserTurnController) StrategiesParams() UserTurnStrategiesParams {
	return c.params
}

// UpdateStrategies replaces the current strategies with the given ones. The
// chains that go are cleaned up and the ones that arrive are set up with the
// same pipeline configuration the controller was given, so a caller swapping
// them does not have to hand it over again. The decisions the old strategies
// made go with them.
//
// Empty chains fall back to the defaults, as they do at construction.
func (c *UserTurnController) UpdateStrategies(strategies UserTurnStrategies) error {
	c.cleanupStrategies()
	strategies.fillDefaults()
	c.mu.Lock()
	c.strategies = strategies
	c.stopStrategies = stopStrategiesOf(strategies)
	c.startDecisions = nil
	c.stopDecisions = nil
	c.mu.Unlock()
	return c.setupStrategies()
}

// setupStrategies binds every strategy to the shared environment and hands it
// the pipeline configuration.
func (c *UserTurnController) setupStrategies() error {
	c.mu.Lock()
	start, stop, st, done := c.strategies.Start, c.stopStrategies, c.setup, c.setupDone
	c.mu.Unlock()
	if !done {
		return errNotSetUp
	}
	for _, s := range start {
		s.attach(s, c.startEnv())
		if err := s.Setup(st); err != nil {
			return err
		}
	}
	for _, s := range stop {
		s.attach(s, c.stopEnv())
		if err := s.Setup(st); err != nil {
			return err
		}
	}
	warnIfNoStopStrategyListensToTheUser(stop)
	return nil
}

// warnIfNoStopStrategyListensToTheUser warns about a stop chain that only
// waits for the turn's inference. The inference runs once the stop strategies
// that listen to the user have decided, so with none of them only the stop
// timeout runs it.
func warnIfNoStopStrategyListensToTheUser(stop []StopStrategy) {
	if len(stop) == 0 {
		return
	}
	for _, s := range stop {
		if !s.WaitsForInference() {
			return
		}
	}
	slog.Warn("turns: every user turn stop strategy waits for the turn's inference, so " +
		"nothing runs it until the user turn stop timeout; add a stop strategy that " +
		"detects the user stopping, such as SpeechTimeoutStop")
}

// cleanupStrategies releases what every strategy holds.
//
// The mutex is held across the Cleanup calls, because cleaning a strategy up
// stops its timers and a timer is only stopped safely against its own callback
// with the mutex held. The callback takes the mutex and re-reads the cancel
// flag, which is what stops a timer that had already fired from acting on a
// turn that is over. Releasing the mutex first raced the write to that flag
// against the callback's read of it, and left the callback free to run after
// cleanup had returned.
//
// No strategy's Cleanup takes this mutex again, and one holding a lock of its
// own takes that lock underneath this one, so the order here is the order
// everywhere else.
func (c *UserTurnController) cleanupStrategies() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, s := range c.strategies.Start {
		s.Cleanup()
	}
	for _, s := range c.stopStrategies {
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
	for _, s := range c.stopStrategies {
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

// Process taps one frame: it updates the speaking state, re-arms the watchdog,
// and hands the frame to the start then the stop strategies. It holds the mutex
// throughout, so the strategies' synchronous callbacks run safely without
// re-locking.
func (c *UserTurnController) Process(f frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.processLocked(f)
}

// ProcessFromHook is Process for a caller already running inside one of the
// controller's hooks, which hold its mutex. It is how the owner hands the
// strategies the LLMContextFrame it pushes for the turn's inference, so a stop
// strategy that waits for the inference waits for that one.
func (c *UserTurnController) ProcessFromHook(f frames.Frame) { c.processLocked(f) }

// processLocked is Process with the mutex held.
func (c *UserTurnController) processLocked(f frames.Frame) {
	switch f.(type) {
	case *frames.UserStartedSpeakingFrame, *frames.ProposedUserStartedSpeakingFrame,
		*frames.VADUserStartedSpeakingFrame:
		c.handleSpeechStarted()
		// The user started talking, so the stop timeout starts over.
		c.rearmWatchdog()
	case *frames.UserStoppedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame,
		*frames.VADUserStoppedSpeakingFrame:
		c.userSpeaking = false
		c.rearmWatchdog()
		c.endUserTurnIfDecided()
	case *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame:
		c.rearmWatchdog()
	}

	c.processStartStrategies(f)
	c.processStopStrategies(f)

	// Handled once the stop strategies have seen it, so an inference started
	// here is the one they then wait for.
	if _, ok := f.(*frames.UserTurnResponseCancelFrame); ok {
		c.handleUserTurnResponseCancel()
	}
}

// processStartStrategies hands the frame to every start strategy. Every
// strategy in a StrategyModeAll list needs every frame to decide on, so only a
// StrategyModeFirst list honors the deprecated Stop. The last strategy's Stop
// hides the frame from no one.
func (c *UserTurnController) processStartStrategies(f frames.Frame) {
	first := c.params.startMode() == StrategyModeFirst
	for i, s := range c.strategies.Start {
		if s.Process(f) == Stop && first {
			if i < len(c.strategies.Start)-1 {
				c.warnProcessFrameResult()
			}
			return
		}
	}
}

// processStopStrategies is processStartStrategies for the stop strategies.
func (c *UserTurnController) processStopStrategies(f frames.Frame) {
	first := c.params.stopMode() == StrategyModeFirst
	for i, s := range c.stopStrategies {
		if s.Process(f) == Stop && first {
			if i < len(c.stopStrategies)-1 {
				c.warnProcessFrameResult()
			}
			return
		}
	}
}

// warnProcessFrameResult says, once, that a deprecated Stop was honored.
func (c *UserTurnController) warnProcessFrameResult() {
	if c.warnedProcessFrameResult {
		return
	}
	c.warnedProcessFrameResult = true
	slog.Warn("turns: ProcessFrameResult is deprecated and will be removed; use StrategyMode instead")
}

// handleSpeechStarted clears the stop decisions made so far, and the inference
// triggered for them. It resumes the turn when the user had stopped, or when
// that inference was already triggered.
func (c *UserTurnController) handleSpeechStarted() {
	triggered := c.inference != inferenceNone
	resumed := c.userTurn && (!c.userSpeaking || triggered)
	params := UserTurnResumedParams{
		EnableInterruptions: c.enableInterruptions,
		InferenceTriggered:  triggered,
	}
	c.userSpeaking = true
	c.stopDecisions = nil
	c.inference = inferenceNone
	if resumed && c.hooks.Resumed != nil {
		c.hooks.Resumed(c.ctx, params)
	}
}

// startEnv builds the environment shared with start strategies.
func (c *UserTurnController) startEnv() strategyEnv {
	return strategyEnv{
		mu:      &c.mu,
		started: c.onUserTurnStarted,
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
		mu:                  &c.mu,
		inferenceTriggered:  c.onUserTurnInferenceTriggered,
		speculationCanceled: c.onUserTurnSpeculationCanceled,
		stopped:             c.onUserTurnStopped,
		push:                c.pushHook(),
		broadcast:           c.broadcastHook(),
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

// onUserTurnStarted takes a start strategy's report that the user started.
func (c *UserTurnController) onUserTurnStarted(s StartStrategy, params UserTurnStartedParams) {
	if c.params.startMode() == StrategyModeAll {
		c.addUserTurnStartDecision(s, params)
		return
	}
	c.triggerUserTurnStart(s, params)
}

// addUserTurnStartDecision records a start decision, and starts the turn once
// every start strategy has decided.
func (c *UserTurnController) addUserTurnStartDecision(s StartStrategy, params UserTurnStartedParams) {
	if c.userTurn {
		return
	}
	c.startDecisions = setDecision(c.startDecisions, startDecision{s, params},
		func(d startDecision) bool { return d.strategy == s })
	if len(c.startDecisions) == len(c.strategies.Start) {
		c.triggerUserTurnStart(s, c.userTurnStartParams())
	}
}

// userTurnStartParams combines the start decisions: one saying the turn should
// not interrupt, or was announced elsewhere, applies to the whole turn.
func (c *UserTurnController) userTurnStartParams() UserTurnStartedParams {
	params := UserTurnStartedParams{EnableInterruptions: true, EnableUserSpeakingFrames: true}
	for _, d := range c.startDecisions {
		params.EnableInterruptions = params.EnableInterruptions && d.params.EnableInterruptions
		params.EnableUserSpeakingFrames = params.EnableUserSpeakingFrames && d.params.EnableUserSpeakingFrames
	}
	return params
}

// triggerUserTurnStart opens a turn (guarded against double-start), readies
// every strategy for it, and notifies upward.
func (c *UserTurnController) triggerUserTurnStart(s StartStrategy, params UserTurnStartedParams) {
	if c.userTurn {
		return
	}
	c.userTurn = true
	c.startDecisions = nil
	c.rearmWatchdog()
	c.enableInterruptions = params.EnableInterruptions
	c.notifyTurnStarted()
	if c.hooks.Started != nil {
		c.hooks.Started(c.ctx, s, params)
	}
}

// onUserTurnInferenceTriggered takes a stop strategy's speculation. The
// controller runs the turn's inference itself once the stop strategies have
// decided, so a strategy only starts one to speculate, and an ordinary one is
// ignored.
func (c *UserTurnController) onUserTurnInferenceTriggered(s StopStrategy, speculation *UserTurnSpeculation) {
	if speculation == nil {
		return
	}
	c.triggerUserTurnInference(s, speculation)
}

// onUserTurnSpeculationCanceled withdraws a speculative reply. Unlike the other
// decisions it is not gated on a turn being open: a speculation outlives the
// turn that started it when the turn ends without resolving it, and withdrawing
// it then is the whole point. Every strategy learns the speculation is gone, as
// when its service withdraws it.
func (c *UserTurnController) onUserTurnSpeculationCanceled(StopStrategy) {
	if c.hooks.SpeculationCanceled != nil {
		c.hooks.SpeculationCanceled(c.ctx)
	}
	c.processLocked(frames.NewUserTurnResponseCancelFrame())
}

// onUserTurnStopped takes a stop strategy's report that the user is done.
func (c *UserTurnController) onUserTurnStopped(s StopStrategy, params UserTurnStoppedParams) {
	if c.params.stopMode() == StrategyModeAll {
		c.addUserTurnStopDecision(s, params)
		return
	}
	c.firstUserTurnStop(s, params)
}

// handleUserTurnResponseCancel voids what was decided on a canceled answer.
//
// Decisions from strategies that wait for the inference were made on the
// canceled answer, so they no longer hold. The other decisions still say the
// user is done, including one that confirmed the answer, so once they are all
// in, the turn's answer runs again.
func (c *UserTurnController) handleUserTurnResponseCancel() {
	c.inference = inferenceNone
	kept := c.stopDecisions[:0:0]
	for _, d := range c.stopDecisions {
		if d.strategy.WaitsForInference() {
			continue
		}
		d.params.ConfirmsSpeculation = false
		kept = append(kept, d)
	}
	c.stopDecisions = kept
	if len(c.stopDecisions) > 0 && c.readyForInference() {
		c.triggerUserTurnInference(c.stopDecisions[0].strategy, nil)
	}
}

// triggerUserTurnInference runs the turn's inference, or a speculation on it.
// It fires only while a turn is open, and the turn stays open afterward: only
// the turn stopping flips that.
func (c *UserTurnController) triggerUserTurnInference(s StopStrategy, speculation *UserTurnSpeculation) {
	if !c.userTurn {
		return
	}
	c.inference = inferenceTriggered
	if speculation != nil {
		c.inference = inferenceSpeculative
	}
	// Re-arm the watchdog, so a stuck turn (the inference ran but nothing
	// decided on it) still times out and ends.
	c.rearmWatchdog()
	if c.hooks.InferenceTriggered != nil {
		c.hooks.InferenceTriggered(c.ctx, s, speculation)
	}
}

// firstUserTurnStop takes a stop decision in a StrategyModeFirst list.
//
// The first decision runs the turn's inference and ends the turn. A stop
// strategy that waits for that inference still gets its say: the turn then
// ends on its decision instead, keeping what the earlier decisions said about
// the turn end.
func (c *UserTurnController) firstUserTurnStop(s StopStrategy, params UserTurnStoppedParams) {
	if !c.userTurn {
		return
	}
	c.setStopDecision(s, params)
	if !s.WaitsForInference() && c.inference == inferenceNone {
		c.triggerUserTurnInference(s, nil)
	}
	if c.userTurnStopDecided() {
		c.triggerUserTurnStop(s, c.userTurnStopParams())
	}
}

// addUserTurnStopDecision takes a stop decision in a StrategyModeAll list.
//
// A decision counts while the user speaks, as when a service still hears them
// while an eager answer runs. Only the turn end waits for them to stop, and new
// speech clears the decisions.
func (c *UserTurnController) addUserTurnStopDecision(s StopStrategy, params UserTurnStoppedParams) {
	if !c.userTurn {
		return
	}
	c.setStopDecision(s, params)
	// The eager strategy confirms its speculation when the transcript commits.
	// If the speculation was already withdrawn, for example because the LLM
	// wrote an incomplete marker, there is nothing left to confirm, so it is
	// handled like the withdrawal: the decision becomes a plain one and the
	// committed turn gets its own inference.
	if params.ConfirmsSpeculation && c.inference != inferenceSpeculative {
		c.handleUserTurnResponseCancel()
	}
	// Once every stop strategy that listens to the user has decided, run the
	// turn's inference. This comes before the turn-end check below, so when
	// this decision also ends the turn, the inference still starts first.
	if c.inference == inferenceNone && c.readyForInference() {
		c.triggerUserTurnInference(s, nil)
	}
	if c.userTurnStopDecided() {
		c.triggerUserTurnStop(s, c.userTurnStopParams())
	}
}

// setStopDecision records a stop decision, keeping a strategy's place in the
// order when it decides again.
func (c *UserTurnController) setStopDecision(s StopStrategy, params UserTurnStoppedParams) {
	c.stopDecisions = setDecision(c.stopDecisions, stopDecision{s, params},
		func(d stopDecision) bool { return d.strategy == s })
}

// hasStopDecision reports whether s has decided.
func (c *UserTurnController) hasStopDecision(s StopStrategy) bool {
	for _, d := range c.stopDecisions {
		if d.strategy == s {
			return true
		}
	}
	return false
}

// userTurnStopDecided reports whether the decisions made end the turn. A
// StrategyModeAll list needs every stop strategy. A StrategyModeFirst list ends
// the turn on its first decision or, when a stop strategy waits for the
// inference, on that strategy's decision.
func (c *UserTurnController) userTurnStopDecided() bool {
	if c.params.stopMode() == StrategyModeAll {
		return len(c.stopDecisions) == len(c.stopStrategies)
	}
	waiting := false
	for _, s := range c.stopStrategies {
		if !s.WaitsForInference() {
			continue
		}
		waiting = true
		if c.hasStopDecision(s) {
			return true
		}
	}
	return !waiting && len(c.stopDecisions) > 0
}

// endUserTurnIfDecided ends a turn decided while the user was speaking, once
// they stop. New speech clears the decisions, so they are about the speech just
// heard.
func (c *UserTurnController) endUserTurnIfDecided() {
	if len(c.stopDecisions) > 0 && c.userTurnStopDecided() {
		last := c.stopDecisions[len(c.stopDecisions)-1].strategy
		c.triggerUserTurnStop(last, c.userTurnStopParams())
	}
}

// readyForInference reports whether only the stop strategies that wait for the
// inference are left to decide.
func (c *UserTurnController) readyForInference() bool {
	for _, s := range c.stopStrategies {
		if !s.WaitsForInference() && !c.hasStopDecision(s) {
			return false
		}
	}
	return true
}

// userTurnStopParams combines the stop decisions: one saying the turn end was
// announced elsewhere, or that it confirms a speculation, applies to the whole
// turn.
func (c *UserTurnController) userTurnStopParams() UserTurnStoppedParams {
	params := UserTurnStoppedParams{EnableUserSpeakingFrames: true}
	for _, d := range c.stopDecisions {
		params.EnableUserSpeakingFrames = params.EnableUserSpeakingFrames && d.params.EnableUserSpeakingFrames
		params.ConfirmsSpeculation = params.ConfirmsSpeculation || d.params.ConfirmsSpeculation
	}
	return params
}

// triggerUserTurnStop closes a turn (guarded against double-stop), resets the
// strategies, and notifies upward.
func (c *UserTurnController) triggerUserTurnStop(s StopStrategy, params UserTurnStoppedParams) {
	if !c.userTurn {
		return
	}
	// Never end the turn while the user is audibly speaking. The decisions stay,
	// and the turn ends once the user stops.
	if c.userSpeaking {
		return
	}
	c.userTurn = false
	c.stopDecisions = nil
	c.inference = inferenceNone
	c.rearmWatchdog()
	c.notifyTurnStopped()
	if c.hooks.Stopped != nil {
		c.hooks.Stopped(c.ctx, s, params)
	}
}

// notifyTurnStarted readies every strategy for the turn now beginning. Start
// strategies ready themselves for the next detection, and stop strategies arm
// to detect this turn's end.
func (c *UserTurnController) notifyTurnStarted() {
	for _, s := range c.strategies.Start {
		s.TurnStarted()
	}
	for _, s := range c.stopStrategies {
		s.TurnStarted()
	}
}

// notifyTurnStopped tells every strategy the turn ended, whichever strategy (or
// the stop timeout) ended it.
func (c *UserTurnController) notifyTurnStopped() {
	for _, s := range c.strategies.Start {
		s.TurnStopped()
	}
	for _, s := range c.stopStrategies {
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

// onWatchdog ends a turn that has gone a whole stop timeout without user
// activity. A stop strategy with a decision of its own scheduled ends the turn
// when it fires, so the timeout waits for it, and goes on watching the turn
// while it stays open.
func (c *UserTurnController) onWatchdog() {
	if !c.userTurn {
		return
	}
	if !c.userSpeaking && !c.decisionScheduled() {
		if c.hooks.StopTimeout != nil {
			c.hooks.StopTimeout(c.ctx)
		}
		// The turn ends with what the decisions made so far said about it, so a
		// confirmed speculation is not answered a second time. Nothing decided
		// the end itself, so there is no strategy to attribute it to.
		c.triggerUserTurnStop(nil, c.userTurnStopParams())
		return
	}
	c.rearmWatchdog()
}

// decisionScheduled reports whether a stop strategy has a decision of its own
// scheduled.
func (c *UserTurnController) decisionScheduled() bool {
	for _, s := range c.stopStrategies {
		if s.DecisionScheduled() {
			return true
		}
	}
	return false
}

// setDecision puts d in decisions, in place of the one is matches, or at the
// end when there is none.
func setDecision[D any](decisions []D, d D, is func(D) bool) []D {
	for i := range decisions {
		if is(decisions[i]) {
			decisions[i] = d
			return decisions
		}
	}
	return append(decisions, d)
}

// startMode is the start mode in force, the default resolved.
func (p UserTurnStrategiesParams) startMode() StrategyMode {
	if p.StartMode == StrategyModeDefault {
		return StrategyModeFirst
	}
	return p.StartMode
}

// stopMode is the stop mode in force, the default resolved.
func (p UserTurnStrategiesParams) stopMode() StrategyMode {
	if p.StopMode == StrategyModeDefault {
		return StrategyModeAll
	}
	return p.StopMode
}
