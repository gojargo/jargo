package flows

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// The built-in action types.
const (
	// actionTTSSay speaks a fixed line.
	actionTTSSay = "tts_say"
	// actionEndConversation ends the conversation, optionally speaking a goodbye
	// first.
	actionEndConversation = "end_conversation"
	// actionFunction runs a handler in pipeline order, once everything queued
	// ahead of it has been processed.
	actionFunction = "function"
)

// FunctionActionFrame carries a function action to run once it reaches the end
// of the pipeline. Queueing it rather than running the handler on the spot is
// what makes the handler happen at the right moment: after the speech queued
// ahead of it has been said, rather than while it is still being synthesized.
type FunctionActionFrame struct {
	frames.BaseControlFrame
	// Action is the action that named the handler.
	Action ActionConfig
	// Function is the handler to run.
	Function ActionHandler
}

// NewFunctionActionFrame builds a FunctionActionFrame.
func NewFunctionActionFrame(action ActionConfig, fn ActionHandler) *FunctionActionFrame {
	return &FunctionActionFrame{
		BaseControlFrame: frames.NewBaseControlFrame("FunctionActionFrame"),
		Action:           action,
		Function:         fn,
	}
}

// ActionFinishedFrame marks an action as complete. An action that queues frames
// queues one of these behind them, so the action counts as finished when its
// frames have been processed rather than when it returned.
type ActionFinishedFrame struct {
	frames.BaseControlFrame
}

// NewActionFinishedFrame builds an ActionFinishedFrame.
func NewActionFinishedFrame() *ActionFinishedFrame {
	return &ActionFinishedFrame{BaseControlFrame: frames.NewBaseControlFrame("ActionFinishedFrame")}
}

// Watcher reports frames that reach the end of the pipeline. *pipeline.Task
// satisfies it. The action manager uses it to learn when its actions have
// finished and when the assistant's turn is over, and says which frames it
// wants to hear about.
type Watcher interface {
	OnReachedDownstream(fn func(frames.Frame))
	SetReachedDownstreamFilter(f pipeline.FrameFilter)
}

// actionManager registers and runs a flow's actions.
//
// Actions run on the way into and out of a node, and are how a flow does
// something other than talk: speaking a fixed line, ending the call, or running
// a handler at a particular point in the pipeline. Three are built in
// ("tts_say", "end_conversation" and "function") and an application registers
// whatever else it needs.
type actionManager struct {
	enq   Enqueuer
	fm    *FlowManager
	mu    sync.Mutex
	cond  *sync.Cond
	names map[string]ActionHandler
	// ongoing counts the actions that have been started and not yet finished.
	// Waiting for it to reach zero is how one action waits for the last.
	ongoing int
	// deferred are the post-actions of a node that waits for the user, held back
	// until the assistant's first turn at that node is over.
	deferred []ActionConfig
}

// newActionManager builds an action manager, registers the built-in actions and
// starts watching the end of the pipeline.
func newActionManager(enq Enqueuer, watch Watcher, fm *FlowManager) *actionManager {
	am := &actionManager{
		enq:   enq,
		fm:    fm,
		names: make(map[string]ActionHandler, 3),
	}
	am.cond = sync.NewCond(&am.mu)

	// register only fails on a nil handler, and these three are method values
	// on am, so none of them can be.
	_ = am.register(actionTTSSay, am.handleTTSSay)
	_ = am.register(actionEndConversation, am.handleEndConversation)
	_ = am.register(actionFunction, am.handleFunction)

	if watch != nil {
		watch.SetReachedDownstreamFilter(pipeline.FrameTypes(
			&ActionFinishedFrame{},
			&FunctionActionFrame{},
			&frames.BotStoppedSpeakingFrame{},
		))
		watch.OnReachedDownstream(am.frameReachedDownstream)
	}
	return am
}

// frameReachedDownstream is what the end of the pipeline reports to. It is where
// an action that queued frames is finally counted as finished, and where the
// post-actions a node deferred are run.
func (am *actionManager) frameReachedDownstream(f frames.Frame) {
	switch fr := f.(type) {
	case *FunctionActionFrame:
		// The handler runs here, at the point in the pipeline the frame reached,
		// which is the whole reason it was queued rather than called directly.
		if err := fr.Function(context.Background(), fr.Action, am.fm); err != nil {
			slog.Error("flows: function action failed", "err", err)
		}
		am.finishAction()
	case *frames.BotStoppedSpeakingFrame:
		// The assistant's turn is over, so a node that was waiting to run its
		// post-actions may run them now. Only when nothing else is in flight: an
		// action of our own may be what made the bot speak, and its turn ending
		// says nothing about the node's first response having happened.
		am.mu.Lock()
		idle := am.ongoing == 0
		am.mu.Unlock()
		if idle {
			am.runDeferredPostActions()
		}
	case *ActionFinishedFrame:
		am.finishAction()
	}
}

// register records a handler for an action type.
func (am *actionManager) register(actionType string, h ActionHandler) error {
	if h == nil {
		return errf(ErrAction, fmt.Sprintf("handler for %q must not be nil", actionType))
	}
	am.mu.Lock()
	am.names[actionType] = h
	am.mu.Unlock()
	slog.Debug("flows: registered action handler", "type", actionType)
	return nil
}

// registered reports the handler for an action type.
func (am *actionManager) registered(actionType string) (ActionHandler, bool) {
	am.mu.Lock()
	defer am.mu.Unlock()
	h, ok := am.names[actionType]
	return h, ok
}

// registerFromConfig registers the handler an action config carries, for a type
// nothing has registered yet. An action that names an unregistered type and
// brings no handler is a mistake worth reporting rather than skipping, since the
// flow expects it to run.
func (am *actionManager) registerFromConfig(a ActionConfig) error {
	if a.Type == "" {
		return errf(ErrAction, "action is missing its type")
	}
	if _, ok := am.registered(a.Type); ok {
		return nil
	}
	if a.Handler == nil {
		return errf(ErrAction, fmt.Sprintf(
			"action %q is not registered: give it a handler or register one", a.Type))
	}
	return am.register(a.Type, a.Handler)
}

// executeActions runs a list of actions in order.
//
// Between one action and the next it waits where waiting is what keeps them in
// order: see waitForOngoing. An action that fails stops the list, since the
// actions after it were written expecting it to have happened.
func (am *actionManager) executeActions(ctx context.Context, actions []ActionConfig) error {
	if len(actions) == 0 {
		return nil
	}

	previous := ""
	for _, a := range actions {
		if a.Type == "" {
			return errf(ErrAction, "action is missing its type")
		}
		h, ok := am.registered(a.Type)
		if !ok {
			return errf(ErrAction, fmt.Sprintf("no handler registered for action %q", a.Type))
		}

		am.mu.Lock()
		before := am.ongoing
		am.mu.Unlock()

		am.waitForOngoing(previous, a.Type)

		if err := h(ctx, a, am.fm); err != nil {
			// Undo the start this action counted, so a failure does not leave the
			// count showing work that will never finish and block the next wait.
			am.mu.Lock()
			started := am.ongoing > before
			am.mu.Unlock()
			if started {
				am.finishAction()
			}
			return errf(ErrAction, fmt.Sprintf("action %q failed: %v", a.Type, err))
		}

		previous = a.Type
		slog.DebugContext(ctx, "flows: executed action", "type", a.Type)

		if a.Type == actionEndConversation {
			// Nothing after this can run: the pipeline is ending, so the frames a
			// later action queued would never be delivered and the wait for them
			// would never end.
			break
		}
	}

	// The last action may still be in flight, and what comes next is usually the
	// node's own context update, which takes effect earlier in the pipeline.
	am.waitForOngoing(previous, "")
	return nil
}

// waitForOngoing waits for the actions already started to finish, when going
// straight on to the next one would let it take effect first.
//
// It turns on where in the pipeline each action has its effect. An action that
// queues a frame has its effect where that frame is handled, so two actions
// whose frames are handled at the same point or later need no wait between
// them: the pipeline already orders them. A wait is needed when what comes next
// takes effect earlier, which is where the ordering would otherwise invert.
func (am *actionManager) waitForOngoing(previous, upcoming string) {
	wait := false
	switch previous {
	case actionTTSSay:
		// Speech takes effect at the TTS service. Another spoken action, the end of
		// the conversation and a queued function all take effect there or later, so
		// they need no wait. Anything else does: the end of a list of actions is
		// followed by the node's context update, which takes effect earlier, and a
		// custom action could do anything.
		switch upcoming {
		case actionTTSSay, actionEndConversation, actionFunction:
		default:
			wait = true
		}
	case actionFunction:
		// A function takes as long as it takes and does not hold up the pipeline
		// while it runs, so the next thing always waits for it.
		wait = true
	default:
		// Nothing started yet, or a custom action, which is not waited on: what it
		// does is its own business and there is no frame of ours to wait for.
	}
	if !wait {
		return
	}

	am.mu.Lock()
	for am.ongoing > 0 {
		am.cond.Wait()
	}
	am.mu.Unlock()
}

// scheduleDeferredPostActions holds a node's post-actions back until the
// assistant's first turn at that node is over.
func (am *actionManager) scheduleDeferredPostActions(actions []ActionConfig) {
	am.mu.Lock()
	am.deferred = actions
	am.mu.Unlock()
}

// clearDeferredPostActions drops any post-actions still waiting, which is what
// leaving a node does: they belonged to the node being left.
func (am *actionManager) clearDeferredPostActions() {
	am.mu.Lock()
	am.deferred = nil
	am.mu.Unlock()
}

// runDeferredPostActions runs the post-actions a node deferred, once.
func (am *actionManager) runDeferredPostActions() {
	am.mu.Lock()
	actions := am.deferred
	am.deferred = nil
	am.mu.Unlock()
	if len(actions) == 0 {
		return
	}
	if err := am.executeActions(context.Background(), actions); err != nil {
		slog.Error("flows: deferred post-actions failed", "err", err)
	}
}

// startAction counts an action as started.
func (am *actionManager) startAction() {
	am.mu.Lock()
	am.ongoing++
	am.mu.Unlock()
}

// finishAction counts an action as finished and wakes whoever is waiting for the
// last of them.
func (am *actionManager) finishAction() {
	am.mu.Lock()
	if am.ongoing > 0 {
		am.ongoing--
	}
	if am.ongoing == 0 {
		am.cond.Broadcast()
	}
	am.mu.Unlock()
}

// handleTTSSay speaks the action's text.
func (am *actionManager) handleTTSSay(ctx context.Context, a ActionConfig, _ *FlowManager) error {
	if a.Text == "" {
		slog.ErrorContext(ctx, "flows: tts_say action has no text")
		return nil
	}

	am.startAction()
	speak := frames.NewTTSSpeakFrame(a.Text)
	speak.AppendToContext = a.appendTextToContext()
	am.enq.QueueFrame(speak)
	// Queued behind the speech, so the action counts as finished once the speech
	// has been handled rather than once it was asked for.
	am.enq.QueueFrame(NewActionFinishedFrame())
	return nil
}

// handleEndConversation ends the conversation, speaking a goodbye first when the
// action carries one.
func (am *actionManager) handleEndConversation(_ context.Context, a ActionConfig, _ *FlowManager) error {
	am.startAction()
	if a.Text != "" {
		speak := frames.NewTTSSpeakFrame(a.Text)
		speak.AppendToContext = a.appendTextToContext()
		am.enq.QueueFrame(speak)
	}
	am.enq.QueueFrame(frames.NewEndFrame())
	// No ActionFinishedFrame: the EndFrame queued ahead of it stops the pipeline,
	// so it would never arrive to be counted.
	return nil
}

// handleFunction queues the action's handler to run in pipeline order.
func (am *actionManager) handleFunction(ctx context.Context, a ActionConfig, _ *FlowManager) error {
	if a.Handler == nil {
		slog.ErrorContext(ctx, "flows: function action has no handler")
		return nil
	}
	am.startAction()
	// Queued rather than run here so it happens at the right point in the
	// pipeline, once the work queued ahead of it is done. It is counted as
	// finished when the handler returns, not when the frame is delivered, which is
	// why no ActionFinishedFrame goes with it.
	am.enq.QueueFrame(NewFunctionActionFrame(a, a.Handler))
	return nil
}
