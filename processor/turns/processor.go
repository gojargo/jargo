package turns

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// The events a UserTurnProcessor raises around each turn it decides.
const (
	// EventUserTurnStarted fires when a user turn starts, carrying the
	// StartStrategy that decided it began.
	//
	//	events.On(p.Events(), turns.EventUserTurnStarted,
	//	    func(ctx context.Context, s turns.StartStrategy) { … })
	EventUserTurnStarted = "on_user_turn_started"
	// EventUserTurnInferenceTriggered fires when there is enough signal to start
	// LLM inference, carrying the StopStrategy that decided. It fires together
	// with EventUserTurnStopped for most strategies, and alone when a strategy
	// further down the chain gates finalization on someone else's verdict.
	EventUserTurnInferenceTriggered = "on_user_turn_inference_triggered"
	// EventUserTurnStopped fires when a user turn is semantically final,
	// carrying the StopStrategy that decided. It is nil when nothing decided:
	// the turn was closed because no strategy did.
	EventUserTurnStopped = "on_user_turn_stopped"
	// EventUserTurnStopTimeout fires when no stop strategy triggered before the
	// watchdog gave up. It carries no argument.
	EventUserTurnStopTimeout = "on_user_turn_stop_timeout"
	// EventUserTurnIdle fires when the user has been idle for the configured
	// timeout. It carries no argument.
	EventUserTurnIdle = "on_user_turn_idle"
)

// UserTurnProcessor manages the user-turn lifecycle as a processor of its own.
//
// It drives a UserTurnController, which runs the configured start and stop
// strategies, and a UserIdleController. What reaches the pipeline (the user
// speaking frames, the interruption) depends on which strategy decided and what
// it asked for.
//
// Use it where the turn decision has to be made once and shared: several
// aggregators fanning off one turn, or a pipeline that wants the decision at a
// particular point rather than inside the aggregator. Where one aggregator owns
// the turn, configure the strategies on it instead with
// aggregators.WithTurns, which keeps the decision on the same frames as the
// aggregation.
type UserTurnProcessor struct {
	*processor.Base
	turn *UserTurnController
	idle *UserIdleController
}

// NewUserTurnProcessor builds a turn processor from the given configuration.
func NewUserTurnProcessor(cfg Config) *UserTurnProcessor {
	p := &UserTurnProcessor{}
	p.Base = processor.New("UserTurnProcessor", p)
	for _, name := range []string{
		EventUserTurnStarted, EventUserTurnInferenceTriggered, EventUserTurnStopped,
		EventUserTurnStopTimeout, EventUserTurnIdle,
	} {
		// Asynchronous: a handler may do anything, and the turn must not wait on
		// whatever is listening to it.
		p.Events().Register(name, false)
	}

	p.turn = NewUserTurnController(cfg.Strategies, cfg.StopTimeout)
	onIdle := func(ctx context.Context, c *UserIdleController) error {
		p.Events().Call(ctx, EventUserTurnIdle, p)
		if cfg.OnIdle == nil {
			return nil
		}
		return cfg.OnIdle(ctx, c)
	}
	p.idle = NewUserIdleController(IdleConfig{Timeout: cfg.IdleTimeout, Callback: onIdle})
	p.turn.SetHooks(ControllerHooks{
		Started:            p.onTurnStarted,
		Stopped:            p.onTurnStopped,
		InferenceTriggered: p.onInferenceTriggered,
		StopTimeout:        p.onStopTimeout,
		Push: func(ctx context.Context, f frames.Frame, dir processor.Direction) {
			_ = p.PushFrame(ctx, f, dir)
		},
		Broadcast: func(ctx context.Context, build func() frames.Frame) {
			_ = p.Broadcast(ctx, build)
		},
	})
	return p
}

// Controller is the turn controller this processor drives, so a caller can read
// the strategies in force or replace them.
func (p *UserTurnProcessor) Controller() *UserTurnController { return p.turn }

// Setup wires the controllers.
func (p *UserTurnProcessor) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	p.idle.Setup(ctx, p)
	return p.turn.Setup(ctx, s)
}

// Cleanup releases what the controllers hold.
func (p *UserTurnProcessor) Cleanup(ctx context.Context) error {
	p.turn.Cleanup()
	p.idle.Cleanup()
	return p.Base.Cleanup(ctx)
}

// ProcessFrame gives each frame to the controllers and forwards it.
//
// The frame is forwarded before the controllers see it, so the decisions they
// raise are queued behind the frame that caused them rather than ahead of it.
func (p *UserTurnProcessor) ProcessFrame(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := p.forward(ctx, f, dir); err != nil {
		return err
	}
	p.turn.Process(f)
	p.idle.Process(f)
	return nil
}

// forward sends the frame on, except a proposal this processor's own strategies
// resolve: a proposal is resolved once, and forwarding one would let a resolver
// further down the pipeline decide the same turn a second time.
func (p *UserTurnProcessor) forward(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	switch f.(type) {
	case *frames.ProposedUserStartedSpeakingFrame:
		if p.turn.ResolvesProposedTurnStartFrames() {
			return nil
		}
	case *frames.ProposedUserStoppedSpeakingFrame:
		if p.turn.ResolvesProposedTurnStopFrames() {
			return nil
		}
	case *frames.EndFrame, *frames.CancelFrame:
		// The session is over, so the controllers stop here rather than waiting
		// for teardown. The frame is forwarded first so nothing is held up
		// behind them.
		if err := p.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		p.turn.Cleanup()
		p.idle.Cleanup()
		return nil
	}
	return p.PushFrame(ctx, f, dir)
}

// Push implements Emitter, so the idle controller's callback can reach the
// pipeline.
func (p *UserTurnProcessor) Push(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	return p.PushFrame(ctx, f, dir)
}

// Broadcast, which Emitter also requires, is promoted from processor.Base.

// onTurnStarted announces the turn the strategies opened.
func (p *UserTurnProcessor) onTurnStarted(
	ctx context.Context, strategy StartStrategy, params UserTurnStartedParams,
) {
	slog.Debug("user started speaking", "processor", p.Name(), "strategy", strategyName(strategy))
	if params.EnableUserSpeakingFrames {
		_ = p.Broadcast(ctx, func() frames.Frame { return frames.NewUserStartedSpeakingFrame() })
	}
	// The idle watchdog tracks the turn whether or not the frame was emitted.
	p.idle.Process(frames.NewUserStartedSpeakingFrame())
	if params.EnableInterruptions {
		_ = p.BroadcastInterruption(ctx)
	}
	p.Events().Call(ctx, EventUserTurnStarted, p, strategy)
}

// onTurnStopped announces the turn the strategies closed.
func (p *UserTurnProcessor) onTurnStopped(
	ctx context.Context, strategy StopStrategy, params UserTurnStoppedParams,
) {
	slog.Debug("user stopped speaking", "processor", p.Name(), "strategy", strategyName(strategy))
	if params.EnableUserSpeakingFrames {
		_ = p.Broadcast(ctx, func() frames.Frame { return frames.NewUserStoppedSpeakingFrame() })
	}
	p.idle.Process(frames.NewUserStoppedSpeakingFrame())
	p.Events().Call(ctx, EventUserTurnStopped, p, strategy)
}

// onInferenceTriggered reports that there is enough to answer on.
func (p *UserTurnProcessor) onInferenceTriggered(ctx context.Context, strategy StopStrategy) {
	slog.Debug("user turn inference triggered",
		"processor", p.Name(), "strategy", strategyName(strategy))
	p.Events().Call(ctx, EventUserTurnInferenceTriggered, p, strategy)
}

// onStopTimeout reports a turn closed because no strategy closed it.
func (p *UserTurnProcessor) onStopTimeout(ctx context.Context) {
	p.Events().Call(ctx, EventUserTurnStopTimeout, p)
}

// strategyName names a strategy for a log line, and says so when nothing
// decided.
func strategyName(s any) string {
	if s == nil {
		return "none"
	}
	return fmt.Sprintf("%T", s)
}

var _ Emitter = (*UserTurnProcessor)(nil)
