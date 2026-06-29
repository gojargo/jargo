package turns

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Config configures a UserTurnProcessor.
type Config struct {
	// Strategies are the start/stop strategy chains; the zero value uses the
	// defaults (VAD + transcription start, Smart-Turn stop) — but the Smart-Turn
	// default needs a turn.Analyzer, so most callers build Strategies explicitly.
	Strategies UserTurnStrategies
	// StopTimeout is the watchdog that force-stops a stuck turn; 0 uses 5s.
	StopTimeout time.Duration
	// IdleTimeout enables the idle watchdog; a value <= 0 disables it.
	IdleTimeout time.Duration
	// OnIdle fires when the conversation goes idle. Required to enable idle.
	OnIdle IdleCallback
	// MuteStrategies suppress user input while engaged (e.g. while the bot
	// speaks or a tool call runs). They are OR-reduced; empty means never mute.
	MuteStrategies []MuteStrategy
}

// UserTurnProcessor is the pipeline node that drives the turn and idle
// controllers. It is transparent — every frame is forwarded unchanged and also
// tapped to the controllers — and emits its turn decisions
// (UserStarted/StoppedSpeakingFrame) and interruptions via broadcast. Place it
// downstream of the VAD processor, STT, and the LLM/TTS services so it sees the
// speaking, transcription, response and tool-call frames that define a turn.
type UserTurnProcessor struct {
	*processor.Base
	turn *UserTurnController
	idle *UserIdleController

	muteStrategies []MuteStrategy
	muteMu         sync.Mutex
	muted          bool
}

// NewUserTurnProcessor builds a UserTurnProcessor.
func NewUserTurnProcessor(cfg Config) *UserTurnProcessor {
	p := &UserTurnProcessor{muteStrategies: cfg.MuteStrategies}
	p.Base = processor.New("UserTurn", p)
	p.turn = NewUserTurnController(cfg.Strategies, cfg.StopTimeout)
	p.idle = NewUserIdleController(IdleConfig{Timeout: cfg.IdleTimeout, Callback: cfg.OnIdle})
	p.turn.SetHooks(ControllerHooks{
		Started:   p.onTurnStarted,
		Stopped:   p.onTurnStopped,
		Push:      func(ctx context.Context, f frames.Frame, dir processor.Direction) { _ = p.PushFrame(ctx, f, dir) },
		Broadcast: func(ctx context.Context, f frames.Frame) { _ = p.Broadcast(ctx, f) },
	})
	return p
}

// Setup wires the controllers.
func (p *UserTurnProcessor) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	p.turn.Setup(ctx)
	p.idle.Setup(ctx, p)
	return nil
}

// Cleanup tears down the controllers.
func (p *UserTurnProcessor) Cleanup(ctx context.Context) error {
	p.turn.Cleanup()
	p.idle.Cleanup()
	return p.Base.Cleanup(ctx)
}

// ProcessFrame forwards every frame downstream and taps it to both controllers.
func (p *UserTurnProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch f.(type) {
	case *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame:
		// Lifecycle frames are never muted and must keep their ordering.
	default:
		if p.suppressed(ctx, f) {
			return nil
		}
	}
	if err := p.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	p.turn.Process(f)
	p.idle.Process(f)
	return nil
}

// suppressed runs the mute strategies and reports whether this user-input frame
// should be dropped. It emits UserMute frames on a change of state.
func (p *UserTurnProcessor) suppressed(ctx context.Context, f frames.Frame) bool {
	if len(p.muteStrategies) == 0 {
		return false
	}
	p.muteMu.Lock()
	defer p.muteMu.Unlock()

	should := false
	for _, m := range p.muteStrategies {
		if m.ShouldMute(f) { // call all, so each updates its state
			should = true
		}
	}
	if should != p.muted {
		p.muted = should
		if should {
			_ = p.Broadcast(ctx, frames.NewUserMuteStartedFrame())
		} else {
			_ = p.Broadcast(ctx, frames.NewUserMuteStoppedFrame())
		}
	}
	if !p.muted {
		return false
	}
	switch f.(type) {
	case *frames.InterruptionFrame, *frames.VADUserStartedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame,
		*frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame, *frames.InputAudioRawFrame,
		*frames.InterimTranscriptionFrame, *frames.TranscriptionFrame:
		return true
	}
	return false
}

// Push implements Emitter.
func (p *UserTurnProcessor) Push(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	return p.PushFrame(ctx, f, dir)
}

// Broadcast implements Emitter, sending a frame both downstream and upstream so
// turn decisions reach the whole pipeline.
func (p *UserTurnProcessor) Broadcast(ctx context.Context, f frames.Frame) error {
	if err := p.PushFrame(ctx, f, processor.Downstream); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, processor.Upstream)
}

// onTurnStarted broadcasts the turn-start decision and barges in, and feeds the
// idle controller a synthetic user-started frame so it tracks the turn.
func (p *UserTurnProcessor) onTurnStarted(ctx context.Context, params UserTurnStartedParams) {
	if params.EnableUserSpeakingFrames {
		_ = p.Broadcast(ctx, frames.NewUserStartedSpeakingFrame())
	}
	p.idle.Process(frames.NewUserStartedSpeakingFrame())
	if params.EnableInterruptions {
		_ = p.Broadcast(ctx, frames.NewInterruptionFrame())
	}
}

// onTurnStopped broadcasts the turn-stop decision and feeds the idle controller
// a synthetic user-stopped frame.
func (p *UserTurnProcessor) onTurnStopped(ctx context.Context, params UserTurnStoppedParams) {
	if params.EnableUserSpeakingFrames {
		_ = p.Broadcast(ctx, frames.NewUserStoppedSpeakingFrame())
	}
	p.idle.Process(frames.NewUserStoppedSpeakingFrame())
}

// Compile-time check that the processor is an Emitter.
var _ Emitter = (*UserTurnProcessor)(nil)
