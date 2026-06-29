// Package turns manages the user-turn lifecycle, ported from Pipecat's turns
// subsystem. A UserTurnProcessor drives a UserTurnController (which runs
// pluggable start and stop strategies) and a UserIdleController. Turn detection
// is decoupled: voice activity comes from a vad.Processor upstream as
// VADUser*SpeakingFrames, and the end-of-turn model lives inside a stop
// strategy; the subsystem reasons over frames, not raw audio (except the
// turn-analyzer stop strategy, which is fed InputAudioRawFrames).
package turns

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// ProcessFrameResult is what a strategy returns from Process to control the
// per-frame strategy loop.
type ProcessFrameResult int

const (
	// Continue evaluates the next strategy in the chain.
	Continue ProcessFrameResult = iota
	// Stop short-circuits the remaining strategies for this frame.
	Stop
)

// UserTurnStartedParams describes how a start strategy wants a turn opened.
type UserTurnStartedParams struct {
	// EnableInterruptions broadcasts an InterruptionFrame so the bot is barged
	// in on turn start.
	EnableInterruptions bool
	// EnableUserSpeakingFrames broadcasts a UserStartedSpeakingFrame on turn
	// start. External integrations disable this when they emit it themselves.
	EnableUserSpeakingFrames bool
}

// DefaultStartedParams is the params a typical start strategy uses.
func DefaultStartedParams() UserTurnStartedParams {
	return UserTurnStartedParams{EnableInterruptions: true, EnableUserSpeakingFrames: true}
}

// UserTurnStoppedParams describes how a stop strategy wants a turn closed.
type UserTurnStoppedParams struct {
	// EnableUserSpeakingFrames broadcasts a UserStoppedSpeakingFrame on turn
	// stop.
	EnableUserSpeakingFrames bool
}

// DefaultStoppedParams is the params a typical stop strategy uses.
func DefaultStoppedParams() UserTurnStoppedParams {
	return UserTurnStoppedParams{EnableUserSpeakingFrames: true}
}

// Emitter lets a controller or strategy push frames into the pipeline. The
// UserTurnProcessor implements it. Broadcast sends a frame both downstream and
// upstream (the analog of Pipecat's broadcast_frame), which is how turn
// decisions and interruptions reach the whole pipeline.
type Emitter interface {
	// Push sends a frame to the neighbor in dir.
	Push(ctx context.Context, f frames.Frame, dir processor.Direction) error
	// Broadcast sends a frame both downstream and upstream.
	Broadcast(ctx context.Context, f frames.Frame) error
}
