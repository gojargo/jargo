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

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// ProcessFrameResult is what a strategy returns from Process.
//
// The result is deprecated in favor of StrategyMode: strategies return
// Continue, and every strategy sees every frame. A list whose strategies all
// have to agree, such as a wake phrase next to VAD, is StrategyModeAll. The type
// stays in the strategy interfaces until the result is removed.
type ProcessFrameResult int

const (
	// Continue evaluates the next strategy in the chain. It is what every
	// strategy returns.
	Continue ProcessFrameResult = iota
	// Stop short-circuits the remaining strategies for this frame, and only in
	// a StrategyModeFirst list, with a deprecation warning.
	//
	// Deprecated: use StrategyMode instead. Every strategy sees every frame,
	// and a list whose strategies all have to agree is StrategyModeAll.
	Stop
)

// UserTurnResumedParams describes the user speaking again in an open turn.
type UserTurnResumedParams struct {
	// EnableInterruptions reports whether the turn interrupts the bot, as
	// decided when it started.
	EnableInterruptions bool
	// InferenceTriggered reports whether the turn's inference was already
	// triggered. The new speech voids the answer it started.
	InferenceTriggered bool
}

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
	// ConfirmsSpeculation reports that this turn end confirms a speculative
	// reply that was already generated. The reply is waiting on the turn frame
	// this emits, so running the model again would answer the same turn twice.
	ConfirmsSpeculation bool
}

// UserTurnSpeculation is a speculative inference a stop strategy has in flight.
//
// It is produced from an eager end of turn: the turn is not over yet, so the
// inference runs against a provisional conversation and its reply is held back
// until the turn is confirmed.
type UserTurnSpeculation struct {
	// Text is the user turn text the inference was run against.
	Text string
	// Perception is how the user said it, as read at the eager end of turn,
	// or nil. Part of expressive conversation (alpha).
	Perception []expressive.PerceptionSignal
}

// DefaultStoppedParams is the params a typical stop strategy uses.
func DefaultStoppedParams() UserTurnStoppedParams {
	return UserTurnStoppedParams{EnableUserSpeakingFrames: true}
}

// Emitter lets a controller or strategy push frames into the pipeline. The
// UserTurnProcessor implements it. Broadcast sends a frame both downstream and
// upstream, which is how turn decisions and interruptions reach the whole
// pipeline.
type Emitter interface {
	// Push sends a frame to the neighbor in dir.
	Push(ctx context.Context, f frames.Frame, dir processor.Direction) error
	// Broadcast builds one frame per direction and sends them both ways. It takes
	// a constructor rather than a frame because the two directions must not share
	// one instance: each is processed on its own goroutine, and a frame is owned
	// by a single goroutine at a time.
	Broadcast(ctx context.Context, build func() frames.Frame) error
}
