package observers_test

import (
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

// push reports one frame to an observer the way a processor handover does.
func push(o processor.Observer, f frames.Frame, dir processor.Direction) {
	pushAt(o, f, dir, 0)
}

// pushAt reports one frame as having been handed over at the given point on the
// pipeline clock, which is what the observers time turns against.
func pushAt(o processor.Observer, f frames.Frame, dir processor.Direction, at time.Duration) {
	o.OnPushFrame(processor.FramePushed{Frame: f, Direction: dir, Timestamp: at})
}

// TestBroadcastSiblingCountedOnce pins the invariant that a broadcast pair (two
// distinct frames, one per direction) drives a single turn transition.
//
// The halves carry different ids by design, so the id deduper cannot pair them;
// only BroadcastSiblingID can. TurnTracking's state machine happens to absorb a
// duplicate barge-in on its own, so this test passes with or without the sibling
// guard: it documents the contract every observer relies on rather than catching
// a regression in this one.
func TestBroadcastSiblingCountedOnce(t *testing.T) {
	var started, ended int
	o := observers.NewTurnTracking(observers.TurnTrackingConfig{
		OnTurnStarted: func(int) { started++ },
		OnTurnEnded:   func(int, time.Duration, bool) { ended++ },
	})

	// Turn one begins with the pipeline and the bot starts speaking.
	push(o, frames.NewStartFrame(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	started, ended = 0, 0

	// The user barges in. UserTurnProcessor.Broadcast builds one frame per
	// direction and pairs them, so both halves reach the observer.
	down := frames.NewUserStartedSpeakingFrame()
	up := frames.NewUserStartedSpeakingFrame()
	down.SetBroadcastSiblingID(up.ID())
	up.SetBroadcastSiblingID(down.ID())

	push(o, down, processor.Downstream)
	push(o, up, processor.Upstream)

	if ended != 1 || started != 1 {
		t.Errorf("turn transitions = %d ended / %d started, want exactly 1 each", ended, started)
	}
}
