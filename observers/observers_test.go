package observers_test

import (
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

func TestTurnTrackingStartAndInterruption(t *testing.T) {
	var mu sync.Mutex
	var started []int
	var ended []struct {
		turn        int
		interrupted bool
	}
	o := observers.NewTurnTracking(observers.TurnTrackingConfig{
		OnTurnStarted: func(turn int) {
			mu.Lock()
			started = append(started, turn)
			mu.Unlock()
		},
		OnTurnEnded: func(turn int, _ time.Duration, interrupted bool) {
			mu.Lock()
			ended = append(ended, struct {
				turn        int
				interrupted bool
			}{turn, interrupted})
			mu.Unlock()
		},
	})

	// Pipeline start opens turn 1.
	push(o, frames.NewStartFrame(), processor.Downstream)
	// Bot answers turn 1.
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream)
	// User barges in: turn 1 ends interrupted, turn 2 starts.
	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	mu.Lock()
	defer mu.Unlock()
	if len(started) != 2 || started[0] != 1 || started[1] != 2 {
		t.Fatalf("started turns = %v, want [1 2]", started)
	}
	if len(ended) != 1 || ended[0].turn != 1 || !ended[0].interrupted {
		t.Fatalf("ended = %+v, want turn 1 interrupted", ended)
	}
}

func TestUserBotLatency(t *testing.T) {
	var got time.Duration
	var n int
	o := observers.NewUserBotLatency(observers.LatencyConfig{
		OnLatency: func(d time.Duration) { got = d; n++ },
	})
	push(o, frames.NewUserStoppedSpeakingFrame(), processor.Downstream)
	time.Sleep(10 * time.Millisecond)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream)

	if n != 1 {
		t.Fatalf("OnLatency called %d times, want 1", n)
	}
	if got < 10*time.Millisecond {
		t.Fatalf("latency = %s, want >= 10ms", got)
	}
}

func TestStartupTimingFiresOnce(t *testing.T) {
	var n int
	o := observers.NewStartupTiming(observers.StartupConfig{
		OnStartup: func(time.Duration) { n++ },
	})
	push(o, frames.NewStartFrame(), processor.Downstream)
	push(o, frames.NewTTSAudioRawFrame([]byte{0, 0}, 24000, 1), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream)
	if n != 1 {
		t.Fatalf("OnStartup called %d times, want 1", n)
	}
}

// TestBroadcastSiblingCountedOnce pins the invariant that a broadcast pair — two
// distinct frames, one per direction — drives a single turn transition.
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

// push reports one frame to an observer the way a processor handover does.
func push(o processor.Observer, f frames.Frame, dir processor.Direction) {
	o.OnPushFrame(processor.FramePushed{Frame: f, Direction: dir})
}
