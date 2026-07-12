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
	o.OnFrame(frames.NewStartFrame(), processor.Downstream)
	// Bot answers turn 1.
	o.OnFrame(frames.NewBotStartedSpeakingFrame(), processor.Upstream)
	// User barges in: turn 1 ends interrupted, turn 2 starts.
	o.OnFrame(frames.NewUserStartedSpeakingFrame(), processor.Downstream)

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
	o.OnFrame(frames.NewUserStoppedSpeakingFrame(), processor.Downstream)
	time.Sleep(10 * time.Millisecond)
	o.OnFrame(frames.NewBotStartedSpeakingFrame(), processor.Upstream)

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
	o.OnFrame(frames.NewStartFrame(), processor.Downstream)
	o.OnFrame(frames.NewTTSAudioRawFrame([]byte{0, 0}, 24000, 1), processor.Downstream)
	o.OnFrame(frames.NewBotStartedSpeakingFrame(), processor.Upstream)
	if n != 1 {
		t.Fatalf("OnStartup called %d times, want 1", n)
	}
}
