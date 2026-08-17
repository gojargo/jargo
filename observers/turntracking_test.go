package observers_test

import (
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

// turnEnd is one recorded end of a turn.
type turnEnd struct {
	turn        int
	interrupted bool
}

// turnRecorder collects the turn boundaries an observer reports, from whichever
// goroutine reports them: the turn-end timer fires on its own.
type turnRecorder struct {
	mu      sync.Mutex
	started []int
	ended   []turnEnd
	endSig  chan struct{}
}

func newTurnRecorder() *turnRecorder {
	return &turnRecorder{endSig: make(chan struct{}, 8)}
}

func (r *turnRecorder) config(timeout time.Duration) observers.TurnTrackingConfig {
	return observers.TurnTrackingConfig{
		TurnEndTimeout: timeout,
		OnTurnStarted: func(turn int) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.started = append(r.started, turn)
		},
		OnTurnEnded: func(turn int, _ time.Duration, interrupted bool) {
			r.mu.Lock()
			r.ended = append(r.ended, turnEnd{turn, interrupted})
			r.mu.Unlock()
			select {
			case r.endSig <- struct{}{}:
			default:
			}
		},
	}
}

func (r *turnRecorder) snapshot() ([]int, []turnEnd) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]int(nil), r.started...), append([]turnEnd(nil), r.ended...)
}

// waitForEnd blocks until a turn ends, failing if none does.
func (r *turnRecorder) waitForEnd(t *testing.T) {
	t.Helper()
	select {
	case <-r.endSig:
	case <-time.After(3 * time.Second):
		t.Fatal("the turn never ended")
	}
}

// turnEndTimeoutTest is short enough to wait out in a test.
const turnEndTimeoutTest = 100 * time.Millisecond

// TestTurnTrackingEndsATurnAfterTheBotFinishes covers the ordinary end of a
// turn. The bot falling silent does not end it outright, because the bot may
// simply be between utterances; the turn ends once it has stayed silent for the
// turn-end timeout.
func TestTurnTrackingEndsATurnAfterTheBotFinishes(t *testing.T) {
	r := newTurnRecorder()
	o := observers.NewTurnTracking(r.config(turnEndTimeoutTest))

	push(o, frames.NewStartFrame(), processor.Downstream)
	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream)
	push(o, frames.NewBotStoppedSpeakingFrame(), processor.Upstream)

	// The turn is still open: the bot may have more to say.
	if _, ended := r.snapshot(); len(ended) != 0 {
		t.Fatalf("ended = %+v, want none yet: the bot had only just stopped", ended)
	}

	r.waitForEnd(t)

	_, ended := r.snapshot()
	if len(ended) != 1 {
		t.Fatalf("ended = %+v, want one", ended)
	}
	if ended[0].interrupted {
		t.Error("the turn ended as interrupted, want a normal end: nobody barged in")
	}
}

// TestTurnTrackingRearmsWhileTheBotSpeaksInBursts covers a bot that pauses
// between utterances. Each new utterance cancels the pending end, so a reply
// delivered in pieces is one turn rather than several.
func TestTurnTrackingRearmsWhileTheBotSpeaksInBursts(t *testing.T) {
	r := newTurnRecorder()
	o := observers.NewTurnTracking(r.config(turnEndTimeoutTest))

	push(o, frames.NewStartFrame(), processor.Downstream)
	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	for range 3 {
		push(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream)
		push(o, frames.NewBotStoppedSpeakingFrame(), processor.Upstream)
		// Less than the timeout, so the next utterance arrives before the turn
		// would have ended.
		time.Sleep(turnEndTimeoutTest / 4)
	}

	if _, ended := r.snapshot(); len(ended) != 0 {
		t.Fatalf("ended = %+v, want none: the bot kept speaking", ended)
	}

	r.waitForEnd(t)

	started, ended := r.snapshot()
	if len(ended) != 1 || ended[0].turn != 1 {
		t.Errorf("ended = %+v, want turn 1 once: the bursts were counted as separate turns", ended)
	}
	if len(started) != 1 {
		t.Errorf("started = %v, want one turn for the whole reply", started)
	}
}

// TestTurnTrackingCountsSpeechBeforeTheBotAsOneTurn covers a user who speaks
// again before getting an answer. The turn is still waiting on the bot, so the
// second utterance belongs to it rather than opening another.
func TestTurnTrackingCountsSpeechBeforeTheBotAsOneTurn(t *testing.T) {
	r := newTurnRecorder()
	o := observers.NewTurnTracking(r.config(turnEndTimeoutTest))

	push(o, frames.NewStartFrame(), processor.Downstream)
	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	started, ended := r.snapshot()
	if len(started) != 1 || started[0] != 1 {
		t.Errorf("started = %v, want just turn 1: the bot had not answered yet", started)
	}
	if len(ended) != 0 {
		t.Errorf("ended = %+v, want none", ended)
	}
}

// TestTurnTrackingFlushesTheOpenTurnOnShutdown covers the pipeline stopping with
// a turn still running. It is reported as interrupted, so a turn in progress is
// accounted for rather than lost.
func TestTurnTrackingFlushesTheOpenTurnOnShutdown(t *testing.T) {
	r := newTurnRecorder()
	o := observers.NewTurnTracking(r.config(turnEndTimeoutTest))

	push(o, frames.NewStartFrame(), processor.Downstream)
	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	push(o, frames.NewCancelFrame(), processor.Downstream)

	_, ended := r.snapshot()
	if len(ended) != 1 {
		t.Fatalf("ended = %+v, want the open turn flushed", ended)
	}
	if !ended[0].interrupted {
		t.Error("the flushed turn was reported as a normal end, want interrupted")
	}
}

// TestTurnTrackingIgnoresShutdownWithNoOpenTurn is the other side of that: with
// nothing running there is nothing to flush, and no turn is invented.
func TestTurnTrackingIgnoresShutdownWithNoOpenTurn(t *testing.T) {
	r := newTurnRecorder()
	o := observers.NewTurnTracking(r.config(turnEndTimeoutTest))

	push(o, frames.NewEndFrame(), processor.Downstream)

	started, ended := r.snapshot()
	if len(started) != 0 || len(ended) != 0 {
		t.Errorf("started = %v, ended = %+v, want neither: no turn was ever open", started, ended)
	}
}

// TestTurnTrackingStartsWithThePipelineAndEndsOnABargeIn covers the two ends of
// the common case: the first turn opens with the pipeline rather than waiting
// for the user, and a user talking over the bot ends that turn as interrupted
// and opens the next.
func TestTurnTrackingStartsWithThePipelineAndEndsOnABargeIn(t *testing.T) {
	r := newTurnRecorder()
	o := observers.NewTurnTracking(r.config(turnEndTimeoutTest))

	push(o, frames.NewStartFrame(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream)
	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	started, ended := r.snapshot()
	if len(started) != 2 || started[0] != 1 || started[1] != 2 {
		t.Fatalf("started turns = %v, want [1 2]", started)
	}
	if len(ended) != 1 || ended[0].turn != 1 || !ended[0].interrupted {
		t.Fatalf("ended = %+v, want turn 1 interrupted", ended)
	}
}

// TestTurnTrackingTimesTurnsOnThePipelineClock covers what a turn's duration
// means: the span between the frames that opened and closed it, taken from the
// pipeline clock rather than from when the observer happened to be told.
func TestTurnTrackingTimesTurnsOnThePipelineClock(t *testing.T) {
	var got time.Duration
	o := observers.NewTurnTracking(observers.TurnTrackingConfig{
		TurnEndTimeout: turnEndTimeoutTest,
		OnTurnEnded:    func(_ int, d time.Duration, _ bool) { got = d },
	})

	pushAt(o, frames.NewStartFrame(), processor.Downstream, time.Second)
	pushAt(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream, 2*time.Second)
	pushAt(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream, 4*time.Second)

	if got != 3*time.Second {
		t.Errorf("turn duration = %s, want 3s: the turn ran from 1s to 4s", got)
	}
}

// TestTurnTrackingEndsATimedOutTurnWhenTheBotFellSilent covers the turn-end
// timeout's effect on the duration. The wait exists to tell a pause between
// utterances apart from the end of the reply; it is not part of the turn, so a
// turn that ends on the timeout is measured to the moment the bot fell silent.
func TestTurnTrackingEndsATimedOutTurnWhenTheBotFellSilent(t *testing.T) {
	r := newTurnRecorder()
	var got time.Duration
	cfg := r.config(turnEndTimeoutTest)
	ended := cfg.OnTurnEnded
	cfg.OnTurnEnded = func(turn int, d time.Duration, interrupted bool) {
		got = d
		ended(turn, d, interrupted)
	}
	o := observers.NewTurnTracking(cfg)

	pushAt(o, frames.NewStartFrame(), processor.Downstream, time.Second)
	pushAt(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream, 2*time.Second)
	pushAt(o, frames.NewBotStoppedSpeakingFrame(), processor.Upstream, 3*time.Second)

	r.waitForEnd(t)

	if got != 2*time.Second {
		t.Errorf("turn duration = %s, want 2s: the wait for more speech is not part of the turn", got)
	}
}
