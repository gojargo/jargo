package observers_test

import (
	"bytes"
	"log/slog"
	"strings"
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

// TestLoggerLogsFramesItIsGiven covers the debugging observer: it reports every
// frame that reaches it, at the level it was configured with.
func TestLoggerLogsFramesItIsGiven(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewLogger(observers.LoggerConfig{
		Logger: slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})),
		Level:  slog.LevelDebug,
	})

	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)

	if got := buf.String(); !strings.Contains(got, "UserStartedSpeakingFrame") {
		t.Errorf("logged %q, want it to name the frame", got)
	}
}

// TestLoggerRespectsItsFilter covers the reason the filter exists: a pipeline
// pushes far too many frames to log them all, so a caller narrows it to the ones
// being investigated.
func TestLoggerRespectsItsFilter(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewLogger(observers.LoggerConfig{
		Logger: slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})),
		Level:  slog.LevelDebug,
		Filter: func(f frames.Frame) bool {
			_, want := f.(*frames.BotStartedSpeakingFrame)
			return want
		},
	})

	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	if buf.Len() != 0 {
		t.Errorf("logged a frame the filter rejected: %q", buf.String())
	}

	push(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream)
	if got := buf.String(); !strings.Contains(got, "BotStartedSpeakingFrame") {
		t.Errorf("logged %q, want the frame the filter accepted", got)
	}
}
