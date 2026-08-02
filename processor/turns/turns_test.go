package turns_test

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
)

// recorder captures the turn-decision frames the processor broadcasts.
type recorder struct{ ch chan string }

func newRecorder() *recorder { return &recorder{ch: make(chan string, 64)} }

func (r *recorder) onDown(f frames.Frame) {
	switch f.(type) {
	case *frames.UserStartedSpeakingFrame:
		r.ch <- "started"
	case *frames.UserStoppedSpeakingFrame:
		r.ch <- "stopped"
	case *frames.InterruptionFrame:
		r.ch <- "interruption"
	}
}

func (r *recorder) expect(t *testing.T, want string) {
	t.Helper()
	select {
	case got := <-r.ch:
		if got != want {
			t.Fatalf("event = %q, want %q", got, want)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %q", want)
	}
}

func (r *recorder) expectNone(t *testing.T, d time.Duration) {
	t.Helper()
	select {
	case got := <-r.ch:
		t.Fatalf("unexpected event %q", got)
	case <-time.After(d):
	}
}

func runTurns(t *testing.T, cfg turns.Config) (*recorder, *pipeline.Task, chan error) {
	t.Helper()
	rec := newRecorder()
	agg := aggregators.New(frames.NewLLMContext("test"), aggregators.WithTurns(cfg))
	task := pipeline.NewTask(pipeline.New(agg.User()), pipeline.TaskParams{OnReachedDownstream: rec.onDown})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return rec, task, done
}

func finish(t *testing.T, task *pipeline.Task, done chan error) {
	t.Helper()
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

func finalTranscript(text string) *frames.TranscriptionFrame {
	f := frames.NewTranscriptionFrame(text, "user", "")
	f.Finalized = true
	return f
}

// TestVADStartSpeechTimeoutStop covers the model-free default flow: VAD onset
// opens the turn (with barge-in), and a silence timer plus a finalized
// transcript closes it.
func TestVADStartSpeechTimeoutStop(t *testing.T) {
	stop := turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: 30 * time.Millisecond})
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{stop},
		},
		StopTimeout: 2 * time.Second,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")
	rec.expect(t, "interruption")

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, ""))
	task.QueueFrame(finalTranscript("hello"))
	rec.expect(t, "stopped")

	finish(t, task, done)
}

// TestWatchdogForceStop covers the stop-timeout watchdog finalizing a turn that
// got stuck open with the user silent.
func TestWatchdogForceStop(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			// A stop strategy that never fires on its own.
			Stop: []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 40 * time.Millisecond,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")
	rec.expect(t, "interruption")
	// User went silent; the watchdog force-stops after StopTimeout.
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, ""))
	rec.expect(t, "stopped")

	finish(t, task, done)
}

// fakeTurn is a turn.Analyzer whose batch analysis always reports Complete.
type fakeTurn struct{}

func (fakeTurn) SetSampleRate(int)                            {}
func (fakeTurn) AppendAudio([]byte, bool) turn.EndOfTurnState { return turn.Incomplete }
func (fakeTurn) AnalyzeEndOfTurn() (turn.EndOfTurnState, float64, error) {
	return turn.Complete, 1, nil
}
func (fakeTurn) UpdateVADStartSecs(float64) {}
func (fakeTurn) Clear()                     {}
func (fakeTurn) Close() error               { return nil }

// TestTurnAnalyzerStop covers the Smart-Turn stop: the model reports complete on
// VAD stop and a finalized transcript closes the turn.
func TestTurnAnalyzerStop(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: fakeTurn{}})},
		},
		StopTimeout: 2 * time.Second,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")
	rec.expect(t, "interruption")

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, ""))
	task.QueueFrame(finalTranscript("hello there"))
	rec.expect(t, "stopped")

	finish(t, task, done)
}

// countingTurn records how many times Clear is called.
type countingTurn struct {
	fakeTurn
	clears atomic.Int64
}

func (c *countingTurn) Clear() { c.clears.Add(1) }

// TestTurnAnalyzerClearedOnStopNotStart verifies the analyzer is cleared when a
// turn ends but not when it begins.
func TestTurnAnalyzerClearedOnStopNotStart(t *testing.T) {
	analyzer := &countingTurn{}
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: analyzer})},
		},
		StopTimeout: 2 * time.Second,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")
	rec.expect(t, "interruption")
	if n := analyzer.clears.Load(); n != 0 {
		t.Fatalf("analyzer cleared %d times on turn start, want 0", n)
	}

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, ""))
	task.QueueFrame(finalTranscript("hello there"))
	rec.expect(t, "stopped")
	if n := analyzer.clears.Load(); n != 1 {
		t.Fatalf("analyzer cleared %d times through turn stop, want 1", n)
	}

	finish(t, task, done)
}

// TestDeferredFinalization covers deferred(): the wrapped detector triggers
// inference but cannot finalize; only the completion strategy finalizes.
func TestDeferredFinalization(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop: []turns.StopStrategy{
				turns.Deferred(turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: fakeTurn{}})),
				turns.NewExternalCompletionStop(),
			},
		},
		StopTimeout: 2 * time.Second,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")
	rec.expect(t, "interruption")

	// The analyzer would finalize, but it is deferred — no stop yet.
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, ""))
	task.QueueFrame(finalTranscript("hello"))
	rec.expectNone(t, 150*time.Millisecond)

	// The completion gate finalizes.
	task.QueueFrame(frames.NewUserTurnInferenceCompletedFrame())
	rec.expect(t, "stopped")

	finish(t, task, done)
}

// TestMuteSuppressesDuringBotSpeech covers an AlwaysUserMute strategy dropping a
// barge-in while the bot speaks and allowing it once the bot stops.
func TestMuteSuppressesDuringBotSpeech(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		MuteStrategies: []turns.MuteStrategy{turns.NewAlwaysUserMute()},
		StopTimeout:    2 * time.Second,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2)) // muted: suppressed
	rec.expectNone(t, 150*time.Millisecond)

	task.QueueFrame(frames.NewBotStoppedSpeakingFrame()) // unmute
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")

	finish(t, task, done)
}

// TestIdleFires covers the idle controller arming on bot-stopped and firing.
func TestIdleFires(t *testing.T) {
	fired := make(chan struct{}, 1)
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		IdleTimeout: 30 * time.Millisecond,
		OnIdle: func(context.Context, *turns.UserIdleController) error {
			fired <- struct{}{}
			return nil
		},
	}
	_, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	select {
	case <-fired:
	case <-time.After(2 * time.Second):
		t.Fatal("idle did not fire")
	}
	finish(t, task, done)
}

// incompleteTurn is an analyzer that judges the turn unfinished.
type incompleteTurn struct{ fakeTurn }

func (incompleteTurn) AnalyzeEndOfTurn() (turn.EndOfTurnState, float64, error) {
	return turn.Incomplete, 0.17, nil
}

// The analyzer's verdict is reported as metrics, so a turn that ended on the
// safety-net timeout can be told from one the analyzer judged unfinished. It is
// the only thing that says which happened.
func TestTurnAnalyzerReportsItsPrediction(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: incompleteTurn{}})},
		},
		StopTimeout: time.Second,
	}
	agg := aggregators.New(frames.NewLLMContext("test"), aggregators.WithTurns(cfg))

	var mu sync.Mutex
	var pred *frames.TurnPrediction
	task := pipeline.NewTask(pipeline.New(agg.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if mf, ok := f.(*frames.MetricsFrame); ok && mf.Turn != nil {
				mu.Lock()
				pred = mf.Turn
				mu.Unlock()
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, ""))
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if pred == nil {
		t.Fatal("no turn prediction was reported")
	}
	if pred.Complete {
		t.Error("prediction reported complete, want incomplete")
	}
	if pred.Probability != 0.17 {
		t.Errorf("probability = %v, want 0.17", pred.Probability)
	}
}

// A verdict can land in the instant before the turn opens: the analyzer judges
// the speech complete on the VAD stop, and the transcript that follows is what
// opens the turn, wiping the verdict along with the rest of the previous turn's
// state. The transcript fallback is what recovers it, so the turn still closes
// on its own rather than hanging until the stop watchdog.
func TestTurnAnalyzerTranscriptWithNoVADStopStillStops(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart(), turns.NewTranscriptionStart(turns.TranscriptionStartConfig{})},
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: fakeTurn{}})},
		},
		StopTimeout: 3 * time.Second,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewSTTMetadataFrame(120 * time.Millisecond))
	// The analyzer judges the speech complete, but no turn is open yet to close.
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.02, ""))
	rec.expectNone(t, 50*time.Millisecond)

	// The transcript opens the turn, discarding that verdict, and must still be
	// what ends it.
	start := time.Now()
	task.QueueFrame(finalTranscript("comment ça va"))
	rec.expect(t, "started")
	rec.expect(t, "interruption")
	rec.expect(t, "stopped")
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("turn took %v to stop, want the STT safety net rather than the %v watchdog", elapsed, cfg.StopTimeout)
	}

	finish(t, task, done)
}

// An interim transcript means more speech is still in flight, so an earlier
// finalized transcript no longer covers all of it. The turn must fall back to
// the STT safety net instead of closing the moment the VAD stops.
func TestTurnAnalyzerInterimReopensFinalizedTranscript(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: fakeTurn{}})},
		},
		StopTimeout: 3 * time.Second,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewSTTMetadataFrame(500 * time.Millisecond))
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")
	rec.expect(t, "interruption")

	// The STT endpointer finalizes on a pause too short for the VAD to call it a
	// stop, then keeps transcribing. Each frame is left to land before the next
	// is queued: VAD frames are system frames and would otherwise overtake the
	// transcripts, which are data frames.
	task.QueueFrame(finalTranscript("hello"))
	rec.expectNone(t, 30*time.Millisecond)
	task.QueueFrame(frames.NewInterimTranscriptionFrame("hello how", "user", ""))
	rec.expectNone(t, 30*time.Millisecond)

	// The VAD stop must no longer find a finalized transcript to close on.
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.02, ""))
	rec.expectNone(t, 200*time.Millisecond)

	// The tail arrives and closes the turn.
	task.QueueFrame(finalTranscript("hello how are you"))
	rec.expect(t, "stopped")

	finish(t, task, done)
}

// A transcript can arrive with no VAD stop behind it, from speech soft enough
// that the VAD never bracketed it. Nothing will start the silence timers in that
// case, so the strategy measures inactivity from the transcript itself rather
// than leaving the turn to the stop watchdog.
func TestSpeechTimeoutTranscriptWithNoVADStopStillStops(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewTranscriptionStart(turns.TranscriptionStartConfig{})},
			Stop: []turns.StopStrategy{
				turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: 30 * time.Millisecond}),
			},
		},
		StopTimeout: 3 * time.Second,
	}
	rec, task, done := runTurns(t, cfg)

	start := time.Now()
	task.QueueFrame(finalTranscript("hello there"))
	rec.expect(t, "started")
	rec.expect(t, "interruption")
	rec.expect(t, "stopped")
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("turn took %v to stop, want the speech timeout rather than the %v watchdog", elapsed, cfg.StopTimeout)
	}

	finish(t, task, done)
}

// A strategy can decide a turn is over on a signal that resolves late, by which
// time the user has started speaking again. Finalizing then would cut them off,
// so the turn stays open and the watchdog closes it once they fall silent.
func TestNoFinalizeWhileUserIsSpeaking(t *testing.T) {
	cfg := turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 150 * time.Millisecond,
	}
	rec, task, done := runTurns(t, cfg)

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")
	rec.expect(t, "interruption")

	// The verdict lands while the user is still audibly speaking.
	task.QueueFrame(frames.NewUserTurnInferenceCompletedFrame())
	rec.expectNone(t, 80*time.Millisecond)

	// Once they stop, the watchdog finalizes the turn that was held open.
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, ""))
	rec.expect(t, "stopped")

	finish(t, task, done)
}
