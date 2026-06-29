package turns_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/turns"
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

func runTurns(t *testing.T, p *turns.UserTurnProcessor) (*recorder, *pipeline.Task, chan error) {
	t.Helper()
	rec := newRecorder()
	task := pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{OnReachedDownstream: rec.onDown})
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
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{stop},
		},
		StopTimeout: 2 * time.Second,
	})
	rec, task, done := runTurns(t, p)

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
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			// A stop strategy that never fires on its own.
			Stop: []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 40 * time.Millisecond,
	})
	rec, task, done := runTurns(t, p)

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
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: fakeTurn{}})},
		},
		StopTimeout: 2 * time.Second,
	})
	rec, task, done := runTurns(t, p)

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	rec.expect(t, "started")
	rec.expect(t, "interruption")

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, ""))
	task.QueueFrame(finalTranscript("hello there"))
	rec.expect(t, "stopped")

	finish(t, task, done)
}

// TestDeferredFinalization covers deferred(): the wrapped detector triggers
// inference but cannot finalize; only the completion strategy finalizes.
func TestDeferredFinalization(t *testing.T) {
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop: []turns.StopStrategy{
				turns.Deferred(turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: fakeTurn{}})),
				turns.NewExternalCompletionStop(),
			},
		},
		StopTimeout: 2 * time.Second,
	})
	rec, task, done := runTurns(t, p)

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
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		MuteStrategies: []turns.MuteStrategy{turns.NewAlwaysUserMute()},
		StopTimeout:    2 * time.Second,
	})
	rec, task, done := runTurns(t, p)

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
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		IdleTimeout: 30 * time.Millisecond,
		OnIdle: func(context.Context, *turns.UserIdleController) error {
			fired <- struct{}{}
			return nil
		},
	})
	_, task, done := runTurns(t, p)

	task.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	select {
	case <-fired:
	case <-time.After(2 * time.Second):
		t.Fatal("idle did not fire")
	}
	finish(t, task, done)
}
