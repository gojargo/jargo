package turns_test

// Tests for the UserTurnProcessor: the turn decision made in a processor of its
// own, so it can be shared. Ported from upstream's user turn processor tests.

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/events"
)

// runProcessor starts a task around the given processors and collects what
// reaches the end of the pipeline.
func runProcessor(t *testing.T, ps ...processor.Processor) (*pipeline.Worker, chan frames.Frame, chan error) {
	t.Helper()
	seen := make(chan frames.Frame, 64)
	task := pipeline.NewWorker(pipeline.New(ps...), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, seen, runDone
}

// await waits for a frame the predicate accepts.
func await(seen chan frames.Frame, match func(frames.Frame) bool) bool {
	deadline := time.After(3 * time.Second)
	for {
		select {
		case f := <-seen:
			if match(f) {
				return true
			}
		case <-deadline:
			return false
		}
	}
}

// The default strategies open the turn on the VAD and close it once the speech
// timeout has elapsed with a transcript in hand.
func TestUserTurnProcessorDefaultStrategies(t *testing.T) {
	p := turns.NewUserTurnProcessor(turns.Config{})

	started := make(chan turns.StartStrategy, 1)
	stopped := make(chan turns.StopStrategy, 1)
	events.On(p.Events(), turns.EventUserTurnStarted, func(_ context.Context, s turns.StartStrategy) {
		select {
		case started <- s:
		default:
		}
	})
	events.On(p.Events(), turns.EventUserTurnStopped, func(_ context.Context, s turns.StopStrategy) {
		select {
		case stopped <- s:
		default:
		}
	})

	task, seen, runDone := runProcessor(t, p)
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	if !await(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.UserStartedSpeakingFrame)
		return ok
	}) {
		t.Fatal("the VAD start never opened a turn")
	}
	select {
	case s := <-started:
		if _, ok := s.(*turns.VADStart); !ok {
			t.Errorf("the turn was opened by %T, want the VAD strategy", s)
		}
	case <-time.After(time.Second):
		t.Fatal("the turn start was never reported")
	}

	task.QueueFrame(frames.NewTranscriptionFrame("hello there", "u", "ts"))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))

	if !await(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.UserStoppedSpeakingFrame)
		return ok
	}) {
		t.Fatal("the turn never closed")
	}
	select {
	case s := <-stopped:
		if _, ok := s.(*turns.SpeechTimeoutStop); !ok {
			t.Errorf("the turn was closed by %T, want the speech-timeout strategy", s)
		}
	case <-time.After(time.Second):
		t.Fatal("the turn stop was never reported")
	}
}

// A proposal leaves the decision here, so the processor announces the turn and
// barges in itself, and the proposal does not travel on to a second resolver.
func TestUserTurnProcessorResolvesProposals(t *testing.T) {
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	})
	task, seen, runDone := runProcessor(t, p)
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewProposedUserStartedSpeakingFrame())

	var started, interrupted, forwarded int
	deadline := time.After(3 * time.Second)
	for started == 0 || interrupted == 0 {
		select {
		case f := <-seen:
			switch f.(type) {
			case *frames.UserStartedSpeakingFrame:
				started++
			case *frames.InterruptionFrame:
				interrupted++
			case *frames.ProposedUserStartedSpeakingFrame:
				forwarded++
			}
		case <-deadline:
			t.Fatalf("the proposal was not resolved (started=%d interrupted=%d)", started, interrupted)
		}
	}
	if forwarded != 0 {
		t.Errorf("the proposal was forwarded %d times; the resolver consumes it", forwarded)
	}
}

// Nothing here resolves proposals, so they travel on for a resolver further
// along to decide.
func TestUserTurnProcessorForwardsUnresolvedProposals(t *testing.T) {
	p := turns.NewUserTurnProcessor(turns.Config{})
	task, seen, runDone := runProcessor(t, p)
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewProposedUserStartedSpeakingFrame())
	if !await(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.ProposedUserStartedSpeakingFrame)
		return ok
	}) {
		t.Error("an unresolved proposal must travel on")
	}
}

// The watchdog closes a turn no strategy closed, and says so.
func TestUserTurnProcessorStopTimeout(t *testing.T) {
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 200 * time.Millisecond,
	})

	timedOut := make(chan struct{}, 1)
	events.OnSignal(p.Events(), turns.EventUserTurnStopTimeout, func(context.Context) {
		select {
		case timedOut <- struct{}{}:
		default:
		}
	})

	task, seen, runDone := runProcessor(t, p)
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))

	select {
	case <-timedOut:
	case <-time.After(3 * time.Second):
		t.Fatal("a turn nothing closed was never timed out")
	}
	if !await(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.UserStoppedSpeakingFrame)
		return ok
	}) {
		t.Error("the timed-out turn was never announced as closed")
	}
}

// The end-of-turn parameters the pipeline runs under are published when it
// starts, so a processor downstream can size its own behavior to them and
// clients and observers can mirror them.
func TestTurnAnalyzerStopPublishesItsParams(t *testing.T) {
	p := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop: []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{
				Analyzer: stubAnalyzer{},
			})},
		},
	})
	task, seen, runDone := runProcessor(t, p)
	defer func() { task.StopWhenDone(); <-runDone }()

	if !await(seen, func(f frames.Frame) bool {
		sc, ok := f.(*frames.SpeechControlParamsFrame)
		return ok && sc.TurnParams != nil && sc.TurnParams.StopSecs == turn.DefaultParams().StopSecs
	}) {
		t.Error("the end-of-turn parameters were never published")
	}
}

// stubAnalyzer is a turn analyzer that judges nothing, for the parameters it
// carries rather than the verdicts it reaches.
type stubAnalyzer struct{}

func (stubAnalyzer) SetSampleRate(int)                            {}
func (stubAnalyzer) AppendAudio([]byte, bool) turn.EndOfTurnState { return turn.Incomplete }
func (stubAnalyzer) AnalyzeEndOfTurn() (turn.EndOfTurnState, float64, error) {
	return turn.Incomplete, 0, nil
}
func (stubAnalyzer) SpeechTriggered() bool      { return false }
func (stubAnalyzer) UpdateVADStartSecs(float64) {}
func (stubAnalyzer) Params() turn.Params        { return turn.DefaultParams() }
func (stubAnalyzer) Clear()                     {}
func (stubAnalyzer) Close() error               { return nil }
