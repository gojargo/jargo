package aggregators_test

// Tests for the turn strategies a service recommends through its metadata, and
// for the proposals those strategies resolve. Ported from upstream's user
// aggregator tests.

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/events"
)

// runPair starts a task around the given processors and collects the frames that
// reach the end of the pipeline.
func runPair(t *testing.T, ps ...processor.Processor) (*pipeline.Worker, chan frames.Frame, chan error) {
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

// awaitMatching waits for want frames the predicate accepts to reach the end of the
// pipeline, and reports how many arrived. It returns as soon as the count is
// reached, so a test asserting frames do arrive does not pay the timeout.
func awaitMatching(seen chan frames.Frame, match func(frames.Frame) bool, want int) int {
	n := 0
	deadline := time.After(3 * time.Second)
	for n < want {
		select {
		case f := <-seen:
			if match(f) {
				n++
			}
		case <-deadline:
			return n
		}
	}
	return n
}

// drainCount counts the matching frames that arrive over a grace period. It is
// for asserting frames do *not* arrive, where there is nothing to wait for.
func drainCount(seen chan frames.Frame, match func(frames.Frame) bool) int {
	n := 0
	deadline := time.After(time.Second)
	for {
		select {
		case f := <-seen:
			if match(f) {
				n++
			}
		case <-deadline:
			return n
		}
	}
}

// isProposal accepts either half of a proposed user turn.
func isProposal(f frames.Frame) bool {
	switch f.(type) {
	case *frames.ProposedUserStartedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame:
		return true
	}
	return false
}

func is[T frames.Frame](f frames.Frame) bool {
	_, ok := f.(T)
	return ok
}

// A service that does its own end-of-turn detection recommends external
// strategies through its metadata, and the aggregator adopts them when the
// application configured none of its own.
func TestServiceRecommendedTurnStrategiesAreApplied(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{}))
	task, seen, runDone := runPair(t, pair.User())
	defer func() { task.StopWhenDone(); <-runDone }()

	md := frames.NewSTTMetadataFrame(0)
	md.ServiceName = "some-stt"
	md.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{})
	task.QueueFrame(md)

	// The service now drives the turn itself, announcing both boundaries. The
	// adopted strategies take that decision as made: the turn runs the model
	// without the aggregator announcing it a second time or barging in.
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewTranscriptionFrame("hello there", "u", "ts"))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	var started, interrupted, ran int
	deadline := time.After(3 * time.Second)
	for ran == 0 {
		select {
		case f := <-seen:
			switch f.(type) {
			case *frames.UserStartedSpeakingFrame:
				started++
			case *frames.InterruptionFrame:
				interrupted++
			case *frames.LLMContextFrame:
				ran++
			}
		case <-deadline:
			t.Fatal("the externally-driven turn produced no LLMContextFrame")
		}
	}
	// The defaults would have opened the turn on the transcript and barged in
	// there. Adopting means neither happens.
	if interrupted != 0 {
		t.Errorf("InterruptionFrames = %d, want 0: the recommendation was not adopted", interrupted)
	}
	if started != 1 {
		t.Errorf("UserStartedSpeakingFrames = %d, want just the one the service sent", started)
	}
}

// Strategies the application configured always win, so the recommendation is
// ignored and the turn is still driven by them.
func TestConfiguredTurnStrategiesOverrideTheServiceRecommendation(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 30 * time.Second,
	}))
	task, seen, runDone := runPair(t, pair.User())
	defer func() { task.StopWhenDone(); <-runDone }()

	md := frames.NewSTTMetadataFrame(0)
	md.ServiceName = "some-stt"
	md.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{})
	task.QueueFrame(md)

	// The signals the recommended strategies would have acted on. The configured
	// ones ignore them, so nothing runs the model.
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewTranscriptionFrame("hello there", "u", "ts"))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	if drainCount(seen, is[*frames.LLMContextFrame]) != 0 {
		t.Error("the recommendation overruled the strategies the application configured")
	}
}

// A proposal leaves the decision with the aggregator, so it announces the turn
// and barges in itself.
func TestProposalsProduceTurnFramesAndAnInterruption(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	}))
	task, seen, runDone := runPair(t, pair.User())
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewProposedUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewTranscriptionFrame("Hello!", "u", "ts"))
	task.QueueFrame(frames.NewProposedUserStoppedSpeakingFrame())

	var started, stopped, interrupted, ran, proposals int
	deadline := time.After(2 * time.Second)
	for ran == 0 {
		select {
		case f := <-seen:
			switch f.(type) {
			case *frames.UserStartedSpeakingFrame:
				started++
			case *frames.UserStoppedSpeakingFrame:
				stopped++
			case *frames.InterruptionFrame:
				interrupted++
			case *frames.LLMContextFrame:
				ran++
			case *frames.ProposedUserStartedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame:
				proposals++
			}
		case <-deadline:
			t.Fatalf("the proposed turn never ran the model (started=%d stopped=%d)", started, stopped)
		}
	}
	if started != 1 {
		t.Errorf("UserStartedSpeakingFrames = %d, want 1", started)
	}
	if interrupted != 1 {
		t.Errorf("InterruptionFrames = %d, want 1", interrupted)
	}
	// The resolver owns the proposal, so it does not travel on to a second one.
	if proposals != 0 {
		t.Errorf("proposals forwarded = %d, want 0 (the resolver consumes them)", proposals)
	}
}

// The emitter already announced this turn, so the aggregator adopts it and
// emits nothing of its own.
func TestRealTurnFramesAreAdoptedWithoutReEmission(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	}))
	task, seen, runDone := runPair(t, pair.User())
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewTranscriptionFrame("Hello!", "u", "ts"))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	var started, stopped, interrupted, ran int
	deadline := time.After(2 * time.Second)
	for ran == 0 {
		select {
		case f := <-seen:
			switch f.(type) {
			case *frames.UserStartedSpeakingFrame:
				started++
			case *frames.UserStoppedSpeakingFrame:
				stopped++
			case *frames.InterruptionFrame:
				interrupted++
			case *frames.LLMContextFrame:
				ran++
			}
		case <-deadline:
			t.Fatal("the adopted turn never ran the model")
		}
	}
	// Only the two frames that were sent, passed through: no second pair and no
	// interruption.
	if started != 1 || stopped != 1 {
		t.Errorf("speaking frames = %d started / %d stopped, want 1 each", started, stopped)
	}
	if interrupted != 0 {
		t.Errorf("InterruptionFrames = %d, want 0 (the emitter owns the interruption)", interrupted)
	}
}

// Nothing here resolves a proposal, so it travels on for a resolver further
// along to decide, and the configured strategies drive the turn.
func TestProposalsTravelOnWhenNothingResolvesThem(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 30 * time.Second,
	}))
	task, seen, runDone := runPair(t, pair.User())
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewProposedUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewProposedUserStoppedSpeakingFrame())

	if n := awaitMatching(seen, isProposal, 2); n != 2 {
		t.Errorf("proposals forwarded = %d, want 2 (nothing here resolves them)", n)
	}
}

// A service sitting between the two halves broadcasts a copy each way, so a
// proposal reaches the assistant half too. It stops there when the user half is
// the one resolving it.
func TestProposalsAreConsumedByTheAssistantHalfWhenTheUserHalfResolves(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	}))
	task, seen, runDone := runPair(t, pair.Assistant())
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewProposedUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewProposedUserStoppedSpeakingFrame())

	if n := drainCount(seen, isProposal); n != 0 {
		t.Errorf("proposals forwarded = %d, want 0 (the user half resolves them)", n)
	}
}

// With nothing resolving them, the assistant half lets them travel on.
func TestProposalsPassTheAssistantHalfWhenTheUserHalfIgnoresThem(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 30 * time.Second,
	}))
	task, seen, runDone := runPair(t, pair.Assistant())
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewProposedUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewProposedUserStoppedSpeakingFrame())

	if n := awaitMatching(seen, isProposal, 2); n != 2 {
		t.Errorf("proposals forwarded = %d, want 2 (nothing resolves them)", n)
	}
}

// turnDetectingSTT stands in for a transcription service whose provider reports
// the turn boundaries. It pushes the final transcript and then proposes the
// stop, which is the order such a service sends them in.
type turnDetectingSTT struct {
	*processor.Base
}

func newTurnDetectingSTT() *turnDetectingSTT {
	p := &turnDetectingSTT{}
	p.Base = processor.New("TurnDetectingSTT", p)
	return p
}

func (p *turnDetectingSTT) ProcessFrame(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := p.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.InterimTranscriptionFrame); !ok {
		return nil
	}
	tx := frames.NewTranscriptionFrame("Hello!", "u", "ts")
	if err := p.PushFrame(ctx, tx, processor.Downstream); err != nil {
		return err
	}
	return p.Broadcast(ctx, func() frames.Frame {
		return frames.NewProposedUserStoppedSpeakingFrame()
	})
}

// A service that pushes the transcript before proposing the stop closes the turn
// on the stop signal itself.
//
// Both timers are set far longer than the test runs, so nothing but the stop
// signal can close the turn. That path needs the transcript to have arrived
// first, which holds only while the proposal stays ordered behind it.
func TestTurnClosesOnTheStopSignalWhenTheTranscriptPrecedesIt(t *testing.T) {
	convo := frames.NewLLMContext("system")
	interrupt := true
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{
				turns.NewExternalStart(turns.ExternalStartConfig{EnableInterruptions: &interrupt}),
			},
			Stop: []turns.StopStrategy{
				turns.NewExternalStop(turns.ExternalStopConfig{Timeout: 30 * time.Second}),
			},
		},
		StopTimeout: 30 * time.Second,
	}))
	task, seen, runDone := runPair(t, newTurnDetectingSTT(), pair.User())
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(frames.NewProposedUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewInterimTranscriptionFrame("Hel", "u", "ts"))

	if n := awaitMatching(seen, is[*frames.LLMContextFrame], 1); n != 1 {
		t.Fatal("the turn never closed on the stop signal, so only the aggregation timer was left to close it")
	}
}
