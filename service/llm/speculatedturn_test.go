package llm_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// TestTheSpeculatedTurnPassesTheLLMBeforeItsResponse checks the turn an eager
// end of turn answers reaches the processors after the LLM ahead of the reply,
// which the LLM holds until the turn is confirmed.
func TestTheSpeculatedTurnPassesTheLLMBeforeItsResponse(t *testing.T) {
	convo := frames.NewLLMContext("Be brief.")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.EagerStrategies(turns.EagerStrategiesConfig{}),
	}))
	svc := llm.New("EchoLLM", stampGen{})

	var (
		mu    sync.Mutex
		order []string
	)
	task := pipeline.NewWorker(pipeline.New(pair.User(), svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.LLMContextUserTurnFrame:
			if fr.Speculation {
				order = append(order, "speculated turn")
			} else {
				order = append(order, "turn")
			}
		case *frames.LLMFullResponseStartFrame:
			order = append(order, "response")
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for _, f := range []frames.Frame{
		frames.NewProposedUserStartedSpeakingFrame(),
		frames.NewEagerTranscriptionFrame("book a flight", "u", "ts"),
		frames.NewTranscriptionFrame("book a flight", "u", "ts"),
		frames.NewProposedUserStoppedSpeakingFrame(),
	} {
		task.QueueFrame(f)
		time.Sleep(60 * time.Millisecond)
	}
	time.Sleep(800 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if len(order) < 2 || order[0] != "speculated turn" || order[1] != "response" {
		t.Errorf("order = %v, want the speculated turn and then the response", order)
	}
}
