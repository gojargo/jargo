package llmworker_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/workers/llmworker"
)

// The context worker is an LLM worker that keeps a conversation of its own,
// rather than sharing the one a transport pipeline carries. What it saves a
// caller is wiring the aggregator pair around the model, so what is worth
// testing is that the wiring is really there.

// runContextWorker stands up a context worker and runs it.
func runContextWorker(t *testing.T, cfg llmworker.ContextConfig) *llmworker.ContextWorker {
	t.Helper()
	ctx := t.Context()
	off := false
	cfg.Name = "context-worker"
	cfg.LLM = newRecordingLLM()
	cfg.WorkerConfig.EnableRTVI = &off
	cfg.WorkerConfig.EnableTurnTracking = &off
	cfg.WorkerConfig.IdleTimeout = -1
	on := true
	cfg.Active = &on

	w := llmworker.NewContext(cfg)
	msgBus := bus.NewAsyncQueueBus()
	msgBus.Start(ctx)
	t.Cleanup(msgBus.Stop)
	w.Attach(ctx, registry.New("test-runner"), msgBus.Bus)

	runDone := make(chan error, 1)
	go func() { runDone <- w.Run(ctx) }()
	t.Cleanup(func() {
		w.StopWhenDone()
		select {
		case <-runDone:
		case <-time.After(5 * time.Second):
			t.Error("the worker did not finish")
		}
	})
	return w
}

// TestContextWorkerOwnsItsConversation checks the worker keeps a conversation
// of its own, and that the one a caller hands it is the one it keeps.
func TestContextWorkerOwnsItsConversation(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	w := runContextWorker(t, llmworker.ContextConfig{Context: convo})

	if w.Context() != convo {
		t.Error("the worker is not keeping the conversation it was given")
	}
	if w.UserAggregator() == nil || w.AssistantAggregator() == nil {
		t.Error("the worker built no aggregator pair, so nothing gathers the turns")
	}
}

// TestContextWorkerStartsAnEmptyConversation checks a caller that hands it none
// gets one anyway, rather than a worker with nothing to write into.
func TestContextWorkerStartsAnEmptyConversation(t *testing.T) {
	w := runContextWorker(t, llmworker.ContextConfig{})

	if w.Context() == nil {
		t.Fatal("the worker has no conversation")
	}
	if msgs := w.Context().Messages(); len(msgs) != 0 {
		t.Errorf("the conversation starts with %+v, want it empty", msgs)
	}
}

// TestContextWorkerGathersATurn checks the aggregators are really in the
// pipeline: what the model says is gathered into the conversation the worker
// owns, which is the whole reason to use this over a plain LLM worker.
func TestContextWorkerGathersATurn(t *testing.T) {
	w := runContextWorker(t, llmworker.ContextConfig{})
	ctx := context.Background()

	w.QueueFrame(ctx, frames.NewLLMFullResponseStartFrame())
	w.QueueFrame(ctx, frames.NewLLMTextFrame("Bonjour."))
	w.QueueFrame(ctx, frames.NewLLMFullResponseEndFrame())

	if !waitFor(3*time.Second, func() bool {
		for _, m := range w.Context().Messages() {
			if m.Role == frames.RoleAssistant && m.Text == "Bonjour." {
				return true
			}
		}
		return false
	}) {
		t.Errorf("the conversation holds %+v, want the turn gathered into it",
			w.Context().Messages())
	}
}

// TestContextWorkerIsStillAnLLMWorker checks it keeps the holding its base
// gives it, since a context worker's tools face the same ordering problem.
func TestContextWorkerIsStillAnLLMWorker(t *testing.T) {
	w := runContextWorker(t, llmworker.ContextConfig{})

	if w.ToolCallActive() {
		t.Error("a tool was reported running before any call")
	}
	if w.Worker == nil {
		t.Error("the context worker is not built on an LLM worker")
	}
}
